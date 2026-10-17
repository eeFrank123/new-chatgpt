# v11：NPU标量小形状与强制内联对照

## 实测依据与假设

v10两个布局均通过12/12，时延接近，不能支持继续改变片上地址即可明显提速的判断。本轮转向小形状固定指令开销。未取得完整测试参数、NPU剖析或生成汇编，额外调用开销与小形状Scalar更快均为待验证假设。

两个推荐包均保留512行/核、自动同步、DMA阈值和相同大形状实现。只在CPP与Kernel CMake的后端默认值上不同：5为Scalar、4为Vector对照。Host没有新增读取输入或计算几何结果。

## 内联依据

官方 [asc-devkit](https://gitcode.com/cann/asc-devkit) 9.0.0分支，固定提交1909ae266695b7d9ce40e62bb928c036719d9146：

- tools/build/asc_op_compile_base/asc_op_compiler/compile_op.py第615行为wrapper添加inline __attribute__((always_inline))。
- 同目录get_op_tiling.py第1136行使用相同属性。
- impl/utils/sys_macros.h的__aicore__是执行空间属性，不保证所有普通inline函数展开。

因此只给项目辅助函数增加集中开关，不修改SDK或kernel入口。尚无生成汇编，不能声称旧版有具体数量的call/栈访问，或已节省特定时长。

官方 [SIMD关键字文档](https://asc.gitcode.com/guide/programming_guide/language_extension/simd_builtin_keywords.html) 区分普通inline和设备强制内联。此页是持续开发版本，CANN9兼容判断优先使用同分支生成器先例。[函数限制页](https://asc.gitcode.com/guide/technical_appendix/cpp_standard_support/syntax_restrictions/functions.html) 的always_inline被忽略规则位于SIMT VF章节，不能直接套用本项目的A2 SIMD辅助函数。

## Scalar数据与同步

两次精确DMA把boxes和anchors放到UB中两个32B对齐段；阈值仅存在时DMA。动态比例沿用4B Tiling字段DMA。float32经MTE2_S后读取；half的原始AoS及可选half阈值所有padding填零，经MTE2_V、一次Cast、V_S后读取float32。

NPU标量单元执行四步float运算及严格大于比较，SetValue写UB，经S_MTE3后精确DMA。half兼容输出以uint16写0x0000/0x3c00，无额外输出Cast。保留InitSocState和最终PIPE_ALL。

SDK900的LocalTensor GetValue/SetValue支持基本类型的标量UB读写，kernel_event.h公开实现V_S和S_MTE3。官方 [Scalar计算单元](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00031.html) 也说明了读取UB及与搬运流水同步的用法。

| UB区域 | 起始字节 | 长度字节 |
| --- | ---: | ---: |
| 原始输入及half阈值附加段 | 0 | 1536 |
| float阈值 | 1536 | 32 |
| 动态比例 | 1568 | 32 |
| half转float结果 | 1600 | 1600 |
| float或uint16输出 | 3200 | 256 |

实际half输入最多768B，加32B阈值块仍位于原始区；Cast最多400个half。输出两种类型共享物理区域，但每个IO模板只启用一种。

## 浮点与设备限制

Scalar函数的Clang pragma禁止contract和reassociate，Kernel参数增加-ffp-contract=off与-fno-fast-math。四步float变量不用volatile，避免主动引入栈存取。没有FMA、代数重排、epsilon或直接以y2替代参考线。

实际源码通过CPU功能模型与独立float32参考比较，覆盖相等及相邻边界、非标准比例、half先转换、消去误差、有限极值和非正规数。模型新增标量UB操作epoch，防止时间戳0使缺少S_MTE3假通过；缺失MTE2_S/V_S/S_MTE3三个负例必须失败。

本地没有Bisheng与Ascend910B，无法确认设备编译诊断、标量float指令降低、设备舍入/非正规数模式或真实时延。API可用不代表新路径已经设备验证。若Scalar变慢，可用相同内联调整的Vector对照分离因素。

不重试明显回退的1024行/核、paired DMA或Scalar GM阈值，不删除必要同步，不启用缺少910B证据的退出DCCI开关，不移动计算到Host。不承诺90分或超过第一名。
