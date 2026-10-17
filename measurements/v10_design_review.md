# v10 设计核查：UB 布局与编译固定开销

日期：2026-10-17。本文记录源码核查与待评测实验，不包含 v10 的 NPU 时延测量，也不保证比赛得分。

## 证据范围

本地检查的官方 SDK 源码位于 `build/api-review/asc-devkit-900`，来源为 [CANN asc-devkit](https://gitcode.com/cann/asc-devkit)，固定提交为 `1909ae266695b7d9ce40e62bb928c036719d9146`。以下 SDK 行号均对应此提交，而不是比赛平台实际生成文件的行号。可使用 `git -C build/api-review/asc-devkit-900 show HEAD:<文件路径>` 复核。

本机没有昇腾 NPU，也没有比赛平台的完整 BiSheng 编译命令或生成汇编。因此，源码、地址空间语法检查和 CPU 模型检查不能替代平台编译、真实指令检查及性能评测。此前提交的 Pass 结果只属于此前版本。

## 已核实的编译与入口行为

| 项目 | 官方 SDK 依据 | 本轮处理 |
| --- | --- | --- |
| V220 内核优化等级 | `tools/build/asc_op_compile_base/asc_op_compiler/ascendc_compile_v220.py:75` 定义 V220 命令生成函数；第 87、89 行默认传入 `-O3`。 | 不能因工程没有显式设置 Release 就认定内核以 O0 编译；不增加未经验证的激进数学选项。实际平台参数仍以平台完整命令为准。 |
| Scalar DCCI 选项 | 同文件第 99 行默认含 `-cce-aicore-dcci-insert-for-scalar=false`。 | 该选项不能等同于移除内核退出 DCCI，也不能由此推断退出时延。 |
| 自动同步 | `tools/build/asc_op_compile_base/asc_op_compiler/compile_op.py:849` 的选项处理函数，在第 851–854 行按配置加入 `--cce-auto-sync` 和 `-api-deps-filter`。 | 两个 v10 包均保持 `SBVG_MANUAL_SYNC_ONLY=OFF`，不把关闭自动同步与地址变化混在同一次实验中。 |
| SoC 状态初始化 | `impl/basic_api/kernel_utils_base.h:52` 的 `InitSocStateImpl`，第 56、58–65 行在 A2 设置 atomic、mask 相关状态。 | 保留入口初始化。源码只有少量状态指令，尚无数据证明其解释了排名中的微秒级差距。 |

项目入口保留一次 `InitSocState()` 与返回前的 `PipeBarrier<PIPE_ALL>()`。内核模板选择 AIV-only，当前实现没有为了这些小张量额外创建 TPipe 或引入跨核同步；不能据此假定编译器没有生成任何其他序言、尾声或栈访问。

## A2 AIV 的 GET_TILING_DATA 路径

SDK 的 `tools/build/asc_op_compile_base/asc_op_compiler/get_op_tiling.py:1142` 生成 `InitTilingData`。第 1145–1151 行的直接 GM 标量复制分支适用于 CPU 调试、特定 Cube 分支或代码通道查询；它不是本算子 A2 AIV 正常执行时的分支。

对于 `__NPU_ARCH__ == 2201` 的实际 AIV 分支，生成逻辑是：

1. 第 1157–1158 行选择 UB 地址 0，并按 32 字节向上取整搬运长度。
2. 第 1166–1168 行通过 `copy_gm_to_ubuf` 从 GM 搬入 UB。
3. 第 1173–1174 行执行 `MTE2 -> S` 事件等待。
4. 第 1179 行从 UB 复制到本地 tiling 结构。
5. 第 1183–1184 行在非 CPU 调试模式执行 `pipe_barrier(PIPE_ALL)`。

因此，不能把通用分支的 GET_TILING_DATA 当成直接由 Scalar Unit 从 GM 读取 tiling，再据此决定插入或删除 DCache 刷新。末尾 ALL 同步也保护了随后输入 DMA 对 UB 地址 0 的复用。本轮保留 SDK 的通用读取路径；静态模板路径继续按原设计避免不必要的 tiling 读取。以上不排除编译器产生其他标量 GM/栈访问。

## 不采用未经 A2 验证的退出 DCCI 开关

官方 [SuperKernel 开发文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00029.html) 讨论了单算子编译末尾 DCache 清理机制及 Scalar GM 访问的一致性要求；该页面列出的 SuperKernel 产品是 A3 与 Atlas 350，不能直接作为 Ascend910B 上禁用退出 DCCI 的兼容性证明。

官方 [ops-nn MR 2495](https://gitcode.com/cann/ops-nn/merge_requests/2495) 的 [变更文件接口](https://gitcode.com/api/v5/repos/cann/ops-nn/pulls/2495/files) 显示，在提交 `cbd1869987ec9cd3c9ddf907b82b59852b774cc5` 的 `scripts/kernel/binary_config/ascendc_config.json` 第 396–399 行，新增的 `-cce-aicore-dcci-before-kernel-end=false` 配置限定在 `ascend950`。即使其中 Conv3DV2 还支持 ascend910b，该开关也只写在 ascend950 的选项中。

SDK `compile_op.py:739–741` 对退出 DCCI 自定义选项的特殊处理，也仅在 C310/M510 判断下调用。当前没有取得这个开关在 910B 上能正确解析、产生预期机器指令并保持缓存一致性的证据。本轮两个提交包均不使用该开关，也不手工增删整缓存 DCCI。不把未确认的退出开销计作已实现收益。

## v10 的可对照改动

官方 [UB Bank 分配优化](https://www.hiascend.com/doc_center/source/en/canncommercial/800/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0025.html) 说明了 bank 与 bank group 冲突，包括同一指令双源落在同一 bank group 时的读冲突，以及向量源、目的落在同一 bank 时的读写冲突。该文档支持研究地址布局；它不证明当前算子的冲突比例或预期提速幅度。

小形状实现采用相同的向量计算函数，通过 `SBVG_BANKED_COMPACT` 只选择 UB 地址：

| UB 区域 | banked 主包，值 0 | compact 对照包，值 1 |
| --- | --- | --- |
| 原始输入、threshold、half 转换区 | 保持低地址 | 相同 |
| float32 的 y1/y2/cy 列首地址 | `65536 + 1568` 字节 | `1568` 字节 |
| 三个 float32 列间距 | 每列 256 字节 | 相同 |
| float32 输出兼数学临时区 | `131072 + 2816` 字节 | `2816` 字节 |
| 运算、mask、比较、选择、搬运和显式同步 | 相同源码序列 | 相同源码序列 |

64 KiB 地址位移用于改变 bank 切片，不能单独保证改变 bank group；列间偏移与每条指令的实际源、目的地址也必须一起检查。两个包都使用 out-of-place 数学临时量，在坐标区与输出区之间交替写入，避免把整个依赖链持续写回同一数组。两个包均保留原浮点运算顺序和严格大于比较。

这里的“相同序列”指源码操作序列；尚未用平台生成汇编确认两包机器指令只差地址。提交对照的目的正是检验布局是否对当前平台有效，不能先把 banked 命名解释为更快。

较大形状的类实现为两包共享：float32 列间距改为 `pitch_ = padded_ | 64U`（元素数）。在 `padded_` 为 64 的倍数时，pitch 保持 64 对齐，并使相邻列的地址差按 512 字节取模为 256 字节，对应上述 bank group 地址模型中的 8 个 32 字节块偏移。half 输入的三个 Cast 彼此独立，统一在其后等待一次 `PIPE_V`；后续数学同样在坐标区和输出区之间交替写入。该改动也是待平台验证的实验，不是已测得的性能提升。

## 同步与语义边界

两个包均保留输入 DMA 后的 `MTE2 -> V`、向量生产者与消费者之间的必要 `PIPE_V`、输出 DMA 前的 `V -> MTE3` 以及入口返回前的 `PIPE_ALL`。half 的三次独立 Cast 共享一个随后的向量屏障，后续读取必须发生在此屏障之后。没有依赖“auto-sync 大概会补齐”而删除现有跨流水事件。

算子继续按 float32 分步执行 `y2-y1`、比例乘法、加 y1、加 threshold 后比较；不引入 FMA、epsilon 或输入值特化。不存在的 threshold 不可访问 GM；输入与输出尾块继续按真实长度搬运。unit ratio 路径保留减法与加法，不直接以 y2 替代可能存在逐步舍入差异的参考线计算。

本轮验证数量、最终压缩包摘要及平台实测结果由整体验证记录补充。需要平台确认的项目包括：两包均编译通过、全部正确性点通过、逐测试点时延，以及 banked 与 compact 的差异是否超过重复提交波动。在取得这些数据前，不宣称达到 90 分或超过第一名。
