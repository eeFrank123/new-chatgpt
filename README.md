# SafetyBeltViolationGeometry · 题目 2135

Ascend 910B 的 Ascend C 几何判定融合算子。比赛提交目录是 `code/`。

用户确认 v5 标准包在平台提交863315中通过12/12，得分50.40、截图排名11；耗时为 `[2.26,1.90,2.55,3.57,2.19,2.70,2.31,1.91,2.49,3.64,2.11,2.62]` μs。完整历史保存在 `measurements/platform_observations.json`。

v6在平台提交864342中编译失败。日志确认CANN9.0.0的opbuild代码生成阶段单独调用GNU11.4.0并指定 `-std=c++11`，没有采用项目的CXX_STANDARD=17。共享头中两个constexpr函数的局部数组不符合C++11规则。

当前v6.1将两个函数改为单条return表达式，保持全部11组映射、16B Tiling布局及v6性能优化。已新增无CANN桩的真实共享头C++11常量求值检查，并将完整Host模型也改为C++11严格编译。此前C++17功能测试没有覆盖这个构建阶段。

v6.1尚未经过真实CANN/NPU完整编译和性能验证，不能声称已超过榜首。第4/10点可能涉及通用非对齐路径，但题目页面只明确第1点B32/ratio1，不能把该推测写成已知测试形状。

## v6 修改

### 非对齐路径只提取真实坐标

部分tile原先把AoS输入补零到64行，再Gather整个补齐范围。v6提前仅将三列紧凑坐标各自最后一个64行块清零，随后用Counter Gather只提取真实行；输入AoS尾部不再填充或读取。清零起点为 `floor(rows/64)*64`，始终32B对齐，先完成预填充再覆盖有效前缀。

这部分Vector初始化与MTE2搬运重叠。完整tile仍使用原有Normal Gather，避免给已对齐大Batch增加计数模式开销。输出字节数32B对齐时用DataCopy，否则保留精确DataCopyPad，不多写GM尾部。

### 小路径减少重复设置

小路径Compare复用四步算术已有的normal mask；float和完整half路径省去仅为half尾块预填充设置的早期Vector屏障。float32 B32/64和float16 B64以完整256B源向量使用Normal Gather，其他小形状继续Counter Gather。所有实际计算依赖屏障和最终PIPE_ALL保留。[Compare官方接口](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0067.html)支持复用mask。

### 常见元数据使用独立内核

B={1,32,33,64,128,129,256,257,512,1024,4096} 与精确 float32 waist_ratio={0.8,0.9,1.0} 可选择 `STATIC_CASE` 编译期参数。在这些内核中，源码不执行 GET_TILING_DATA，直接用常量配置启动输入搬运，避免先读取 GM Tiling 再确定 DMA 长度的串行依赖。

还要求实际每核行数与专用内核一致：B64/128/129/256/257/512分别对应64/128/192/256/320/512行；B1024/4096对应512行/核。B129/257专用路径仍只读写129/257个有效行。可用核数不足或其他分核配置不匹配时自动回到通用内核。所有其他合法B和属性也完整支持。Host和Kernel共享一份constexpr形状映射，测试使用独立字面量映射。

专门化只使用形状、属性和实际分核元数据；坐标和可选 threshold 每次仍从真实输入读取，严格执行题目公式。没有缓存输出或依据固定输入值返回结果。

共有272个dtype/IO_MODE/STATIC_CASE模板组合，其中264个静态、8个通用。构建数量较v5增加，真实平台编译耗时尚未验证。[官方Tiling模板编程](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00025.html)支持此类UINT参数和独立内核。

### 两个小 Batch 计算后端

普通包使用精简后的Gather小路径。`strided`对照包保留v5的B≤64三次多块DataCopyPad方案：直接从框和挂点的对应列提取y1、y2、cy，每个值在UB中占一行32B，然后向量计算，最后由MTE3压缩写回连续输出。该路径不执行Gather。两包的中大Batch路径相同。

half 在四步运算前转换为 float32；不足8行的末尾只在 UB 中补零，避免内存 Compare 读取未初始化向量。GM 不多读或多写。strided 小路径最大显式 UB 为16,576B。跨行搬运会增加 DMA 块数和部分片上流量，是否比 Gather 快需平台比较。[DataCopyPad 官方参数规则](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0265.html)。

### 保留v5的向量初始化与类型转换优化

Select 的 true 值只初始化8个float，再用零源步长广播，替代初始化每行的1.0。half 的三列按实际 padded 行数紧密排列，一次 Cast 转换，替代 v4 的三次 Cast。最大显式 UB 降至 f32 164,480B、half 输出兼容148,096B，另留8KiB SDK空间。

保留 v4 的 AIV-only、K_MAX_SHAPE_DIM=0、ASCENDC_DUMP=0、静态 UB、全部手工同步和最终 PIPE_ALL。两个推荐包均保留编译器自动同步，避免同时改变多个因素。

## 计算与接口

```text
height = fp32(y2) - fp32(y1)
scaled = fp32(waist_ratio) * height
reference = fp32(y1) + scaled
boundary = reference + fp32(threshold)
output = fp32(fp32(cy) > boundary)
```

按四步独立 float32 向量运算执行，不用 FMA、重排、epsilon 或 ratio=1 简化。相等输出0，输出顺序与输入一致。缺省 threshold 仍由 Host 写入0.0的 Tiling 字段；静态内核以同一编译期0.0执行，始终不访问缺省输入地址。核心计算全部在 NPU。

默认输出 float32，同时保留题目原空工程的 half 阈值/输出兼容 IO。其他合法比例、非对齐 Batch、可用核数较少的情况由通用路径处理。

## 提交

1. 先提交修复包 `dist/safetybelt_2135_v6_1_rows512.zip`，替代编译失败的原v6标准包。
2. 如需对比DMA直接提取坐标的小路径，使用 `dist/safetybelt_2135_v6_1_rows512_strided.zip`。

两包的代码只在 Kernel CPP 和 Kernel CMake 的 `SBVG_STRIDED_SMALL` 默认值0/1处不同。两者都保持目标512行/核和自动同步。

**完整替换ZIP中 `code/` 下的11个文件。** v6没有新增文件，仍需要v5新增的 `op_kernel/safety_belt_violation_geometry_strided.h`；Host、TilingKey、Tiling头和Kernel必须一起更新。不要混用旧版库。若平台需要逐文件编辑，请先解压再整体替换。

版本确认：Kernel CPP第一行为 `// SBVG_VERSION: v6_1_cpp11_fix`；随后 `SBVG_STRIDED_SMALL` 默认值普通包0、strided包1。模板头STATIC_CASE列表为0..33，Tiling头两个constexpr函数均仅有一条return，不再定义局部数组。

保存新记录的提交 ID、精确分数及全部12点耗时。若编译失败，保留完整编译日志。旧版 ZIP 保留用于回退。

## 验证

```bash
python tests/verify_cpu_model.py --rows-per-core 512
python tests/verify_cpu_model.py --rows-per-core 512 --strided-small
python tests/verify_cpu_model.py --rows-per-core 4096
python tests/verify_cpu_model.py --rows-per-core 512 --legacy-gather-api
```

每个最终配置检查9,337组输入、6,436,287行精确输出，其中740,800个相等边界；另检查262,144种分区。CPU模型编译实际 Host/Kernel，独立NumPy参考逐元素比较。

每个配置首先使用 `g++ -std=c++11 -pedantic-errors` 编译 `tests/tiling_cpp11_compile.cpp`，静态断言全部11组形状、越界返回0和16B ABI；共享头使用真实源码，不依赖CANN桩。实际Host源码及Host测试使用同样的C++11严格编译，Host API由模型提供。Kernel功能模型继续使用C++17。这些检查仍不等于完整CANN构建。

Host的512配置检查26个注册配置、744个静态派发和1784个通用回退；4096配置检查648个静态派发和1880个回退，两个配置均拒绝19组非法元数据。模型将静态路径的Tiling指针设为空，确认该路径零读取；通用路径每核读取一次。

多块DataCopyPad和高维half/float Cast模型已有独立契约检查，覆盖GM字节步长、UB32B步长、每行隐式复制、压缩写回和尾块。普通接口模式19组契约正例、28种危险操作拒绝检查；旧Gather API模式为27种。v6小路径另经现代/旧版Gather各4608项独立检查，覆盖所有B=1..64；大路径另经3312项、2,765,568行独立检查，包含129/192和257/320静态尾块。

模型还检查GM边界、UB初始化、跨流水与Vector RAW依赖、每行恰好写一次及返回时mask恢复。它不模拟真实异步执行、WAR/WAW、编译器同步或设备指令时延；不能证明真实NPU编译和性能。MinGW缺少ASan/UBSan运行库，使用显式模型断言，未声称通过sanitizer。汇总见 `validation_results.json`。

## 构建与打包

在安装题目对应 ASC SDK、CANN 和 NPU 的 Linux 环境加载 `set_env.sh` 后：

```bash
cmake -S code -B build/npu-v6 -DASCEND_COMPUTE_UNIT=ascend910b -DSBVG_TARGET_ROWS_PER_CORE=512
cmake --build build/npu-v6 -j4
# 使用独立目录构建 DMA 直取后端：
cmake -S code -B build/npu-v6-strided -DSBVG_STRIDED_SMALL=1
cmake --build build/npu-v6-strided -j4
```

需要 ASCConfig.cmake 和 npu_op_* 接口。设备验证仍可使用 `tests/npu_smoke.cpp` 和 `tools/run_npu_checks.sh`；本地没有真实设备，尚未运行此程序。

```bash
python tools/package_submission.py
python tools/package_submission.py --strided-small
```

每个包仅含11个源码文件，固定ZIP时间戳、LF换行，验证CRC、文件清单和内容一致性；SHA256写入 `dist/manifest.json`、`dist/manifest_strided.json`。`--rows`仍可调整分核，未匹配专门化要求的分块自动回退；不是指定不同Batch。
