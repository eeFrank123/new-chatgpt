# SafetyBeltViolationGeometry · 题目 2135

Ascend 910B 的 Ascend C 几何判定融合算子。比赛提交目录是 `code/`。

用户已确认平台实际提交了 v4，普通和关闭自动同步的两个版本均 Pass，但时延仍约 2.4–3.1μs，分数仍在四十几分。此次截图未显示精确分数，不据此推算。完整 12 点耗时保存在 `measurements/platform_observations.json`。

本轮 v5 改变参数读取与小 Batch 的坐标提取方式。尚未经过真实 CANN/NPU 编译和性能验证，不能声称已超过榜首。

## v5 修改

### 常见元数据使用独立内核

B={1,32,33,256,1024,4096} 与精确 float32 waist_ratio={0.8,0.9,1.0} 可选择 `STATIC_CASE` 编译期参数。在这些内核中，源码不执行 GET_TILING_DATA，直接用常量配置启动输入搬运，避免先读取 GM Tiling 再确定 DMA 长度的串行依赖。

大 Batch 还要求实际每核行数与专用内核一致：B256 对应256行/核，B1024/4096对应512行/核。可用核数不足或其他分核配置不匹配时，自动回到通用内核。所有其他合法 B 和属性也完整支持。

专门化只使用形状、属性和实际分核元数据；坐标和可选 threshold 每次仍从真实输入读取，严格执行题目公式。没有缓存输出或依据固定输入值返回结果。

共有 152 个 dtype/IO_MODE/STATIC_CASE 模板组合，其中 144 个静态、8 个通用。构建数量较 v4 增加，真实平台编译耗时尚未验证。[官方 Tiling 模板编程](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00025.html)支持此类 UINT 参数和独立内核。

### 两个小 Batch 计算后端

普通包保留三次 Gather 的小路径，以便测量本轮参数读取优化。`strided` 对照包对 B≤64 改为三次多块 DataCopyPad：直接从框和挂点的对应列提取 y1、y2、cy，每个值在 UB 中占一行32B，然后向量计算，最后由 MTE3 压缩写回连续输出。该路径不执行 Gather。

half 在四步运算前转换为 float32；不足8行的末尾只在 UB 中补零，避免内存 Compare 读取未初始化向量。GM 不多读或多写。strided 小路径最大显式 UB 为16,576B。跨行搬运会增加 DMA 块数和部分片上流量，是否比 Gather 快需平台比较。[DataCopyPad 官方参数规则](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0265.html)。

### 大路径减少向量初始化与类型转换

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

1. 先提交 `dist/safetybelt_2135_v5_rows512.zip`，比较元数据专门化和大路径优化。
2. 再提交 `dist/safetybelt_2135_v5_rows512_strided.zip`，比较用 DMA 直接提取坐标的小路径。

两包的代码只在 Kernel CPP 和 Kernel CMake 的 `SBVG_STRIDED_SMALL` 默认值0/1处不同。两者都保持目标512行/核和自动同步。

**完整替换 ZIP 中 `code/` 下的11个文件。** v5 新增 `op_kernel/safety_belt_violation_geometry_strided.h`，即使普通包也需要该头文件；Host、TilingKey 和 Kernel 必须一起更新。不要混用旧版库。若平台需要逐文件编辑，请先解压，再整体替换。

版本确认：Kernel CPP 第一行为 `// SBVG_VERSION: v5_static_metadata`；随后 `SBVG_STRIDED_SMALL` 默认值普通包0、strided包1。Host 和模板头也应包含 `STATIC_CASE`。

保存新记录的提交 ID、精确分数及全部12点耗时。若编译失败，保留完整编译日志。旧版 ZIP 保留用于回退。

## 验证

```bash
python tests/verify_cpu_model.py --rows-per-core 512
python tests/verify_cpu_model.py --rows-per-core 512 --strided-small
python tests/verify_cpu_model.py --rows-per-core 4096
python tests/verify_cpu_model.py --rows-per-core 512 --legacy-gather-api
```

每个最终配置检查9,337组输入、6,436,287行精确输出，其中740,800个相等边界；另检查262,144种分区。CPU模型编译实际 Host/Kernel，独立NumPy参考逐元素比较。

Host的512配置检查26个注册配置、192个静态派发和704个通用回退；4096配置检查96个静态派发和800个回退，两个配置均拒绝19组非法元数据。模型将静态路径的 Tiling 指针设为空，确认该路径零读取；通用路径每核读取一次。

多块 DataCopyPad 和高维 half/float Cast 模型已加入独立契约检查，覆盖 GM 字节步长、UB32B步长、每行隐式复制、压缩写回和尾块。普通接口模式19组契约正例、28种危险操作拒绝检查；旧 Gather API模式为27种。strided header另经独立15,360组用例覆盖所有 B=1..64、8种IO模式和三种常见比例。

模型还检查GM边界、UB初始化、跨流水与Vector RAW依赖、每行恰好写一次及返回时mask恢复。它不模拟真实异步执行、WAR/WAW、编译器同步或设备指令时延；不能证明真实NPU编译和性能。MinGW缺少ASan/UBSan运行库，使用显式模型断言，未声称通过sanitizer。汇总见 `validation_results.json`。

## 构建与打包

在安装题目对应 ASC SDK、CANN 和 NPU 的 Linux 环境加载 `set_env.sh` 后：

```bash
cmake -S code -B build/npu-v5 -DASCEND_COMPUTE_UNIT=ascend910b -DSBVG_TARGET_ROWS_PER_CORE=512
cmake --build build/npu-v5 -j4
# 使用独立目录构建 DMA 直取后端：
cmake -S code -B build/npu-v5-strided -DSBVG_STRIDED_SMALL=1
cmake --build build/npu-v5-strided -j4
```

需要 ASCConfig.cmake 和 npu_op_* 接口。设备验证仍可使用 `tests/npu_smoke.cpp` 和 `tools/run_npu_checks.sh`；本地没有真实设备，尚未运行此程序。

```bash
python tools/package_submission.py
python tools/package_submission.py --strided-small
```

每个包仅含11个源码文件，固定ZIP时间戳、LF换行，验证CRC、文件清单和内容一致性；SHA256写入 `dist/manifest.json`、`dist/manifest_strided.json`。`--rows`仍可调整分核，未匹配专门化要求的分块自动回退；不是指定不同Batch。
