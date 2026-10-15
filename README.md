# SafetyBeltViolationGeometry · 题目 2135

针对 Ascend 910B 的 Ascend C 几何判定融合算子。提交工程为 `code/`，保持题目 ZIP 的算子名、模板入口和构建结构。

用户的平台反馈为：早期提交 Pass、43.64 分；v2 提交 Pass、43.33 分，未取得整体收益。截图中的最新第一名为 90.19 分。观察记录见 `measurements/platform_observations.json`，测试点与具体 B/dtype 的对应关系仍未知。

本轮 **v3** 处理 Gather 和 Select 的接口开销，并固定常用块大小的地址与重复次数。**v3 尚未在真实 CANN/NPU 上编译、执行或测量时延，不能声称已超过榜首。** 默认仍为 512 行/核，便于比较；另附 128、256、1024 配置供平台调参。

优先提交 `dist/safetybelt_2135_v3_rows512.zip`，上传其中完整的 `code/` 工程。文件名的 `rows512` 表示每核目标行数为 512，最终分块还会受 Batch 和可用核数影响。测试通过后，再比较 `rows128` 的大 Batch 时延。

若只保留了解压后的源码，查看 `code/op_host/tiling_policy.h` 中 `SBVG_TARGET_ROWS_PER_CORE` 的默认值，以及 `code/CMakeLists.txt` 的同名配置。自行构建时 `-DSBVG_TARGET_ROWS_PER_CORE=...` 会覆盖默认值，实际使用值可在构建目录的 `CMakeCache.txt` 中查询。

## 计算与接口

严格依次执行 float32 `Sub → Muls → Add → Add → GT`：

```text
height = fp32(y2) - fp32(y1)
scaled = fp32(waist_ratio) * height
reference = fp32(y1) + scaled
boundary = reference + fp32(threshold)
output = fp32(fp32(cy) > boundary)
```

float16 坐标在参与公式前转 float32。不使用 FMA、epsilon、差值比较改写或 ratio=1 的简化。输出 `[B,1]`，相等返回 0。threshold 缺省时由 Tiling 传入 0，Kernel 不绑定或访问缺省输入地址。核心计算全部在 NPU。

保留两类规范配置（f32/f16 坐标，f32 阈值和输出），以及附件模板的 half 阈值/输出兼容配置。默认类型推导始终输出 float32；显式 half 输出仅用于原模板兼容。

## 优化内容

- **不读取不需要的 Gather 计数。** y1/y2/cy 的输出数量已由 B 确定。优先使用 CANN v8.5.0 中的四参数 `GatherMask`，分组设置 mask，省去七参数接口的 `get_rsvd_cnt()` 读取及重复模式设置。SDK 将该重载标为 deprecated，工程通过 SFINAE 检查可用性；缺少该重载时自动使用标准七参数接口。此时保持正确性，但不保证获得同等提速。实现依据：[v8.5.0 接口实现](https://raw.gitcode.com/cann/asc-devkit/raw/v8.5.0/impl/basic_api/kernel_operator_vec_gather_mask_intf_impl.h)、[A2 实现](https://raw.gitcode.com/cann/asc-devkit/raw/v8.5.0/impl/basic_api/dav_c220/kernel_operator_vec_gather_mask_impl.h)。计数读取没有显式 V_S 事件，实际耗时和编译器是否消除它仍需设备验证。
- **直接 mode-1 Select。** 通用路径提前准备 32B 的零数据并设置 CMPMASK，再使用公开的五参数 Select；内存输出 Compare 不改变 CMPMASK。避免高层标量 Select 内部准备临时零向量及额外屏障，保留当前数据依赖所需同步。[官方 SetCmpMask 示例](https://www.hiascend.com/doc_center/source/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0224.html)。
- **常用块大小固定布局。** 64、128、256、512、1024 行/核使用 C++ 模板容量，允许编译器折叠 UB 地址与重复次数；其他分块保留通用路径。最后一核仍只搬运和写回实际 rows，不读取越界 GM。32B 对齐的输入改用普通 DataCopy，不对齐尾块和标量阈值仍用 DataCopyPad。
- **静态 UB，移除 TPipe/TBuf。** 显式设置互不重叠的 UB 地址，入口调用 `InitSocState()`，手工配对 MTE2_V、V_MTE3 事件并在返回前等待 `PIPE_ALL`。无需初始化缓冲区分配器和事件池。[CANN 8.5 静态 Tensor 编程](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00019.html)。
- **B≤64 使用专用路径。** 一核、3,200B 显式 UB。Gather 只消费有效坐标，float32 不填充无用尾行；half 三列先初始化尾区，再统一转换。使用寄存器 Compare 和直接 mode-0 Select，省去结果 mask 的 UB 存取及标量 Select 的内部临时向量准备。比较依然为严格 GT。[Compare](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0067.html)、[Select](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0070.html)。
- **四步计算共享 mask。** 显式设置一次有效 mask，依次调用 `Sub/Muls/Add/Add` 的高维接口，避免每一步重复设置计数 mask；依赖 barrier 保留，计算后恢复默认 mask。不合并 float32 运算。
- **正确声明纯向量核。** 工程使用 Tiling 模板，因此在 `ASCENDC_TPL_ARGS_SEL` 中用 `ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY)` 设置核类型。此前受开关控制的 `KERNEL_TASK_TYPE_DEFAULT` 在该模板模式下不生效。见[官方模板示例](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00025.html)和[核类型说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0218.html)。
- **阈值广播随 DMA 完成。** 单元素 `DataCopyPad` 的左右显式 padding 都为 0 时，硬件用首元素补满 32B。float 阈值直接得到八份副本；half 阈值转换这八份副本。最后一次 Add 的源 block/repeat stride 均为 0，复用这个 block。删除 `MTE2_S → GetValue → S_V`，也不需要 Brcb。该 dummy 规则已有 [CANN 8.5 官方文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0265.html)与[官方技术文章](https://www.hiascend.com/developer/techArticles/20250627-1)依据。
- **float16 三列仅调用一次 Cast。** Gather 后的 y1/y2/cy 紧密排列，统一转换，随后全程 float32。
- **分核可调。** 默认 512 配置下，有足够 AIV 时 B=1024 使用 2 核、B=4096 使用 8 核；128 配置分别使用最多 8/32 核。最优核数需要比赛设备实测。[官方启动开销建议](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_00012.html)。
- **尾块安全。** GM 仅搬运有效字节。通用路径的框/挂点补零到 32B，再在 MTE2_V 后补齐 UB 剩余尾区；小 Batch 用有效长度 Gather 和 mask 处理 float32，half 则初始化统一转换所需的尾区。不把未知数据交给后续计算，不多读或多写 GM。
- **支持可用核数较少的情况。** 每核最多处理 4096 行；大 Gather 拆为不超过 2048 行的调用，规避旧版 255-repeat 限制。最坏显式 UB 为 f32 180,832B、f16 164,448B，另留出 8KiB SDK 保留空间。默认 512 行为 f32 22,688B、f16 最多 20,640B。

输入和输出 LocalTensor 分别声明为 VECIN、VECOUT，计算区为 VECCALC，共享同一物理 UB 地址空间。跨流水同步保留 MTE2_V、V_MTE3 各一次，向量依赖使用 PIPE_V barrier，入口末尾有 PIPE_ALL。不使用 TPipe、TBuf、TQue，也不禁用 SDK 保留 UB。

## 构建与实测

在具有题目要求的 ASC SDK、CANN 和 NPU 的 Linux 环境加载 `set_env.sh`，从仓库根目录执行：

```bash
cmake -S code -B build/npu-512 -DASCEND_COMPUTE_UNIT=ascend910b -DSBVG_TARGET_ROWS_PER_CORE=512
cmake --build build/npu-512 -j4
```

需要 SDK 提供 `ASCConfig.cmake` 与 `npu_op_*`；缺失时使用题目匹配版本或指定 `ASC_DIR`。每次更换分核参数使用独立构建目录，并重新构建/加载完整算子包。

先测 512 行配置，再比较 128 行配置；256、1024 为其他备选。观察全部用例和综合分，尤其 B=1024/4096 的 Kernel 时延。优先使用比赛评测器或 msprof 的设备 Kernel 时间；CPU 模型速度和 Host 墙钟时间不能用于预测排名。用户目前只能在线提交，因此每份提交需保存新记录的时间、完整 12 点耗时及分数。

已提供 `tests/npu_smoke.cpp` 与 `tools/run_npu_checks.sh` 供真实设备验证。先安装构建出的自定义算子包，并加载该包的环境脚本，再运行：

```bash
export SBVG_OP_API_INCLUDE=/实际路径/包含生成ACLNN头文件的目录
export SBVG_OP_API_LIB=/实际路径/包含libcust_opapi.so的目录
bash tools/run_npu_checks.sh --compat
# 同一设备上用 msprof 采集 Kernel 时间；不要比较此程序的运行总时间：
msprof --application="./build/npu-tests/npu_smoke --batch 4096 --repeats 30"
```

设备验证程序默认覆盖两组规范 IO、6 个 B、4 个比例、4 种阈值情况；`--compat` 增加模板兼容 IO。程序同时检查 0/1 结果、输入不变和输出保护区。该程序尚未经真实 CANN 编译或执行，保护区只能检测越界写入，越界读取仍需设备内存检查工具确认。

## 本地验证

```bash
python tests/verify_cpu_model.py
python tests/verify_cpu_model.py --rows-per-core 128
python tests/verify_cpu_model.py --rows-per-core 256
python tests/verify_cpu_model.py --rows-per-core 1024
# 模拟 SDK 只暴露标准七参数 GatherMask：
python tests/verify_cpu_model.py --legacy-gather-api
# Linux/GCC 支持相关运行库时可额外启用：
python tests/verify_cpu_model.py --sanitize
```

CPU 功能模型编译实际 Kernel/Host 源码，独立 NumPy 参考逐元素比较。它不属于提交源码，也不是 CPU 回退。报告见 `validation_results.json`：

- 128、256、512、1024 行/核四种配置，以及标准七参数 Gather 的 512 配置，分别通过 9,337 组用例、6,436,287 个输出精确比较，均包含 740,800 个精确相等边界。
- float16/float32，多种 B、比例、缺省/零/正阈值，以及模板兼容 IO 配置。
- 262,144 次合法 Batch/可用核数分块检查；26 组有效 Host 配置和 19 组非法元数据检查。
- 专门覆盖 FMA、重新排列公式、double 中间值、ratio=1 简化和 half 计算溢出可能导致的边界差异。
- 检查 GM 实际分配边界、UB 初始化、DMA 位置、基本跨流水可见性与每行恰好写回一次；检查同一物理 UB 的别名、有效 mask、寄存器比较结果和返回时 mask 模式。新增直接 mode-1 Select 的跨重复变化、65 行尾部、浮点标量和广播用例，以及 SFINAE 回退、float/half 四参数 Gather、普通 DataCopy、内存 Compare 保持 CMPMASK 的检查。

这不是 CANN 编译器或异步硬件仿真，不能证明真实设备时序或性能。模型中 PIPE_V 为无操作，不能用它证明删除向量 RAW 依赖屏障是安全的；当前源码保留这些同步。当前 MinGW 缺少 libasan/libubsan，未通过 sanitizer 验证；上面的结果使用显式模型断言。

## 提交包

```bash
python tools/package_submission.py --prefix safetybelt_2135_v3 --rows 512 128 256 1024
```

生成 `dist/safetybelt_2135_v3_rows512.zip`、`rows128.zip`、`rows256.zip`、`rows1024.zip`；每份仅含完整 `code/` 的 10 个文件，同步更改头文件和 CMake 默认值，使用 LF 换行并生成 `dist/manifest.json` SHA256 清单。旧版和 v2 包保留。

测试、文档和构建缓存不进入比赛提交包。请整体更新工程，包含 v2 的 `code/op_kernel/safety_belt_violation_geometry_small.h` 和 v3 新增的 `code/op_kernel/safety_belt_violation_geometry_gather.h`，也不要复用旧 Host/ACLNN 库。
