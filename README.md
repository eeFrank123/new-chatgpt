# SafetyBeltViolationGeometry · 题目 2135

Ascend 910B 的 Ascend C 几何判定融合算子。比赛提交目录是 `code/`，算子名和工程结构与题目模板一致。

最近平台结果：v3 的 512 行配置 Pass、42.68 分；128 行配置 Pass、34.50 分。测试点 5、6、11、12 在 128 配置下明显退步。完整 12 点耗时保存在 `measurements/platform_observations.json`，具体测试点的 B/dtype 尚未知。

本轮 v4 精简 Kernel 入口与标量代码，不再继续增加核数。它尚未经过真实 CANN/NPU 编译或评测，不能将本地检查结果解释为已经提速或超过第一名。

## 本轮修改

- 删除同一入口内的 64/128/256/512/1024/通用容量 switch。大路径只保留一个固定最大容量的 UB 布局，实际搬运和运算仍按有效长度执行。
- 用 Tiling 模板参数 `IO_MODE` 区分阈值存在性、阈值存储类型和输出存储类型。生成 8 个合法 dtype/IO_MODE 入口，每个入口通过 `if constexpr` 删除不适用的分支。仍正确声明 `ASCENDC_TPL_AIV_ONLY`。
- 不再把三个 IO 标志装入 Tiling，结构由 28B 缩至 16B，只保留 Batch、每核行数、比例和缺省阈值。
- 首次包含 SDK 头前设置 `K_MAX_SHAPE_DIM=0`，减少不使用的 Tensor 形状信息；设置 `ASCENDC_DUMP=0`，关闭设备调试输出支持。kernel CMake 同时传入后一个宏。
- 小 Batch 路径也使用编译期 IO 模式。固定大布局的 half 三列各做一次 Cast，只转换初始化过的有效区域。
- 所有显式流水同步继续保留。另提供独立 `manualsync` 对照包，通过正式编译选项关闭编译器自动同步；它与同名普通包仅有 kernel CMake 默认选项一处差异。

“入口代码过大、Tensor 形状信息和自动同步可能增加固定开销”是待设备验证的性能判断。128 行退步只能支持减少分核方向，不能证明某个新配置必然最快。

实现依据：[官方模板参数编程](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00025.html)、[设置 K_MAX_SHAPE_DIM](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0019.html)、[编译选项调用顺序](https://www.hiascend.com/doc_center/source/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00035.html)、[自动同步说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/opdevg/Ascendcopdevg/atlas_ascendc_10_0068.html)。

## 计算和接口

严格以 float32 依次执行：

```text
height = fp32(y2) - fp32(y1)
scaled = fp32(waist_ratio) * height
reference = fp32(y1) + scaled
boundary = reference + fp32(threshold)
output = fp32(fp32(cy) > boundary)
```

half 坐标在运算前转换为 float32；不用 FMA、重排、epsilon 或 ratio=1 简化。输出顺序与输入一致，相等输出 0。threshold 缺省时 Host 将 0.0 写入 Tiling，Kernel 不绑定或读取缺省输入地址。核心计算全部在 NPU。

默认推导输出 float32，注册题目要求的两类 IO；另外保留原空工程 half 阈值/输出的兼容配置。

保留 v3 的三列 Gather、直接 Compare/Select、阈值 DMA 广播、静态 LocalTensor、B≤64 专用路径及字节级尾块搬运。四参数 Gather 优先，SDK 不提供时自动回退七参数重载。最大显式 UB 为 float32 180,832B、half 输出兼容 164,448B，另外预留 8KiB SDK 空间，小路径 3,200B。

## 提交顺序

1. 先提交 `dist/safetybelt_2135_v4_rows512.zip`，与已经测过的 v3 rows512 比较本轮结构修改。
2. 再提交 `dist/safetybelt_2135_v4_rows512_manualsync.zip`，单独检查关闭自动同步是否有收益。这个配置保留源码中的全部手工同步，但仍需平台检验真实时序、正确性和编译兼容性。
3. `rows1024.zip` 与 `rows4096.zip` 用于普通自动同步模式的分核对照；后者对所有合法 Batch 均为单核。每核目标行数不是运行时固定 Batch，不会跳过任何样本。

上传每个 ZIP 中完整 `code/` 的 10 个文件。v4 没有新增头文件，但 Host、Kernel、Tiling struct 和 TilingKey 必须一起更新；旧 Host/Kernel 混用会造成 ABI 错误。平台不接受 ZIP 时先解压，再整体替换工程文件。保留的旧版压缩包可用于回退。

确认版本：打开 `code/op_kernel/safety_belt_violation_geometry.cpp`，第一行为 `// SBVG_VERSION: v4_compact`。每核目标行数查看 `code/op_host/tiling_policy.h` 的 `SBVG_TARGET_ROWS_PER_CORE`；`manualsync` 包的 `code/op_kernel/CMakeLists.txt` 将 `SBVG_MANUAL_SYNC_ONLY` 默认设为 ON。自行构建时 CMake 参数可覆盖默认值。

保存每次新记录的提交 ID、12 点耗时和分数。用户只能在线评测；没有将 CPU 墙钟时间用于预测比赛排名。

## 本地验证

```bash
python tests/verify_cpu_model.py --rows-per-core 512
python tests/verify_cpu_model.py --rows-per-core 1024
python tests/verify_cpu_model.py --rows-per-core 4096
python tests/verify_cpu_model.py --rows-per-core 512 --legacy-gather-api
```

上述四组均通过，每组 9,337 个输入、6,436,287 个输出精确比较，其中 740,800 个相等边界。另检查全部合法 Batch 与 1..64 可用核数的 262,144 种分区，以及 Host 26 个有效和 19 个非法配置。

CPU 模型编译实际 Host/Kernel 源码，由独立 NumPy 参考逐元素比较。覆盖两类坐标类型、可选阈值、不同 IO 存储类型、随机比例、尾块、GM 边界、UB 初始化、每行只写一次，以及能区分 FMA/公式重排/half 提前运算的数值用例。新增 PIPE_V 内 UB 写后读与 CMPMASK→Select 依赖检查；18 个故意错误被拒绝，旧 Gather 模式为 17 个。

模型不属于提交工程，不是 CPU 回退，也不是 CANN 异步仿真。它不验证 WAR/WAW、真实事件分配复用或编译器插入的同步，不能证明设备性能。当前 MinGW 缺少 ASan/UBSan 运行库，使用显式边界和初始化断言；未声称通过 sanitizer。汇总报告见 `validation_results.json`。

## 构建与打包

在安装题目对应 ASC SDK/CANN/NPU 的 Linux 环境中加载 `set_env.sh` 后：

```bash
cmake -S code -B build/npu-v4 -DASCEND_COMPUTE_UNIT=ascend910b -DSBVG_TARGET_ROWS_PER_CORE=512
cmake --build build/npu-v4 -j4
# 仅在单独构建目录中测试手工同步配置：
cmake -S code -B build/npu-v4-manual -DSBVG_MANUAL_SYNC_ONLY=ON
cmake --build build/npu-v4-manual -j4
```

需要 ASCConfig.cmake 与 npu_op_* 接口。设备测试程序仍是 `tests/npu_smoke.cpp`，运行脚本为 `tools/run_npu_checks.sh`；先安装生成的算子包，设置 `SBVG_OP_API_INCLUDE`、`SBVG_OP_API_LIB`，然后执行 `bash tools/run_npu_checks.sh --compat`。该程序尚未在真实设备上编译运行。

```bash
python tools/package_submission.py
python tools/package_submission.py --rows 512 --manual-sync-only
```

默认生成 v4 的 512/1024/4096 普通包，第二条生成 512 手工同步对照包。每个 ZIP 仅含完整源码；LF 换行，固定 ZIP 时间戳，并验证 CRC/文件清单/内容一致性。SHA256 分别记录在 `dist/manifest.json` 和 `dist/manifest_manualsync.json`。
