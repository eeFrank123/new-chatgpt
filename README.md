# SafetyBeltViolationGeometry · 题目 2135

针对 Ascend 910B 的 Ascend C 几何判定融合算子。提交工程为 `code/`，保持题目 ZIP 的算子名、模板入口和构建结构。

此版针对启动与标量流水开销优化，**尚未在 CANN/NPU 上编译、执行或测量时延，不能声称已超过榜首**。默认 512 行/核是待实测的起点；附有 256、1024 行/核配置供比较。

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

- **正确声明纯向量核。** 工程使用 Tiling 模板，因此在 `ASCENDC_TPL_ARGS_SEL` 中用 `ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY)` 设置核类型。此前受开关控制的 `KERNEL_TASK_TYPE_DEFAULT` 在该模板模式下不生效。见[官方模板示例](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00025.html)和[核类型说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0218.html)。
- **阈值广播随 DMA 完成。** 单元素 `DataCopyPad` 的左右显式 padding 都为 0 时，硬件用首元素补满 32B。float 阈值直接得到八份副本；half 阈值转换这八份副本。最后一次 Add 的源 block/repeat stride 均为 0，复用这个 block。删除 `MTE2_S → GetValue → S_V`，也不需要 Brcb。该 dummy 规则已有 [CANN 8.5 官方文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0265.html)与[官方技术文章](https://www.hiascend.com/developer/techArticles/20250627-1)依据。
- **float16 三列仅调用一次 Cast。** Gather 后的 y1/y2/cy 紧密排列，统一转换，随后全程 float32。
- **减少默认核数。** 有足够 AIV 时，B=1024 从 8 核降至 2 核，B=4096 从 32 核降至 8 核；B≤256 为单核。最优核数需要比赛设备实测。[官方启动开销建议](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_00012.html)。
- **尾块完整初始化。** GM 仅搬运有效字节；框/挂点显式补零到 32B，再在 MTE2_V 后补齐 UB 剩余尾区。不会把未知尾数据交给 Gather/Compare，不多读或多写 GM。
- **支持可用核数较少的情况。** UB 按实际行数申请，最多处理 4096 行；大 Gather 拆为不超过 2048 行的调用，规避旧版 255-repeat 限制。最坏显式 UB 分配为 f32 188,992B、f16 164,416B；默认 512 行为 f32 23,680B、f16 20,608B。

使用三个 TBuf（VECIN、VECCALC、VECOUT），不使用 TQue。跨流水同步仅保留 MTE2_V、V_MTE3 各一次，向量依赖使用 PIPE_V barrier。TPipe 在 Kernel 入口创建。

## 构建与实测

在具有题目要求的 ASC SDK、CANN 和 NPU 的 Linux 环境加载 `set_env.sh`，从仓库根目录执行：

```bash
cmake -S code -B build/npu-512 -DASCEND_COMPUTE_UNIT=ascend910b -DSBVG_TARGET_ROWS_PER_CORE=512
cmake --build build/npu-512 -j4
```

需要 SDK 提供 `ASCConfig.cmake` 与 `npu_op_*`；缺失时使用题目匹配版本或指定 `ASC_DIR`。每次更换分核参数使用独立构建目录，并重新构建/加载完整算子包。

先测 512 行配置，再比较 256、1024 行配置；必要时加测 128。观察全部用例和综合分，尤其 B=1024/4096 的 Kernel 时延。优先使用比赛评测器或 msprof 的设备 Kernel 时间；CPU 模型速度和 Host 墙钟时间不能用于预测排名。

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
python tests/verify_cpu_model.py --rows-per-core 256
python tests/verify_cpu_model.py --rows-per-core 1024
# Linux/GCC 支持相关运行库时可额外启用：
python tests/verify_cpu_model.py --sanitize
```

CPU 功能模型编译实际 Kernel/Host 源码，独立 NumPy 参考逐元素比较。它不属于提交源码，也不是 CPU 回退。报告见 `validation_results.json`：

- 512、256、1024 行/核三种配置分别通过 8,752 组用例、6,131,286 个输出精确比较，均包含 698,767 个精确相等边界。
- float16/float32，多种 B、比例、缺省/零/正阈值，以及模板兼容 IO 配置。
- 262,144 次合法 Batch/可用核数分块检查；26 组有效 Host 配置和 19 组非法元数据检查。
- 专门覆盖 FMA、重新排列公式、double 中间值、ratio=1 简化和 half 计算溢出可能导致的边界差异。
- 检查 GM 实际分配边界、UB 初始化、DMA 位置、基本跨流水可见性与每行恰好写回一次；用故意错误的模型程序确认检查确实会拒绝错误。

这不是 CANN 编译器或异步硬件仿真，不能证明真实设备时序或性能。当前 MinGW 缺少 libasan/libubsan，未通过 sanitizer 验证；上面的结果使用显式模型断言。

## 提交包

```bash
python tools/package_submission.py --rows 512 256 1024
```

生成 `dist/safetybelt_2135_rows512.zip` 等文件；每份仅含完整 `code/`，同步更改头文件和 CMake 默认值，使用 LF 换行并生成 SHA256 清单。测试、文档和构建缓存不进入比赛提交包。不要只替换 Kernel 或复用旧 Host/ACLNN 库。
