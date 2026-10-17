# SafetyBeltViolationGeometry · 题目 2135

Ascend 910B 的 Ascend C 几何判定融合算子，提交源码位于 `code/`。当前版本 **v9_masked_cast**，以已通过平台的512行/核配置继续减少小Batch固定开销。本地没有CANN工具链和NPU，实际提速及最终分数需平台评测确认。

## 最新平台结果

2026-10-17的三次v8提交均通过12/12，耗时单位为μs：

| 测试点 | 512 staged，882474 | 1024 staged，883547 | 1024 Gather，884492 |
| --- | ---: | ---: | ---: |
| 1 | 2.23 | 2.29 | 2.29 |
| 2 | 2.00 | 1.97 | 1.96 |
| 3 | 2.51 | 2.48 | 2.50 |
| 4 | 2.49 | 2.46 | 2.50 |
| 5 | 2.10 | 4.88 | 4.52 |
| 6 | 2.64 | 5.62 | 5.24 |
| 7 | 2.30 | 2.35 | 2.36 |
| 8 | 1.93 | 1.99 | 2.00 |
| 9 | 2.45 | 2.52 | 2.56 |
| 10 | 2.38 | 2.37 | 2.36 |
| 11 | 2.08 | 3.56 | 3.38 |
| 12 | 2.63 | 4.56 | 4.54 |

1024行/核在第5、6、11、12点明显回退，v9仅提供512行/核包。此前paired DMA小路径也出现明显回退，停止推荐。完整历史见 `measurements/platform_observations.json`；各测试点的B、比例和threshold尚未完整公开，不能由耗时认定用例参数。

## v9改动

- **half三列只转换有效行。** Gather后以实际B设置掩码，一次L0 Cast通过3个repeat、步长 `{1,1,8,4}` 将y1、y2、cy转换为float32。无需初始化三列尾部，删除原先192个half的填零和对应屏障。B=1至64均使用此方式，没有新增DMA。
- **消除可证明的恒等操作。** 仅静态比例恰为1.0时省略乘1及其屏障，仍独立计算 `height=y2-y1` 和 `reference=y1+height`；不直接用y2替代参考线。仅threshold缺失时省略加0。传入的threshold照常读取、相加和严格比较。
- **float32沿用分阶段Gather。** B≤64时使用两个MTE2事件，让人员参考线计算有机会与挂点、阈值搬运重叠。保留所有依赖所需的Vector屏障和最终 `PIPE_ALL`。
- **可选阈值读取对照。** 主包继续使用DMA。对照包只将已传入的float32 threshold改为设备侧标量读取：单缓存行DCCI清理失效、`GetValue`、S_V事件2同步后用 `Adds`。每次调用都刷新缓存；half threshold仍使用DMA。该方案可能增加开销，需要单独评测。

所有核心计算和输入数值读取仍在NPU。Host只处理形状、类型、属性与分核，不执行几何计算。

## 提交文件

1. **优先主包：** `dist/safetybelt_2135_v9_rows512_staged.zip`。
2. **可选对照：** `dist/safetybelt_2135_v9_rows512_staged_scalar_threshold.zip`。

**整体替换ZIP内 `code/` 下的15个文件。** Host、Kernel、共享Tiling和TilingKey必须来自同一包。本次新增 `code/op_kernel/safety_belt_violation_geometry_threshold.h`；已有paired、strided头保留供编译选择，推荐包不启用它们。

确认Kernel CPP第一行是 `// SBVG_VERSION: v9_masked_cast`，`SBVG_SMALL_BACKEND` 默认值为3，行/核默认值为512。主包 `SBVG_SCALAR_THRESHOLD` 默认0，对照包默认1；共享阈值头与Kernel CMake应一致。

保存每次提交ID、精确分数及全部12点耗时。编译失败时保留第一条error及前后日志。旧ZIP继续保留用于回退。

## 数值与输入约束

计算遵循题目规定的逐步float32语义：

```text
height    = fp32(y2) - fp32(y1)
scaled    = height * fp32(waist_ratio)
reference = fp32(y1) + scaled
boundary  = reference + fp32(threshold)
output    = fp32(fp32(cy) > boundary)
```

half坐标先精确转换为float32；相等输出0。threshold缺失按0处理，Kernel不访问不存在的地址。仅消除上述恒等操作，不使用FMA、公式重排或epsilon；保留常见比例0.8、0.9、1.0及任意合法比例路径。输出默认float32，并保留题目原空工程的half阈值/输出兼容配置。

B支持1至4096，非对齐尾块按真实GM长度读写。共享Host头继续满足opbuild的C++11要求。

## 本地验证

```powershell
python tests/verify_cpu_model.py --rows-per-core 512 --small-backend staged
python tests/verify_cpu_model.py --rows-per-core 512 --small-backend staged --legacy-gather-api
python tests/verify_cpu_model.py --rows-per-core 512 --small-backend staged --scalar-threshold
python tests/verify_cpu_model.py --rows-per-core 512 --small-backend staged --scalar-threshold --legacy-gather-api
```

验证使用真实Host/Kernel源码与CPU API功能模型，结果与独立NumPy参考精确比较。包括随机输入、相等边界、非标准比例、接近1的比例、浮点消减、大坐标、大阈值和极小比例。每个配置另检查262,144种分区；Host和共享头采用 `g++ -std=c++11 -pedantic-errors` 编译。最终通过状态与准确计数见 `validation_results.json` 和 `build/cpu_model/` 对应结果。

模型检查GM边界、UB初始化及别名、Vector RAW依赖、事件ID配对、跨流水可见性、输出只写一次及mask恢复。新增masked Cast契约覆盖B=1至64，确认未写入非活动行；全掩码及错误源步长两个负例必须被拒绝。阈值标量对照另检查同一GM地址更新数值后刷新缓存。

恒等操作另经过16,777,226项乘1位级检查及6,291,456次几何比较检查，覆盖所有float32非正规数和有符号零、相邻边界及中间高度溢出。这是IEEE/NumPy验证，不能替代设备浮点模式验证。

CPU模型不验证CANN编译、真实异步WAR/WAW、编译器自动同步或NPU时延。通过模型不等于平台通过，也不能据此承诺90分。

## 构建与打包

在题目对应CANN/ASC的Linux环境加载 `set_env.sh` 后构建主包：

```bash
cmake -S code -B build/npu-v9 -DASCEND_COMPUTE_UNIT=ascend910b -DSBVG_TARGET_ROWS_PER_CORE=512 -DSBVG_SMALL_BACKEND=3 -DSBVG_SCALAR_THRESHOLD=OFF
cmake --build build/npu-v9 -j4
```

构建阈值对照时使用独立目录，并设 `-DSBVG_SCALAR_THRESHOLD=ON`。源码打包无需CANN：

```powershell
python tools/package_submission.py --rows 512 --small-backend staged
python tools/package_submission.py --rows 512 --small-backend staged --scalar-threshold
```

每个ZIP仅含15个源码文件，固定时间戳及LF换行，并校验CRC、清单与文件内容。SHA256分别见 `dist/manifest_staged.json` 和 `dist/manifest_staged_scalar_threshold.json`。
