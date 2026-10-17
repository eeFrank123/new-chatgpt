# SafetyBeltViolationGeometry · 题目 2135

Ascend 910B 的 Ascend C 几何判定融合算子，提交源码位于 `code/`。当前为 **v10_banked_pingpong**：保留已通过平台的512行/核配置，改进片上缓冲区布局和逐步计算的写回位置。本地没有CANN工具链和NPU，实际提速及最终分数需平台评测确认。

## 最新平台结果

用户提供的两次v9.1提交均通过12/12。下表耗时单位为μs；截图未显示精确分数。包身份按此前推荐的提交顺序推定，截图本身没有展示配置宏。

| 测试点 | 图一，886940，推定主包 | 图二，887172，推定标量阈值对照 |
| --- | ---: | ---: |
| 1 | 2.22 | 2.68 |
| 2 | 1.97 | 2.52 |
| 3 | 2.42 | 3.06 |
| 4 | 2.48 | 2.91 |
| 5 | 2.17 | 2.08 |
| 6 | 2.65 | 2.62 |
| 7 | 2.24 | 2.42 |
| 8 | 1.87 | 1.90 |
| 9 | 2.44 | 2.46 |
| 10 | 2.42 | 2.40 |
| 11 | 2.09 | 2.06 |
| 12 | 2.61 | 2.55 |

图二前四点明显回退，其余差异较小；v10恢复统一的DMA阈值读取，不推荐继续提交标量阈值方案。1024行/核此前在第5、6、11、12点明显回退，paired DMA小路径也曾明显回退，均停止推荐。完整历史见 `measurements/platform_observations.json`；各测试点的B、比例和threshold尚未完整公开，不能由耗时认定用例参数。

v9最初的编译错误也已在v9.1修复：SDK的 `GM_ADDR` 是 `__gm__ uint8_t*` 宏，两个指针成员须分别声明，不能用逗号合并。v10保留修复及对应本地编译检查。

## v10改动与对照

- **小Batch采用独立目标缓冲区。** B≤64仍使用原有三次Gather和有效行mask。Sub、Mul、Add及threshold加法交替写入两个float缓冲区，避免连续原地更新同一列；保留题目规定的计算顺序和所有依赖屏障。
- **主包分开片上地址。** 将float坐标列与运算暂存区分别放在64KiB、128KiB附近，输入及常量保留在低地址区。这是待NPU评测的布局方案，不能据此宣称消除了全部bank冲突。
- **紧凑地址对照包。** 与主包执行相同的Gather、算术、DMA和同步，仅将小Batch的两个float缓冲区放回原来的紧凑地址。它帮助判断收益来自独立目标写回，还是来自地址分布。
- **大Batch布局同步调整。** 使用 `pitch = padded_ | 64U`，使float列间隔为奇数个256B；原先为偶数个256B时增加64个float，原先为奇数个时保持间距。half三列分别Cast到对应列，保持转换后同一处依赖屏障。输出区复用为交替计算的暂存区，不增加GM搬运或对外中间结果。两个推荐包的大Batch实现完全相同。

两个推荐包均为512行/核、编译器自动同步开启、标量阈值读取关闭。half仍先转float32，静态比例恰为1时省去精确恒等的乘1，threshold缺失时省去加0。所有核心计算和输入数值读取均在NPU；Host处理形状、类型、属性与分核。

## 提交文件

1. **优先主包：** `dist/safetybelt_2135_v10_rows512_banked.zip`。
2. **紧凑地址对照：** `dist/safetybelt_2135_v10_rows512_banked_compact.zip`。

**整体替换ZIP内 `code/` 下的16个文件。** Host、Kernel、共享Tiling和TilingKey必须来自同一包。本次新增 `code/op_kernel/safety_belt_violation_geometry_banked.h`；已有其他后端头保留供编译选择。

检查Kernel CPP第一行是 `// SBVG_VERSION: v10_banked_pingpong`，`SBVG_SMALL_BACKEND` 默认4，行/核默认512。主包 `SBVG_BANKED_COMPACT` 默认0、对照包默认1；banked头与Kernel CMake应一致。两个包 `SBVG_SCALAR_THRESHOLD` 均默认0，`SBVG_MANUAL_SYNC_ONLY` 均为OFF。

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

half坐标先精确转换为float32；相等输出0。threshold缺失按0处理，Kernel不访问不存在的地址。比例为1时仍独立计算 `y2-y1` 和后续加y1，不直接用y2替代参考线。不使用FMA、公式重排或epsilon；保留常见比例0.8、0.9、1.0及任意合法比例路径。输出默认float32，并保留题目原空工程的half阈值/输出兼容配置。

B支持1至4096，非对齐尾块按真实GM长度读写。共享Host头继续满足opbuild的C++11要求。

## 本地验证

```powershell
python tests/verify_cpu_model.py --rows-per-core 512 --small-backend banked
python tests/verify_cpu_model.py --rows-per-core 512 --small-backend banked --legacy-gather-api
python tests/verify_cpu_model.py --rows-per-core 512 --small-backend banked --compact-buffers
python tests/verify_cpu_model.py --rows-per-core 512 --small-backend banked --compact-buffers --legacy-gather-api
python tests/verify_small_layouts.py
```

验证使用真实Host/Kernel源码与CPU API功能模型，结果与独立NumPy参考精确比较。每个配置覆盖9,346个用例、6,436,584行及740,899项相等边界，另检查262,144种分区；Host和共享头采用 `g++ -std=c++11 -pedantic-errors` 编译。独立小Batch检查覆盖B=1至64，每种布局及Gather接口组合各13,824个用例，四组合共55,296个用例。最终通过状态和准确计数见 `validation_results.json` 和 `build/cpu_model/` 对应结果。源码核查依据与未采用的优化方向见 `measurements/v10_design_review.md`。

模型检查GM边界、UB初始化及别名、Vector RAW依赖、事件ID配对、跨流水可见性、输出只写一次及mask恢复。masked Cast契约确认未读取或写入非活动行；全掩码及错误源步长负例必须被拒绝。旧标量阈值方案保留同址缓存刷新契约，但本次推荐包关闭该方案。

恒等操作另经过16,777,226项乘1位级检查及6,291,456次几何比较检查，覆盖所有float32非正规数和有符号零、相邻边界及中间高度溢出。这是IEEE/NumPy验证，不能替代设备浮点模式验证。

CPU模型不验证CANN编译、真实异步WAR/WAW、编译器自动同步、UB银行吞吐或NPU时延。通过模型不等于平台通过，也不能据此承诺90分。

## 构建与打包

在题目对应CANN/ASC的Linux环境加载 `set_env.sh` 后构建主包：

```bash
cmake -S code -B build/npu-v10 -DASCEND_COMPUTE_UNIT=ascend910b -DSBVG_TARGET_ROWS_PER_CORE=512 -DSBVG_SMALL_BACKEND=4 -DSBVG_BANKED_COMPACT=OFF -DSBVG_SCALAR_THRESHOLD=OFF -DSBVG_MANUAL_SYNC_ONLY=OFF
cmake --build build/npu-v10 -j4
```

构建紧凑地址对照时使用独立目录，并设 `-DSBVG_BANKED_COMPACT=ON`。源码打包无需CANN：

```powershell
python tools/package_submission.py --rows 512 --small-backend banked
python tools/package_submission.py --rows 512 --small-backend banked --compact-buffers
```

每个ZIP仅含16个源码文件，固定时间戳及LF换行，并校验CRC、清单与文件内容。SHA256分别见 `dist/manifest_banked.json` 和 `dist/manifest_banked_compact.json`。`--compact-buffers`仅支持banked后端；历史后端、标量阈值及手动同步选项仍保留用于已有实验。
