# SafetyBeltViolationGeometry · 题目 2135

Ascend 910B 的 Ascend C 几何判定融合算子，比赛提交目录为 `code/`。当前版本 **v8_staged_gather**，目标是继续降低实际输入计算的时延。用户目前最佳成绩约51分，精确小数未知；v8尚无平台成绩，不能据本地模型承诺90分或超过榜首。

## 最新平台结果

v7两次提交均通过12/12，全部耗时单位为μs：

| 测试点 | v7 paired DMA，878136 | v7 Gather，878860 |
| --- | ---: | ---: |
| 1 | 6.08 | 2.16 |
| 2 | 5.61 | 2.11 |
| 3 | 2.52 | 2.40 |
| 4 | 2.44 | 2.53 |
| 5 | 2.14 | 2.09 |
| 6 | 2.75 | 2.80 |
| 7 | 6.22 | 2.28 |
| 8 | 6.59 | 2.11 |
| 9 | 2.43 | 2.33 |
| 10 | 2.39 | 2.64 |
| 11 | 2.10 | 2.06 |
| 12 | 2.63 | 2.70 |

paired DMA在第1、2、7、8点明显回退，停止推荐。v8以Gather为基础继续调整；各测试点的具体B、比例和threshold仍未完整公开，不能由耗时直接认定用例参数。完整历史记录见 `measurements/platform_observations.json`。

## v8 改动

- **小Batch float32分阶段搬运。** B≤64时先搬入人员框及必要比例，标记MTE2事件0；再搬入挂点和可选threshold，标记事件1。Vector等待事件0即可Gather y1/y2并计算参考线，随后等待事件1完成cy读取、阈值相加和比较。这样提供参考线计算与后续DMA重叠的机会，实际收益由平台确认。
- **float16保留已通过平台的Gather路径。** 三列坐标先一次转换为float32，再执行计算，避免分阶段路径引入额外转换和依赖。
- **1024行/核保留静态元数据路径。** Host和Kernel同时获得 `SBVG_TARGET_ROWS_PER_CORE`，共享头内对应分区与实际分核一致，避免切换分核配置后意外退回通用Tiling读取路径。512行/核仍为默认。
- 保留常见比例0.8、0.9、1.0的常量路径，以及任意合法float32比例的4B比例DMA路径。未知形状或实际分区不匹配时继续使用通用路径。

新增头文件 `code/op_kernel/safety_belt_violation_geometry_staged.h`。保留编译器自动同步、必要Vector屏障、跨流水事件及退出时的 `PIPE_ALL`。四参数Gather接口可用时继续优先调用；该接口选择本身不代表v8获得了实测提速。

## 提交文件与评测顺序

1. **优先：** `dist/safetybelt_2135_v8_rows512_staged.zip`，与v7 Gather保持512行/核，测试新的float32小Batch流水。
2. **随后：** `dist/safetybelt_2135_v8_rows1024_staged.zip`，比较减少分核后的表现。
3. **必要对照：** `dist/safetybelt_2135_v8_rows1024_gather.zip`，使用原Gather小路径，帮助区分分核与流水变化的影响。

**整体替换每个ZIP内 `code/` 下的14个文件。** Host、Kernel和TilingKey必须来自同一个包。相较v7仅新增上述staged头文件；旧paired、strided头仍在包内供编译选择，推荐包不启用它们。

确认Kernel CPP第一行是 `// SBVG_VERSION: v8_staged_gather`。staged包的 `SBVG_SMALL_BACKEND` 默认值为3，Gather包为0；Kernel CPP和Kernel CMake应一致。行/核默认值可在根CMake及共享Tiling头中查看。

保存每次提交ID、精确分数和全部12点耗时。编译失败时保存第一条error及前后日志。旧ZIP保留在 `dist/` 中用于回退。

## 数值与输入约束

对每行实际输入始终执行四步独立float32运算：

```text
height    = fp32(y2) - fp32(y1)
scaled    = height * fp32(waist_ratio)
reference = fp32(y1) + scaled
boundary  = reference + fp32(threshold)
output    = fp32(fp32(cy) > boundary)
```

half坐标在运算前转为float32，相等输出0；threshold缺失时按0处理，Kernel不访问该输入地址。不使用FMA、公式重排、epsilon或ratio=1简化。输出默认float32，并保留题目原空工程half阈值/输出兼容配置。人员框、挂点和阈值均由Kernel读取，Host只处理输入元数据和属性，核心计算全部在NPU。

B支持1至4096，非对齐尾块按真实GM长度搬运和写回，不依赖额外输入存储空间。共享头继续遵守opbuild的C++11限制，避免v6首次提交曾出现的constexpr局部数组编译错误。

## 本地验证

```powershell
python tests/verify_cpu_model.py --rows-per-core 512 --small-backend staged
python tests/verify_cpu_model.py --rows-per-core 1024 --small-backend staged
python tests/verify_cpu_model.py --rows-per-core 512 --small-backend staged --legacy-gather-api
python tests/verify_cpu_model.py --rows-per-core 1024 --small-backend gather
```

每次完整验证使用真实Host/Kernel源码与CPU API功能模型，对9,337组输入、6,436,287行结果执行独立NumPy精确比较，包含740,800个相等边界和262,144种分区检查。Host注册覆盖26项，另有19项非法输入检查；Host及无CANN桩的共享头采用 `g++ -std=c++11 -pedantic-errors` 编译。完成状态以 `validation_results.json` 及 `build/cpu_model/` 中相应结果文件为准。

模型检查GM访问长度、UB初始化及别名、Vector RAW依赖、跨流水可见性、不同事件ID配对、每行恰好写一次及退出mask恢复。v8还检查等待事件0不会提前暴露事件1的数据。常量比例路径无需读取Tiling；任意比例路径仅DMA比例字段；通用路径读取完整Tiling。

**本地没有CANN工具链和NPU。** CPU模型不能验证真实异步WAR/WAW、编译器自动同步及设备时延，通过模型不等于通过CANN编译或平台评测。

## 构建与打包

在题目对应CANN/ASC的Linux环境加载 `set_env.sh` 后：

```bash
cmake -S code -B build/npu-v8-512 -DASCEND_COMPUTE_UNIT=ascend910b -DSBVG_TARGET_ROWS_PER_CORE=512 -DSBVG_SMALL_BACKEND=3
cmake --build build/npu-v8-512 -j4
cmake -S code -B build/npu-v8-1024 -DASCEND_COMPUTE_UNIT=ascend910b -DSBVG_TARGET_ROWS_PER_CORE=1024 -DSBVG_SMALL_BACKEND=3
cmake --build build/npu-v8-1024 -j4
```

源码打包不需要CANN：

```powershell
python tools/package_submission.py --rows 512 1024 --small-backend staged
python tools/package_submission.py --rows 1024 --small-backend gather
```

ZIP仅含14个源码文件，固定时间戳及LF换行，并检查CRC、清单与文件内容。SHA256见 `dist/manifest_staged.json` 和 `dist/manifest_gather.json`。