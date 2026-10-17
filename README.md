# SafetyBeltViolationGeometry · 题目2135

Ascend910B 的 Ascend C 几何判定融合算子，提交源码位于 `code/`。当前版本 **v11_scalar_small** 增加小 Batch 的 NPU 标量路径，并提供强制内联的向量对照包。本地没有 CANN 编译器或 NPU，实际编译、正确性和提速以平台评测为准。

## 最新平台结果

用户确认 v10 两包按推荐顺序提交，均通过12/12；截图未显示精确分数，用户报告仍为50分出头。单位为μs。

| 测试点 | v10 banked，889172 | v10 compact，889399 |
| --- | ---: | ---: |
| 1 | 2.21 | 2.16 |
| 2 | 2.06 | 2.03 |
| 3 | 2.38 | 2.42 |
| 4 | 2.56 | 2.57 |
| 5 | 2.10 | 2.08 |
| 6 | 2.61 | 2.70 |
| 7 | 2.23 | 2.11 |
| 8 | 2.36 | 2.32 |
| 9 | 2.38 | 2.43 |
| 10 | 2.49 | 2.56 |
| 11 | 2.08 | 2.12 |
| 12 | 2.65 | 2.78 |

地址布局没有表现出明确收益，本轮停止仅调整 UB 地址。1024行/核、paired DMA和标量 GM 阈值读取此前出现回退，继续保留512行/核及 DMA 阈值读取。完整历史见 `measurements/platform_observations.json`。各点完整形状、属性映射仍未知，不根据耗时猜测参数。

## v11 两条路径

主包在 B≤64 时精确 DMA 输入到 UB，随后由 NPU Scalar Unit 读取 y1/y2/cy、逐步执行 float32 几何计算和比较。float32 小路径不需要 Gather、向量数学屏障或 Select；float16 先对连续的原始坐标和可选 half 阈值做一次 Cast，再读取转换后的 float32。输出经 S_MTE3 同步后精确 DMA。所有核心计算及数值读取均在 NPU，无 CPU 回退，也不通过 Scalar GM 指针读取输入。

对照包保持 v10 banked 的 Vector 算法，加入强制内联和严格浮点编译选项。两包 B>64 的分核、算术和搬运相同，仍为512行/核。两包仅选择不同的小 Batch 后端，便于区分标量路径与内联调整的实际影响。

`SBVG_FORCE_INLINE` 只修饰项目辅助函数，不修改 SDK 或 kernel 入口。采用 CANN9 生成器已有的 `inline __attribute__((always_inline))` 写法。Scalar 函数的 Clang pragma 禁止融合、重排，Kernel 编译选项增加 `-ffp-contract=off -fno-fast-math`。尚未取得生成汇编，不能断言旧版存在额外调用或新版必然更快。依据及限制见 `measurements/v11_design_review.md`。

## 提交文件

1. **新路径实验包：** `dist/safetybelt_2135_v11_rows512_scalar.zip`。
2. **向量内联对照包：** `dist/safetybelt_2135_v11_rows512_banked.zip`。

**整体使用 ZIP 内 code/ 下的18个文件，不与旧版混用。** 本次新增 `safety_belt_violation_geometry_scalar.h` 和 `safety_belt_violation_geometry_inline.h`。

CPP 首行均为 `// SBVG_VERSION: v11_scalar_small`。主包 `SBVG_SMALL_BACKEND` 默认5，对照包默认4，CPP 与 Kernel CMake 保持一致。两包强制内联和自动同步均开启，`SBVG_SCALAR_THRESHOLD=0`、`SBVG_BANKED_COMPACT=0`。主包 `SBVG_SCALAR_SMALL_LIMIT=64`。

按以上顺序评测并保留包名、提交ID、精确分数及12点耗时。编译失败时保留第一条 error 及上下文，对照包可独立提交。旧 ZIP 保留用于回退。

## 数值与内存规则

```text
height    = fp32(y2) - fp32(y1)
scaled    = height * fp32(waist_ratio)
reference = fp32(y1) + scaled
boundary  = reference + fp32(threshold)
output    = fp32(fp32(cy) > boundary)
```

half 坐标先转 float32，相等输出0。比例支持所有合法 float32 值。比例为1仍保留减法和加y1，不直接用y2替代。threshold 缺省为0，Kernel 不绑定、不读取不存在的地址。标准输出为float32，并保留原空工程 half 阈值/输出兼容配置；half二值输出使用精确的0x0000/0x3c00编码。

支持 B=1..4096，非对齐尾块按真实 GM 长度搬运。所有参与 Cast 的 padding 均初始化。小路径静态 UB 用量3456B，另保留 SDK 临时区；大形状保留 v10 已检查的布局与同步。Host/共享Tiling继续满足 opbuild 的C++11要求。

## 验证与打包

```powershell
python tests/verify_cpu_model.py --small-backend scalar
python tests/verify_cpu_model.py --small-backend banked
python tests/verify_scalar_small.py
python tools/package_submission.py --small-backend scalar
python tools/package_submission.py --small-backend banked
```

两个现代接口完整配置各覆盖9,346个用例、6,436,584行、740,899项相等边界及262,144种分区，使用真实 Host/Kernel 源码和 CPU API 功能模型，与独立 NumPy 参考精确比较。独立标量检查遍历每个 B=1..64、全部8种合法IO、静态/动态比例和强制/普通内联，共27,648个用例、898,560行。旧接口另通过936个大形状用例、981,720行及模型契约检查。最终验证记录见 `validation_results.json`。

模型检查物理GM边界、UB初始化、Vector RAW与Scalar跨流水可见性。新增负例确认缺少 MTE2_S、V_S、S_MTE3 均会被拒绝。模型不验证 CANN 编译、真实异步 WAR/WAW、Bisheng标量浮点指令降低、自动同步或NPU时延。通过模型不能证明达到90分。

在题目对应的 CANN/ASC Linux 环境加载 set_env.sh 后构建：

```bash
cmake -S code -B build/npu-v11 -DASCEND_COMPUTE_UNIT=ascend910b -DSBVG_SMALL_BACKEND=5 -DSBVG_TARGET_ROWS_PER_CORE=512
cmake --build build/npu-v11 -j4
```

对照包使用独立目录并设 `-DSBVG_SMALL_BACKEND=4`。`SBVG_FORCE_INLINE_ENABLED=OFF` 可恢复普通内联，`SBVG_SCALAR_SMALL_LIMIT` 可设1..64以缩小标量分派范围。打包工具同步修改头文件和 CMake 默认值，校验18文件清单、LF换行、CRC及可复现内容。SHA256和配置见 `dist/manifest_scalar.json`、`dist/manifest_banked.json`。
