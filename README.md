# SafetyBeltViolationGeometry（题目 2135）

基于题目附件模板补全的 Ascend C 自定义算子工程，目标设备为 **Ascend 910B**。
提交源码位于 `code/`，保留了模板的算子名、入口名、TilingKey 模板和工程结构。

## 计算规则

对每一行输入，严格依次进行以下 float32 运算：

```text
height   = float32(y2) - float32(y1)
scaled   = float32(waist_ratio) * height
reference = float32(y1) + scaled
boundary = reference + threshold
output   = float32(float32(cy) > boundary)
```

`person_boxes` 与 `anchor_points` 使用相同的 float16 或 float32 类型。
float16 坐标先转 float32 再参与计算；`threshold` 和输出始终为 float32。
输出形状为 `[B,1]`，相等时返回 `0.0`。不使用 epsilon、融合乘加，
也不把公式改写为 `cy-reference > threshold`；这些变换可能改变边界判定。
即使 `waist_ratio=1`，也保留题目指定的减、乘、加计算顺序。

## 实现要点

- Host 检查输入形状、类型、Batch 范围及比例属性，推导 `[B,1]` 和 float32 输出。
- 可选 `threshold` 存在时，Host 只检查其元数据，Kernel 读取其设备上的一个 float32 值。
  不存在时，Host 将默认值 `0.0` 写入 Tiling；Kernel 不访问该输入地址。
- 分核起点按 64 行对齐，优先每核处理约 128 行；小 Batch 减少启动核数。
  每个计算块最多 256 行，数据仅从 GM 搬入一次，然后在 UB 完成计算。
- `GatherMask` 内置模式 4、6、2 分别提取 `y1`、`y2`、`cy`。
- float32 `Compare` 需要 64 元素对齐，因此在 UB 补齐；GM 的读写长度始终是实际有效行数。
- `DataCopyPad` 处理非 32 字节对齐的末尾数据，避免 B=1、33 等情况越界。
- 使用队列和显式同步保护搬运与计算依赖。无需额外 workspace。
- 每核显式 UB 分配为 float32 路径 10,272 字节，float16 路径 8,736 字节。

## 文件

```text
code/
  CMakeLists.txt
  op_host/
    CMakeLists.txt
    safety_belt_violation_geometry.cpp
    tiling_policy.h
  op_kernel/
    CMakeLists.txt
    safety_belt_violation_geometry.cpp
    safety_belt_violation_geometry_tiling.h
    tiling_key_safety_belt_violation_geometry.h
```

`tests/` 是独立的本地验证工具，不参与 NPU 算子构建；其中的 CPU API 模型
只用于检查源码逻辑，不是算子的 CPU 回退实现。

## 编译与提交

需要支持附件中 `find_package(ASC)` / `npu_op_*` 工程接口的 CANN 开发环境。
先按实际安装位置加载 CANN 的 `set_env.sh`，然后在项目根目录执行：

```bash
cmake -S code -B build/npu -DASCEND_COMPUTE_UNIT=ascend910b
cmake --build build/npu -j 4
```

如果提示找不到 `ASCConfig.cmake`，需要正确配置与题目模板匹配的 CANN SDK，
或通过 `-DASC_DIR=实际包含ASCConfig.cmake的目录` 指定它。
CANNJudge 提交时保留 `code/` 下的工程结构。

当前环境没有 CANN 编译器、运行库或 NPU，**尚未进行 NPU 编译、实际执行或性能测试，
也未提交 CANNJudge**。不能将下面的本地功能检查视为评测通过证明。

## 已执行的本地验证

使用 g++ 编译实际的 Kernel 源码，只将其使用的 Ascend C API 替换为有边界检查的
CPU 功能模型，再与独立的 NumPy float32 参考计算逐元素精确比较。
这不验证 CANN 编译兼容性、设备流水时序或真实 DMA 行为。

```bash
python tests/verify_cpu_model.py --sanitize
```

依赖：Python 3、NumPy、支持 `_Float16` 的 g++（GCC 12 或更新）。
上述命令启用 AddressSanitizer 和 UndefinedBehaviorSanitizer。

本次执行结果：

- 2,209 组测试全部通过，共精确比较 1,224,195 个输出。
- 包含 205,378 个精确相等边界，结果均为 0。
- float16 / float32；23 种 Batch，包括 1、32、33、256、1024、4096 及多个相邻尾块形状。
- 比例 0.8、0.9、1.0、0.33333334；阈值缺省、0、0.125、7.5。
- 覆盖零高框、零坐标、图像边界、负坐标及阈值上下相邻可表示数。
- 核数 1、8、48，覆盖单核多块与多核执行。
- 对全部 B=1..4096、核数=1..64，完成 262,144 次分块覆盖检查。
- CPU 模型的 GM/UB 边界与 UB 未初始化读取检查通过；缺省阈值以空指针测试。
- ASan / UBSan 未报告错误。

运行记录见 `validation_results.json`。实际 NPU 编译与评测结果仍需在昇腾环境确认。
