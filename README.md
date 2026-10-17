# SafetyBeltViolationGeometry · 题目 2135

Ascend 910B 的 Ascend C 几何判定融合算子。比赛提交目录是 code/。

平台最新提交864709（v6.1标准包）通过12/12，耗时 [2.20,1.92,2.50,3.70,2.12,2.72,2.34,1.99,2.38,3.81,2.07,2.65] μs；截图未显示精确分数。与v5标准包50.40分的记录相比，整体没有明确提升，第4、10点仍最慢。历史记录见 measurements/platform_observations.json。

当前为v7，新版已完成本地功能模型验证，真实CANN编译、正确性和时延仍需比赛平台确认，尚不能声称达到任何分数或超过榜首。

## v7 改动

### 两次DMA直接读取坐标

新增 safety_belt_violation_geometry_paired.h，适用于B≤64。第一次多块DataCopyPad从boxes[1]开始，每隔一个元素读取一个元素，得到交错的y1、y2；第二次从anchors[1]开始提取cy。每个标量在UB中占32B行，直接以向量步长访问，不执行Gather。

相比上一版strided小路径，坐标DMA从三次减少至两次，half转换也从三次减少至两次。四步float32运算使用独立参考线缓冲，避免连续写入覆盖尚未读取的交错输入。最后由MTE3将每个32B行的第一个结果压缩写回连续输出。显式UB为18,656B。

最后一个box读取是boxes[4B-1]，最后一个anchor读取是anchors[2B-1]。不足8行的尾部只在UB中补零，输入、输出均使用精确GM长度。[DataCopyPad官方参数规则](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0265.html)。多块DMA及额外片上流量是否优于Gather，需要平台比较。

### 固定形状支持任意合法比例

原来的快捷内核仅覆盖精确float32比例0.8、0.9、1.0。v7保留这些内核，另为相同11个形状增加任意比例路径：B={1,32,33,64,128,129,256,257,512,1024,4096}。

新路径在输入搬运时，同时从Tiling的waistRatio字段读取精确4B，并将比例广播给向量Mul。它无需先加载整份Tiling再决定输入长度，也不经过MTE2_S/GetValue/S_V。Mul和原Muls均执行一次独立float32乘法。

Tiling布局仍为16B，比例偏移8B，并有静态断言。常见比例使用STATIC_CASE 1..33，任意比例使用34..44；未知形状或实际分核不匹配仍使用通用0号路径。共有360个合法dtype/IO_MODE/STATIC_CASE组合。比例缓冲增加32B，最大显式UB为164,512B，另保留8KiB SDK空间。

第4、10点是否使用其他比例尚不明确，因此这项优化的实际收益须评测确认。除第1点明确B32/ratio1外，其他测试点的具体元数据未知。

### 保留C++11修复和数值语义

v6曾因opbuild使用GNU11.4.0的-std=c++11，拒绝constexpr函数中的局部数组而失败。v6.1修复已在平台通过；v7继续使用单条return的C++11 constexpr，并保留严格C++11检查。

公式仍按四步独立float32运算执行：

    height = fp32(y2) - fp32(y1)
    scaled = height * fp32(waist_ratio)
    reference = fp32(y1) + scaled
    boundary = reference + fp32(threshold)
    output = fp32(fp32(cy) > boundary)

half坐标先转换为float32。相等输出0，缺失threshold按0处理且不访问该输入地址。不使用FMA、公式重排、epsilon或ratio=1简化。输出默认float32，并保留原空工程half阈值/输出兼容配置。核心计算全部在NPU。

## 提交

1. 优先提交 dist/safetybelt_2135_v7_rows512_paired.zip，测试新的两次坐标DMA路径。
2. 再提交 dist/safetybelt_2135_v7_rows512_gather.zip 作为对照；两者均包含任意比例优化，中大Batch路径相同。

**整体替换ZIP中code/下的13个文件。** 本次新增safety_belt_violation_geometry_paired.h和safety_belt_violation_geometry_ratio.h。Host、Kernel和TilingKey必须配套。

版本确认：Kernel CPP首行为 // SBVG_VERSION: v7_paired_ratio_dma；SBVG_SMALL_BACKEND在paired包默认为2、Gather包为0，Kernel CMake的值相同。两包均512行/核，保留编译器自动同步和必要手工同步。

保存提交ID、精确分数及全部12点耗时；若编译失败，保留第一条error及前后日志。旧版ZIP仍在dist中用于回退。

## 验证

    python tests/verify_cpu_model.py --rows-per-core 512 --small-backend paired
    python tests/verify_cpu_model.py --rows-per-core 512 --small-backend gather
    python tests/verify_cpu_model.py --rows-per-core 4096 --small-backend paired
    python tests/verify_cpu_model.py --rows-per-core 512 --small-backend gather --legacy-gather-api

每个配置以真实Host/Kernel源码和CPU API模型进行9,337组输入、6,436,287行独立NumPy精确比较，包含740,800个相等边界；另覆盖262,144种分区。完整Host模型和无CANN桩的共享头均以g++ -std=c++11 -pedantic-errors编译。

模型检查GM长度、UB初始化及别名、Vector RAW依赖、跨流水同步、每行恰好写一次及退出时mask恢复。常量比例路径传入空Tiling，任意比例路径断言每核仅DMA一次4B比例字段，通用路径每核加载一次Tiling。新增独立Mul广播、多repeat步长、间隔目标写入、精确比例DMA契约检查，错误步长、缺少同步及多读字段须被拒绝。

模型不模拟真实异步执行、WAR/WAW或编译器同步，不等同于CANN编译或NPU性能测试。当前本地没有CANN工具链和NPU。汇总见validation_results.json。

## 构建与打包

在有题目对应ASC SDK/CANN的Linux环境加载set_env.sh后：

    cmake -S code -B build/npu-v7 -DASCEND_COMPUTE_UNIT=ascend910b -DSBVG_TARGET_ROWS_PER_CORE=512
    cmake --build build/npu-v7 -j4
    cmake -S code -B build/npu-v7-gather -DSBVG_SMALL_BACKEND=0
    cmake --build build/npu-v7-gather -j4

源码打包不需要CANN：

    python tools/package_submission.py --small-backend paired
    python tools/package_submission.py --small-backend gather

ZIP仅含13个源码文件，固定时间戳、LF换行，检查CRC、清单与内容；SHA256见dist/manifest_paired.json和dist/manifest_gather.json。保留--small-backend strided用于旧三次坐标DMA方案的本地比较。
