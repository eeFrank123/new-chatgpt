/**
 * 安全带低挂高用几何判定融合算子 —— Kernel 侧实现（优化版）
 *
 * 固定计算路径（与题目语义一致，未做任何改动）：
 *   Y_ref = y1 + waist_ratio * (y2 - y1)
 *   violation_score = (cy > Y_ref + threshold) ? 1.0 : 0.0
 *
 * 相对原始提交，只改开销、不改语义：
 *   1. threshold 不再用 GlobalTensor::GetValue() 直接读 GM。该接口是标量流水上的
 *      阻塞式 GM 访问（还会触发 Cache Line 维护），在核启动的关键路径上等于又多了一次
 *      显存的往返延迟。现在改成把 threshold 和 boxes/anchors 一起 DataCopyPad 到 UB，
 *      再从 UB 读标量，标量读取只剩一次 flag 往返，GM 延迟与数据搬运重叠。
 *   2. 去掉尾块的 Duplicate 清零以及它带来的 V->MTE2 跨流水同步。GatherMask 以 256B
 *      为一个 repeat，valid 之后的那部分 UB 是脏数据，但它永远不会写回 GM（CopyOut 只写
 *      valid 行的结果），所以既不需要清零也不需要额外同步。
 *   3. 去掉同一条 vector 流水内相邻指令之间的 PipeBarrier<PIPE_V>()。同一流水按序发射，
 *      数据依赖由硬件保证；插 barrier 只会把流水排空，把本可以重叠的延迟串起来。
 *   4. Select 需要的全 1.0 向量在 Init 里建一次，不再每个 tile 重复 Duplicate。
 *   5. 每个核只处理一个 tile（Host 侧把单核行数限制在 UB 容量内），循环仍在但只跑一次。
 */
#include "kernel_operator.h"
#include "safety_belt_violation_geometry_tiling.h"
#include "tiling_key_safety_belt_violation_geometry.h"

using namespace AscendC;

namespace {
// 单个 tile 的行数上限，必须与 Host 侧 tiling_policy.h 的 kMaxRowsPerCore 一致。
constexpr uint32_t kTileRows = 256U;
// GatherMask 以 256B 为一个 repeat，取 64 行对 f32/f16 都是完整 repeat。
constexpr uint32_t kGroupRows = 64U;
// threshold 是 shape=[1] 的张量，搬进 UB 占用一个 32B block。
constexpr uint32_t kThresholdBlockBytes = 32U;
}  // namespace

template <class T>
class KernelSafetyBeltViolationGeometry {
public:
    __aicore__ inline void Init(
        GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
        GM_ADDR violationScore, const SafetyBeltViolationGeometryTilingData& data) {
        const uint32_t start = GetBlockIdx() * data.rowsPerCore;
        begin_ = start;
        rows_ = start < data.batchSize ? data.batchSize - start : 0U;
        if (rows_ > data.rowsPerCore) {
            rows_ = data.rowsPerCore;
        }
        waistRatio_ = data.waistRatio;
        threshold_ = data.defaultThreshold;  // 未传入 threshold 时 Host 传 0.0
        hasThreshold_ = data.hasThreshold != 0U;
        thresholdIsHalf_ = data.thresholdIsHalf != 0U;
        outputIsHalf_ = data.outputIsHalf != 0U;

        boxesGm_.SetGlobalBuffer((__gm__ T*)personBoxes, data.batchSize * 4U);
        anchorsGm_.SetGlobalBuffer((__gm__ T*)anchorPoints, data.batchSize * 2U);
        if (outputIsHalf_) {
            halfOutputGm_.SetGlobalBuffer((__gm__ half*)violationScore, data.batchSize);
        } else {
            outputGm_.SetGlobalBuffer((__gm__ float*)violationScore, data.batchSize);
        }

        pipe_.InitBuffer(boxesQueue_, 1, kTileRows * 4U * sizeof(T));
        pipe_.InitBuffer(anchorsQueue_, 1, kTileRows * 2U * sizeof(T));
        pipe_.InitBuffer(outputQueue_, 1, kTileRows * sizeof(float));
        pipe_.InitBuffer(coordinates_, 3U * kTileRows * sizeof(float));
        pipe_.InitBuffer(ones_, kTileRows * sizeof(float));
        pipe_.InitBuffer(compareMask_, kTileRows / 8U);
        if constexpr (sizeof(T) == sizeof(half)) {
            pipe_.InitBuffer(halfCoordinates_, 3U * kTileRows * sizeof(half));
        }

        if (hasThreshold_) {
            // 只绑定地址。真正的搬运放到 CopyIn 里、跟 boxes/anchors 一起发出去，
            // 让这几个 MTE2 请求的延迟互相重叠，关键路径上只付一次 GM 往返。
            pipe_.InitBuffer(thresholdBuffer_, kThresholdBlockBytes);
            if (thresholdIsHalf_) {
                thresholdHalfGm_.SetGlobalBuffer((__gm__ half*)threshold, 1U);
            } else {
                thresholdFloatGm_.SetGlobalBuffer((__gm__ float*)threshold, 1U);
            }
        }

        // Select 需要一条全 1.0 的向量，只建一次。
        Duplicate(ones_.Get<float>(), 1.0f, kTileRows);
    }

    __aicore__ inline void Process() {
        if (rows_ == 0U) {
            return;
        }
        bool firstTile = true;
        for (uint32_t offset = 0U; offset < rows_; offset += kTileRows) {
            const uint32_t remaining = rows_ - offset;
            const uint32_t valid = remaining < kTileRows ? remaining : kTileRows;
            const uint32_t padded = ((valid + kGroupRows - 1U) / kGroupRows) * kGroupRows;
            // threshold 只有 4/2 字节，和 boxes/anchors 一起发；放在最后发是为了保证
            // 它不会挡在几十 KB 的有效数据前面。
            CopyIn(begin_ + offset, valid, firstTile && hasThreshold_);
            if (firstTile && hasThreshold_) {
                ReadThreshold();
                firstTile = false;
            }
            if (outputIsHalf_) {
                Compute<half>(padded);
                CopyOut(halfOutputGm_, begin_ + offset, valid);
            } else {
                Compute<float>(padded);
                CopyOut(outputGm_, begin_ + offset, valid);
            }
        }
    }

private:
    // 标量值只能从 UB 读：MTE2_S 等搬运完成，S_V 让后面的 vector 指令等标量就位。
    __aicore__ inline void ReadThreshold() {
        const event_t mte2ToScalar = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(mte2ToScalar);
        WaitFlag<HardEvent::MTE2_S>(mte2ToScalar);
        threshold_ = thresholdIsHalf_
                         ? static_cast<float>(thresholdBuffer_.Get<half>().GetValue(0))
                         : thresholdBuffer_.Get<float>().GetValue(0);
        const event_t scalarToVector = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(scalarToVector);
        WaitFlag<HardEvent::S_V>(scalarToVector);
    }

    __aicore__ inline void CopyIn(uint32_t row, uint32_t valid, bool withThreshold) {
        LocalTensor<T> boxes = boxesQueue_.AllocTensor<T>();
        LocalTensor<T> anchors = anchorsQueue_.AllocTensor<T>();
        const DataCopyExtParams boxCopy{
            1U, valid * 4U * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        const DataCopyExtParams anchorCopy{
            1U, valid * 2U * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        const DataCopyPadExtParams<T> padding{true, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(boxes, boxesGm_[row * 4U], boxCopy, padding);
        DataCopyPad(anchors, anchorsGm_[row * 2U], anchorCopy, padding);
        if (withThreshold) {
            if (thresholdIsHalf_) {
                const DataCopyExtParams copy{
                    1U, static_cast<uint32_t>(sizeof(half)), 0U, 0U, 0U};
                DataCopyPad(thresholdBuffer_.Get<half>(), thresholdHalfGm_, copy,
                            DataCopyPadExtParams<half>{true, 0U, 0U, static_cast<half>(0)});
            } else {
                const DataCopyExtParams copy{
                    1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
                DataCopyPad(thresholdBuffer_.Get<float>(), thresholdFloatGm_, copy,
                            DataCopyPadExtParams<float>{true, 0U, 0U, 0.0f});
            }
        }
        boxesQueue_.EnQue(boxes);
        anchorsQueue_.EnQue(anchors);
    }

    template <typename OutputT>
    __aicore__ inline void Compute(uint32_t padded) {
        LocalTensor<T> boxes = boxesQueue_.DeQue<T>();
        LocalTensor<T> anchors = anchorsQueue_.DeQue<T>();
        LocalTensor<float> y1 = coordinates_.Get<float>();
        LocalTensor<float> y2 = y1[kTileRows];
        LocalTensor<float> cy = y1[2U * kTileRows];
        const GatherMaskParams boxGather{
            1U, static_cast<uint16_t>(padded * 4U * sizeof(T) / 256U), 8U, 0U};
        const GatherMaskParams anchorGather{
            1U, static_cast<uint16_t>(padded * 2U * sizeof(T) / 256U), 8U, 0U};
        uint64_t reserved = 0U;

        if constexpr (sizeof(T) == sizeof(half)) {
            LocalTensor<half> y1Half = halfCoordinates_.Get<half>();
            LocalTensor<half> y2Half = y1Half[kTileRows];
            LocalTensor<half> cyHalf = y1Half[2U * kTileRows];
            GatherMask(y1Half, boxes, static_cast<uint8_t>(4), false, 0, boxGather, reserved);
            GatherMask(y2Half, boxes, static_cast<uint8_t>(6), false, 0, boxGather, reserved);
            GatherMask(cyHalf, anchors, static_cast<uint8_t>(2), false, 0, anchorGather, reserved);
            // f16 输入先转 f32 再参与公式和比较，与参考实现一致。
            Cast(y1, y1Half, RoundMode::CAST_NONE, padded);
            Cast(y2, y2Half, RoundMode::CAST_NONE, padded);
            Cast(cy, cyHalf, RoundMode::CAST_NONE, padded);
        } else {
            // pattern 4/6 从 [x1,y1,x2,y2] 取 y1/y2，pattern 2 从 [cx,cy] 取 cy。
            // 不用标量循环，也不用依赖特殊 stride。
            GatherMask(y1, boxes, static_cast<uint8_t>(4), false, 0, boxGather, reserved);
            GatherMask(y2, boxes, static_cast<uint8_t>(6), false, 0, boxGather, reserved);
            GatherMask(cy, anchors, static_cast<uint8_t>(2), false, 0, anchorGather, reserved);
        }
        PipeBarrier<PIPE_V>();
        boxesQueue_.FreeTensor(boxes);
        anchorsQueue_.FreeTensor(anchors);

        // 保持参考实现的 FP32 运算顺序：不要合并成 FMA、不要改写成
        // cy - Y_ref > threshold，也不要对 waist_ratio == 1 做特判（y1+1.0*(y2-y1)
        // 与 y2 在 fp32 下不一定逐位相等）。
        Sub(y2, y2, y1, padded);
        Muls(y2, y2, waistRatio_, padded);
        Add(y2, y2, y1, padded);
        Adds(y2, y2, threshold_, padded);

        LocalTensor<uint8_t> mask = compareMask_.Get<uint8_t>();
        Compare(mask, cy, y2, CMPMODE::GT, padded);
        LocalTensor<float> one = ones_.Get<float>();
        LocalTensor<OutputT> output = outputQueue_.AllocTensor<OutputT>();
        if constexpr (sizeof(OutputT) == sizeof(float)) {
            // 相等边界：GT 为假，输出 0.0，不需要额外 epsilon。
            Select(output, mask, one, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, padded);
        } else {
            Select(cy, mask, one, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, padded);
            Cast(output, cy, RoundMode::CAST_NONE, padded);
        }
        outputQueue_.EnQue(output);
    }

    template <typename OutputT>
    __aicore__ inline void CopyOut(const GlobalTensor<OutputT>& outputGm, uint32_t row, uint32_t valid) {
        LocalTensor<OutputT> output = outputQueue_.DeQue<OutputT>();
        const DataCopyExtParams copy{
            1U, valid * static_cast<uint32_t>(sizeof(OutputT)), 0U, 0U, 0U};
        DataCopyPad(outputGm[row], output, copy);
        outputQueue_.FreeTensor(output);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> boxesQueue_;
    TQue<QuePosition::VECIN, 1> anchorsQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<TPosition::VECCALC> coordinates_;
    TBuf<TPosition::VECCALC> halfCoordinates_;
    TBuf<TPosition::VECCALC> compareMask_;
    TBuf<TPosition::VECCALC> ones_;
    TBuf<TPosition::VECCALC> thresholdBuffer_;
    GlobalTensor<T> boxesGm_;
    GlobalTensor<T> anchorsGm_;
    GlobalTensor<float> outputGm_;
    GlobalTensor<half> halfOutputGm_;
    GlobalTensor<float> thresholdFloatGm_;
    GlobalTensor<half> thresholdHalfGm_;
    uint32_t begin_ = 0U;
    uint32_t rows_ = 0U;
    float waistRatio_ = 0.0f;
    float threshold_ = 0.0f;
    bool hasThreshold_ = false;
    bool thresholdIsHalf_ = false;
    bool outputIsHalf_ = false;
};

template <typename DT_PERSON_BOXES>
__global__ __aicore__ void safety_belt_violation_geometry(
    GM_ADDR person_boxes, GM_ADDR anchor_points, GM_ADDR threshold,
    GM_ADDR violation_score, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
    GET_TILING_DATA_WITH_STRUCT(SafetyBeltViolationGeometryTilingData, tiling_data, tiling);
    KernelSafetyBeltViolationGeometry<DT_PERSON_BOXES> op;
    op.Init(person_boxes, anchor_points, threshold, violation_score, tiling_data);
    op.Process();
}
