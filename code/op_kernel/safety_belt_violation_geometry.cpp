#include "kernel_operator.h"
#include "safety_belt_violation_geometry_tiling.h"
#include "tiling_key_safety_belt_violation_geometry.h"

using namespace AscendC;

namespace {
constexpr uint32_t kTileRows = 256;
constexpr uint32_t kVectorRows = 64;
}  // namespace

template <class T>
class KernelSafetyBeltViolationGeometry {
public:
    __aicore__ inline void Init(
        GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
        GM_ADDR violationScore, const SafetyBeltViolationGeometryTilingData& data) {
        begin_ = GetBlockIdx() * data.rowsPerCore;
        rows_ = begin_ < data.batchSize ? data.batchSize - begin_ : 0U;
        if (rows_ > data.rowsPerCore) {
            rows_ = data.rowsPerCore;
        }
        waistRatio_ = data.waistRatio;
        threshold_ = data.defaultThreshold;
        outputIsHalf_ = data.outputIsHalf != 0U;
        if (data.hasThreshold != 0U) {
            // An optional input is a device tensor: never dereference it on Host.
            // When absent, do not construct or access its GlobalTensor.
            if (data.thresholdIsHalf != 0U) {
                GlobalTensor<half> thresholdGm;
                thresholdGm.SetGlobalBuffer((__gm__ half*)threshold, 1);
                threshold_ = static_cast<float>(thresholdGm.GetValue(0));
            } else {
                GlobalTensor<float> thresholdGm;
                thresholdGm.SetGlobalBuffer((__gm__ float*)threshold, 1);
                threshold_ = thresholdGm.GetValue(0);
            }
        }
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
        pipe_.InitBuffer(compareMask_, kTileRows / 8U);
        if constexpr (sizeof(T) == sizeof(half)) {
            pipe_.InitBuffer(halfCoordinates_, 3U * kTileRows * sizeof(half));
        }
    }

    __aicore__ inline void Process() {
        for (uint32_t offset = 0; offset < rows_; offset += kTileRows) {
            const uint32_t remaining = rows_ - offset;
            const uint32_t valid = remaining < kTileRows ? remaining : kTileRows;
            const uint32_t padded = (valid + kVectorRows - 1U) / kVectorRows * kVectorRows;
            CopyIn(begin_ + offset, valid, padded);
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
    __aicore__ inline void CopyIn(uint32_t row, uint32_t valid, uint32_t padded) {
        LocalTensor<T> boxes = boxesQueue_.AllocTensor<T>();
        LocalTensor<T> anchors = anchorsQueue_.AllocTensor<T>();
        if (valid != padded) {
            // Padding exists only in UB. GM reads still contain exactly valid rows.
            Duplicate(boxes, static_cast<T>(0), padded * 4U);
            Duplicate(anchors, static_cast<T>(0), padded * 2U);
            const event_t event = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(event);
            WaitFlag<HardEvent::V_MTE2>(event);
        }
        const DataCopyExtParams boxCopy{1, valid * 4U * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        const DataCopyExtParams anchorCopy{1, valid * 2U * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        const DataCopyPadExtParams<T> padding{true, 0, 0, static_cast<T>(0)};
        DataCopyPad(boxes, boxesGm_[row * 4U], boxCopy, padding);
        DataCopyPad(anchors, anchorsGm_[row * 2U], anchorCopy, padding);
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
            1, static_cast<uint16_t>(padded * 4U * sizeof(T) / 256U), 8, 0};
        const GatherMaskParams anchorGather{
            1, static_cast<uint16_t>(padded * 2U * sizeof(T) / 256U), 8, 0};
        uint64_t reserved = 0;

        if constexpr (sizeof(T) == sizeof(half)) {
            LocalTensor<half> y1Half = halfCoordinates_.Get<half>();
            LocalTensor<half> y2Half = y1Half[kTileRows];
            LocalTensor<half> cyHalf = y1Half[2U * kTileRows];
            GatherMask(y1Half, boxes, static_cast<uint8_t>(4), false, 0, boxGather, reserved);
            GatherMask(y2Half, boxes, static_cast<uint8_t>(6), false, 0, boxGather, reserved);
            GatherMask(cyHalf, anchors, static_cast<uint8_t>(2), false, 0, anchorGather, reserved);
            PipeBarrier<PIPE_V>();
            Cast(y1, y1Half, RoundMode::CAST_NONE, padded);
            Cast(y2, y2Half, RoundMode::CAST_NONE, padded);
            Cast(cy, cyHalf, RoundMode::CAST_NONE, padded);
        } else {
            // Patterns 4/6 select y1/y2 from [x1,y1,x2,y2].
            // Pattern 2 selects cy from [cx,cy]. No scalar gather loop is needed.
            GatherMask(y1, boxes, static_cast<uint8_t>(4), false, 0, boxGather, reserved);
            GatherMask(y2, boxes, static_cast<uint8_t>(6), false, 0, boxGather, reserved);
            GatherMask(cy, anchors, static_cast<uint8_t>(2), false, 0, anchorGather, reserved);
        }
        PipeBarrier<PIPE_V>();
        boxesQueue_.FreeTensor(boxes);
        anchorsQueue_.FreeTensor(anchors);

        // Separate FP32 operations preserve the reference rounding order.
        // Do not replace these with FMA, cy-Y_ref > threshold, or a ratio==1 shortcut.
        Sub(y2, y2, y1, padded);
        PipeBarrier<PIPE_V>();
        Muls(y2, y2, waistRatio_, padded);
        PipeBarrier<PIPE_V>();
        Add(y2, y2, y1, padded);
        PipeBarrier<PIPE_V>();
        Adds(y2, y2, threshold_, padded);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint8_t> mask = compareMask_.Get<uint8_t>();
        Compare(mask, cy, y2, CMPMODE::GT, padded);
        Duplicate(y1, 1.0f, padded);
        PipeBarrier<PIPE_V>();
        LocalTensor<OutputT> output = outputQueue_.AllocTensor<OutputT>();
        if constexpr (sizeof(OutputT) == sizeof(float)) {
            Select(output, mask, y1, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, padded);
        } else {
            // The comparison remains FP32. Only the exact binary result is
            // converted for a caller-provided half output from the original template.
            Select(cy, mask, y1, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, padded);
            PipeBarrier<PIPE_V>();
            Cast(output, cy, RoundMode::CAST_NONE, padded);
        }
        outputQueue_.EnQue(output);
    }

    template <typename OutputT>
    __aicore__ inline void CopyOut(const GlobalTensor<OutputT>& outputGm, uint32_t row, uint32_t valid) {
        LocalTensor<OutputT> output = outputQueue_.DeQue<OutputT>();
        const DataCopyExtParams copy{1, valid * static_cast<uint32_t>(sizeof(OutputT)), 0, 0, 0};
        DataCopyPad(outputGm[row], output, copy);
        outputQueue_.FreeTensor(output);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> boxesQueue_, anchorsQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<TPosition::VECCALC> coordinates_, halfCoordinates_, compareMask_;
    GlobalTensor<T> boxesGm_, anchorsGm_;
    GlobalTensor<float> outputGm_;
    GlobalTensor<half> halfOutputGm_;
    bool outputIsHalf_ = false;
    uint32_t begin_ = 0;
    uint32_t rows_ = 0;
    float waistRatio_ = 0.0f;
    float threshold_ = 0.0f;
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
