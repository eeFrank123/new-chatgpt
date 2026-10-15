#include "kernel_operator.h"
#include "safety_belt_violation_geometry_tiling.h"
#include "tiling_key_safety_belt_violation_geometry.h"
#include "safety_belt_violation_geometry_small.h"

using namespace AscendC;

namespace {
constexpr uint32_t kVectorRows = 64U;
__aicore__ inline uint32_t Align32(uint32_t bytes) {
    return (bytes + 31U) / 32U * 32U;
}
}  // namespace

template <class T>
class KernelSafetyBeltViolationGeometry {
public:
    __aicore__ inline void Init(
        GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
        GM_ADDR violationScore, const SafetyBeltViolationGeometryTilingData& data) {
        begin_ = GetBlockIdx() * data.rowsPerCore;
        rows_ = data.batchSize - begin_;  // Host launches only nonempty blocks.
        if (rows_ > data.rowsPerCore) {
            rows_ = data.rowsPerCore;
        }
        padded_ = (rows_ + kVectorRows - 1U) / kVectorRows * kVectorRows;
        ratio_ = data.waistRatio;
        defaultThreshold_ = data.defaultThreshold;
        hasThreshold_ = data.hasThreshold != 0U;
        thresholdIsHalf_ = sizeof(T) == sizeof(half) && data.thresholdIsHalf != 0U;
        outputIsHalf_ = sizeof(T) == sizeof(half) && data.outputIsHalf != 0U;

        boxesGm_.SetGlobalBuffer((__gm__ T*)personBoxes, data.batchSize * 4U);
        anchorsGm_.SetGlobalBuffer((__gm__ T*)anchorPoints, data.batchSize * 2U);
        if (outputIsHalf_) {
            halfOutGm_.SetGlobalBuffer((__gm__ half*)violationScore, data.batchSize);
        } else {
            outGm_.SetGlobalBuffer((__gm__ float*)violationScore, data.batchSize);
        }
        // Never bind or read an absent threshold pointer.
        if (hasThreshold_) {
            if (thresholdIsHalf_) {
                halfThresholdGm_.SetGlobalBuffer((__gm__ half*)threshold, 1U);
            } else {
                thresholdGm_.SetGlobalBuffer((__gm__ float*)threshold, 1U);
            }
        }

        // Static tensors share the physical UB address space. These regions
        // never overlap, and each address is 32B aligned. No TPipe/TBuf setup.
        const uint32_t boxBytes = padded_ * 4U * sizeof(T);
        const uint32_t anchorBytes = padded_ * 2U * sizeof(T);
        const uint32_t calcStart = boxBytes + anchorBytes + 32U;
        boxesUb_ = LocalTensor<T>(TPosition::VECIN, 0U, 4U * padded_);
        anchorsUb_ = LocalTensor<T>(TPosition::VECIN, boxBytes, 2U * padded_);
        thresholdUb_ = LocalTensor<uint8_t>(TPosition::VECIN, boxBytes + anchorBytes, 32U);

        const uint32_t coordBytes = padded_ * 3U * sizeof(float);
        const uint32_t halfCoordBytes = sizeof(T) == sizeof(half) ? padded_ * 3U * sizeof(half) : 0U;
        const uint32_t onesOffset = calcStart + coordBytes + halfCoordBytes;
        const uint32_t maskOffset = onesOffset + padded_ * sizeof(float);
        const uint32_t thresholdOffset = maskOffset + Align32(padded_ / 8U);
        const uint32_t outputOffset = thresholdOffset + 32U;
        coordsUb_ = LocalTensor<float>(TPosition::VECCALC, calcStart, 3U * padded_);
        y1Ub_ = coordsUb_;
        y2Ub_ = coordsUb_[padded_];
        cyUb_ = coordsUb_[2U * padded_];
        if constexpr (sizeof(T) == sizeof(half)) {
            halfCoordsUb_ = LocalTensor<half>(TPosition::VECCALC, calcStart + coordBytes, 3U * padded_);
        }
        onesUb_ = LocalTensor<float>(TPosition::VECCALC, onesOffset, padded_);
        maskUb_ = LocalTensor<uint8_t>(TPosition::VECCALC, maskOffset, Align32(padded_ / 8U));
        floatThresholdUb_ = LocalTensor<float>(TPosition::VECCALC, thresholdOffset, 8U);
        broadcastThresholdUb_ = thresholdIsHalf_ ? floatThresholdUb_
                                               : thresholdUb_.ReinterpretCast<float>();

        outUb_ = LocalTensor<float>(TPosition::VECOUT, outputOffset, padded_);
        if (outputIsHalf_) {
            halfOutUb_ = LocalTensor<half>(TPosition::VECOUT, outputOffset + padded_ * sizeof(float), padded_);
        }
    }

    __aicore__ inline void Process() {
        // Independent vector work overlaps input DMA.
        Duplicate(onesUb_, 1.0f, padded_);
        CopyIn();
        InitializeInputTail();
        PrepareThreshold();
        GatherColumns();
        PipeBarrier<PIPE_V>();
        if constexpr (sizeof(T) == sizeof(half)) {
            // Adjacent gathered columns need just one Cast dispatch.
            Cast(coordsUb_, halfCoordsUb_, RoundMode::CAST_NONE, 3U * padded_);
            PipeBarrier<PIPE_V>();
        }

        // Four separately rounded float32 operations. Do not use FMA or
        // replace y1 + (y2-y1) with y2 even when ratio==1.
        const uint8_t repeats = static_cast<uint8_t>(padded_ / kVectorRows);
        const BinaryRepeatParams contiguous{1U, 1U, 1U, 8U, 8U, 8U};
        const UnaryRepeatParams unary{1U, 1U, 8U, 8U};
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(64U);
        Sub<float, false>(y2Ub_, y2Ub_, y1Ub_, MASK_PLACEHOLDER, repeats, contiguous);
        PipeBarrier<PIPE_V>();
        Muls<float, false>(y2Ub_, y2Ub_, ratio_, MASK_PLACEHOLDER, repeats, unary);
        PipeBarrier<PIPE_V>();
        Add<float, false>(y2Ub_, y2Ub_, y1Ub_, MASK_PLACEHOLDER, repeats, contiguous);
        PipeBarrier<PIPE_V>();
        if (hasThreshold_) {
            // Reuse the first 32B block for every block/repeat; avoid the
            // old MTE2_S -> scalar GetValue -> S_V round trip.
            const BinaryRepeatParams broadcast{1U, 1U, 0U, 8U, 8U, 0U};
            Add<float, false>(y2Ub_, y2Ub_, broadcastThresholdUb_, MASK_PLACEHOLDER, repeats, broadcast);
        } else {
            Adds<float, false>(y2Ub_, y2Ub_, defaultThreshold_, MASK_PLACEHOLDER, repeats, unary);
        }
        PipeBarrier<PIPE_V>();
        ResetMask();
        Compare(maskUb_, cyUb_, y2Ub_, CMPMODE::GT, padded_);
        PipeBarrier<PIPE_V>();
        Select(outUb_, maskUb_, onesUb_, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, padded_);
        if (outputIsHalf_) {
            PipeBarrier<PIPE_V>();
            Cast(halfOutUb_, outUb_, RoundMode::CAST_NONE, padded_);
        }
        constexpr event_t ready = static_cast<event_t>(0);
        SetFlag<HardEvent::V_MTE3>(ready);
        WaitFlag<HardEvent::V_MTE3>(ready);
        if (outputIsHalf_) {
            const DataCopyExtParams copy{1U, rows_ * static_cast<uint32_t>(sizeof(half)), 0U, 0U, 0U};
            DataCopyPad(halfOutGm_[begin_], halfOutUb_, copy);
        } else {
            const DataCopyExtParams copy{1U, rows_ * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(outGm_[begin_], outUb_, copy);
        }
    }

private:
    __aicore__ inline void CopyIn() {
        const DataCopyExtParams boxesCopy{1U, rows_ * 4U * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        const DataCopyExtParams anchorsCopy{1U, rows_ * 2U * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        const uint8_t boxPad = static_cast<uint8_t>((Align32(boxesCopy.blockLen) - boxesCopy.blockLen) / sizeof(T));
        const uint8_t anchorPad = static_cast<uint8_t>((Align32(anchorsCopy.blockLen) - anchorsCopy.blockLen) / sizeof(T));
        DataCopyPad(boxesUb_, boxesGm_[begin_ * 4U], boxesCopy,
                    DataCopyPadExtParams<T>{true, 0U, boxPad, static_cast<T>(0)});
        DataCopyPad(anchorsUb_, anchorsGm_[begin_ * 2U], anchorsCopy,
                    DataCopyPadExtParams<T>{true, 0U, anchorPad, static_cast<T>(0)});
        if (hasThreshold_) {
            if (thresholdIsHalf_) {
                const DataCopyExtParams copy{1U, static_cast<uint32_t>(sizeof(half)), 0U, 0U, 0U};
                DataCopyPad(thresholdUb_.ReinterpretCast<half>(), halfThresholdGm_, copy,
                            DataCopyPadExtParams<half>{true, 0U, 0U, static_cast<half>(0)});
            } else {
                const DataCopyExtParams copy{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
                DataCopyPad(thresholdUb_.ReinterpretCast<float>(), thresholdGm_, copy,
                            DataCopyPadExtParams<float>{true, 0U, 0U, 0.0f});
            }
        }
        constexpr event_t ready = static_cast<event_t>(0);
        SetFlag<HardEvent::MTE2_V>(ready);
        WaitFlag<HardEvent::MTE2_V>(ready);
    }

    __aicore__ inline void InitializeInputTail() {
        // DMA pads only through the next 32B boundary. After MTE2_V, zero
        // the remaining UB suffix up to 64 rows. No extra GM access/event.
        if (rows_ != padded_) {
            const uint32_t boxesReady = Align32(rows_ * 4U * sizeof(T)) / sizeof(T);
            const uint32_t anchorsReady = Align32(rows_ * 2U * sizeof(T)) / sizeof(T);
            if (boxesReady < padded_ * 4U) {
                Duplicate(boxesUb_[boxesReady], static_cast<T>(0), padded_ * 4U - boxesReady);
            }
            if (anchorsReady < padded_ * 2U) {
                Duplicate(anchorsUb_[anchorsReady], static_cast<T>(0), padded_ * 2U - anchorsReady);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrepareThreshold() {
        // Documented DataCopyPad dummy behavior: with BOTH explicit padding
        // counts zero, a non-32B input is filled to 32B with its FIRST value.
        // A scalar float DMA therefore already broadcasts threshold eight
        // times. A scalar half DMA does likewise; convert eight copies.
        // Do not set rightPadding on the threshold DMA: that would instead
        // fill with paddingValue and break this broadcast.
        if (hasThreshold_ && thresholdIsHalf_) {
            Cast(floatThresholdUb_, thresholdUb_.ReinterpretCast<half>(), RoundMode::CAST_NONE, 8U);
        }
    }

    __aicore__ inline void GatherColumns() {
        // Older CANN versions limit GatherMask to 255 repeats. A single-core
        // B=4096 float32 box gather would need 256; split only that large path.
        if (padded_ > 2048U) {
            GatherColumnsRange(0U, 2048U);
            GatherColumnsRange(2048U, padded_ - 2048U);
        } else {
            GatherColumnsRange(0U, padded_);
        }
    }

    __aicore__ inline void GatherColumnsRange(uint32_t offset, uint32_t count) {
        const GatherMaskParams boxes{1U, static_cast<uint16_t>(count * 4U * sizeof(T) / 256U), 8U, 0U};
        const GatherMaskParams anchors{1U, static_cast<uint16_t>(count * 2U * sizeof(T) / 256U), 8U, 0U};
        uint64_t reserved = 0U;
        if constexpr (sizeof(T) == sizeof(half)) {
            GatherMask(halfCoordsUb_[offset], boxesUb_[4U * offset], uint8_t{4U}, false, 0U, boxes, reserved);
            GatherMask(halfCoordsUb_[padded_ + offset], boxesUb_[4U * offset], uint8_t{6U}, false, 0U, boxes, reserved);
            GatherMask(halfCoordsUb_[2U * padded_ + offset], anchorsUb_[2U * offset], uint8_t{2U}, false, 0U, anchors, reserved);
        } else {
            GatherMask(y1Ub_[offset], boxesUb_[4U * offset], uint8_t{4U}, false, 0U, boxes, reserved);
            GatherMask(y2Ub_[offset], boxesUb_[4U * offset], uint8_t{6U}, false, 0U, boxes, reserved);
            GatherMask(cyUb_[offset], anchorsUb_[2U * offset], uint8_t{2U}, false, 0U, anchors, reserved);
        }
    }

    LocalTensor<T> boxesUb_, anchorsUb_;
    LocalTensor<uint8_t> thresholdUb_, maskUb_;
    LocalTensor<float> coordsUb_, y1Ub_, y2Ub_, cyUb_, onesUb_;
    LocalTensor<float> floatThresholdUb_, broadcastThresholdUb_, outUb_;
    LocalTensor<half> halfCoordsUb_, halfOutUb_;
    GlobalTensor<T> boxesGm_, anchorsGm_;
    GlobalTensor<float> thresholdGm_, outGm_;
    GlobalTensor<half> halfThresholdGm_, halfOutGm_;
    uint32_t begin_ = 0U, rows_ = 0U, padded_ = 0U;
    float ratio_ = 0.0f, defaultThreshold_ = 0.0f;
    bool hasThreshold_ = false, thresholdIsHalf_ = false, outputIsHalf_ = false;
};

template <typename DT_PERSON_BOXES>
__global__ __aicore__ void safety_belt_violation_geometry(
    GM_ADDR person_boxes, GM_ADDR anchor_points, GM_ADDR threshold,
    GM_ADDR violation_score, GM_ADDR workspace, GM_ADDR tiling) {
    InitSocState();
    REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
    GET_TILING_DATA_WITH_STRUCT(SafetyBeltViolationGeometryTilingData, tiling_data, tiling);
    if (tiling_data.batchSize <= kVectorRows) {
        RunSafetyBeltSmall<DT_PERSON_BOXES>(person_boxes, anchor_points, threshold,
                                         violation_score, tiling_data);
    } else {
        KernelSafetyBeltViolationGeometry<DT_PERSON_BOXES> op;
        op.Init(person_boxes, anchor_points, threshold, violation_score, tiling_data);
        op.Process();
    }
    PipeBarrier<PIPE_ALL>();
}
