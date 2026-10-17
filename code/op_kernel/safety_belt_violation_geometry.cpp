// SBVG_VERSION: v8_staged_gather
#ifndef SBVG_SMALL_BACKEND
#define SBVG_SMALL_BACKEND 3
#endif
#ifndef ASCENDC_DUMP
#define ASCENDC_DUMP 0
#endif
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include "kernel_operator.h"
#include "safety_belt_violation_geometry_tiling.h"
#include "tiling_key_safety_belt_violation_geometry.h"
#include "safety_belt_violation_geometry_small.h"
#include "safety_belt_violation_geometry_strided.h"
#include "safety_belt_violation_geometry_gather.h"
#include "safety_belt_violation_geometry_ratio.h"
#include "safety_belt_violation_geometry_paired.h"
#include "safety_belt_violation_geometry_staged.h"

using namespace AscendC;

namespace {
constexpr uint32_t kVectorRows = 64U;
constexpr uint32_t kMaxRows = 4096U;
__aicore__ inline uint32_t Align32(uint32_t bytes) {
    return (bytes + 31U) / 32U * 32U;
}
}  // namespace

template <class T, uint32_t IO_MODE, uint32_t FixedRows = 0U, uint32_t FixedBatch = 0U,
          bool DynamicRatio = false>
class KernelSafetyBeltViolationGeometry {
public:
    __aicore__ inline void Init(
        GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
        GM_ADDR violationScore, const SafetyBeltViolationGeometryTilingData& data,
        GM_ADDR tiling = nullptr) {
        const uint32_t coreRows = data.rowsPerCore;
        if constexpr (FixedRows != 0U && FixedBatch != 0U && FixedBatch <= FixedRows) {
            // Metadata dispatch guarantees that this partial shape uses one core.
            begin_ = 0U;
            rows_ = FixedBatch;
            padded_ = FixedRows;
        } else if constexpr (FixedRows != 0U) {
            // Whole partitions keep the constant per-core amount of work.
            begin_ = GetBlockIdx() * coreRows;
            rows_ = FixedRows;
            padded_ = FixedRows;
        } else {
            begin_ = GetBlockIdx() * coreRows;
            rows_ = data.batchSize - begin_;  // Host launches only nonempty blocks.
            if (rows_ > coreRows) {
                rows_ = coreRows;
            }
            padded_ = (rows_ + kVectorRows - 1U) / kVectorRows * kVectorRows;
        }
        ratio_ = data.waistRatio;
        defaultThreshold_ = data.defaultThreshold;
        tiling_ = tiling;

        boxesGm_.SetGlobalBuffer((__gm__ T*)personBoxes, data.batchSize * 4U);
        anchorsGm_.SetGlobalBuffer((__gm__ T*)anchorPoints, data.batchSize * 2U);
        if constexpr (outputIsHalf_) {
            halfOutGm_.SetGlobalBuffer((__gm__ half*)violationScore, data.batchSize);
        } else {
            outGm_.SetGlobalBuffer((__gm__ float*)violationScore, data.batchSize);
        }
        // Never bind or read an absent threshold pointer.
        if constexpr (hasThreshold_) {
            if constexpr (thresholdIsHalf_) {
                halfThresholdGm_.SetGlobalBuffer((__gm__ half*)threshold, 1U);
            } else {
                thresholdGm_.SetGlobalBuffer((__gm__ float*)threshold, 1U);
            }
        }

        // Static tensors share the physical UB address space. These regions
        // never overlap, and each address is 32B aligned. No TPipe/TBuf setup.
        constexpr uint32_t boxBytes = kMaxRows * 4U * sizeof(T);
        constexpr uint32_t anchorBytes = kMaxRows * 2U * sizeof(T);
        constexpr uint32_t calcStart = boxBytes + anchorBytes + 32U;
        boxesUb_ = LocalTensor<T>(TPosition::VECIN, 0U, 4U * kMaxRows);
        anchorsUb_ = LocalTensor<T>(TPosition::VECIN, boxBytes, 2U * kMaxRows);
        thresholdUb_ = LocalTensor<uint8_t>(TPosition::VECIN, boxBytes + anchorBytes, 32U);

        constexpr uint32_t coordBytes = kMaxRows * 3U * sizeof(float);
        constexpr uint32_t halfCoordBytes = sizeof(T) == sizeof(half) ? kMaxRows * 3U * sizeof(half) : 0U;
        constexpr uint32_t onesOffset = calcStart + coordBytes + halfCoordBytes;
        constexpr uint32_t zeroOffset = onesOffset + 32U;
        constexpr uint32_t maskOffset = zeroOffset + 32U;
        constexpr uint32_t maskBytes = kMaxRows / 8U;
        constexpr uint32_t thresholdOffset = maskOffset + maskBytes;
        constexpr uint32_t outputOffset = thresholdOffset + 32U;
        constexpr uint32_t outputBytes = kMaxRows * sizeof(float);
        constexpr uint32_t halfOutputBytes = outputIsHalf_ ? kMaxRows * sizeof(half) : 0U;
        constexpr uint32_t ratioOffset = outputOffset + outputBytes + halfOutputBytes;
        ratioUb_ = LocalTensor<float>(TPosition::VECIN, ratioOffset, 8U);
        static_assert(ratioOffset + 32U + 8192U <= 192U * 1024U,
                      "Dynamic ratio needs one more disjoint 32B UB block");
        // Leave the SDK's 8KiB temporary-buffer reserve available on A2.
        static_assert(outputOffset + outputBytes + halfOutputBytes + 8192U <= 192U * 1024U,
                      "Static tensors must leave room for the SDK UB reserve");
        coordsUb_ = LocalTensor<float>(TPosition::VECCALC, calcStart, 3U * kMaxRows);
        y1Ub_ = coordsUb_;
        // Tight active columns permit one half->float Cast, including tails.
        y2Ub_ = coordsUb_[padded_];
        cyUb_ = coordsUb_[2U * padded_];
        if constexpr (sizeof(T) == sizeof(half)) {
            halfCoordsUb_ = LocalTensor<half>(TPosition::VECCALC, calcStart + coordBytes, 3U * kMaxRows);
        }
        onesUb_ = LocalTensor<float>(TPosition::VECCALC, onesOffset, 8U);
        zerosUb_ = LocalTensor<float>(TPosition::VECCALC, zeroOffset, 8U);
        maskUb_ = LocalTensor<uint8_t>(TPosition::VECCALC, maskOffset, maskBytes);
        floatThresholdUb_ = LocalTensor<float>(TPosition::VECCALC, thresholdOffset, 8U);
        if constexpr (thresholdIsHalf_) {
            broadcastThresholdUb_ = floatThresholdUb_;
        } else {
            broadcastThresholdUb_ = thresholdUb_.ReinterpretCast<float>();
        }

        outUb_ = LocalTensor<float>(TPosition::VECOUT, outputOffset, kMaxRows);
        if constexpr (outputIsHalf_) {
            halfOutUb_ = LocalTensor<half>(TPosition::VECOUT, outputOffset + outputBytes, kMaxRows);
        }
    }

    __aicore__ inline void Process() {
        // Independent vector work overlaps input DMA.
        Duplicate(onesUb_, 1.0f, 8U);
        Duplicate(zerosUb_, 0.0f, 8U);
        InitializeCoordinateTail();
        CopyIn();
        PrepareThreshold();
        GatherColumns();
        PipeBarrier<PIPE_V>();
        // Mode 1 Select takes its false value from CMPMASK. Set it once after
        // the constants are ready; memory-output Compare leaves it unchanged.
        SetCmpMask(zerosUb_);
        if constexpr (sizeof(T) == sizeof(half)) {
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
        safety_belt::MultiplyRatioForKnownShape<DynamicRatio>(
            y2Ub_, y2Ub_, ratioUb_, ratio_, repeats);
        PipeBarrier<PIPE_V>();
        Add<float, false>(y2Ub_, y2Ub_, y1Ub_, MASK_PLACEHOLDER, repeats, contiguous);
        PipeBarrier<PIPE_V>();
        if constexpr (hasThreshold_) {
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
        // Counter mode advances the compact comparison bits automatically.
        // This direct overload avoids the scalar Select's temporary zero fill.
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(padded_);
        const BinaryRepeatParams selectBroadcast{1U, 0U, 1U, 8U, 0U, 8U};
        Select(outUb_, maskUb_, onesUb_, uint8_t{1U}, selectBroadcast);
        ResetMask();
        SetMaskNorm();
        if constexpr (outputIsHalf_) {
            PipeBarrier<PIPE_V>();
            Cast(halfOutUb_, outUb_, RoundMode::CAST_NONE, padded_);
        }
        constexpr event_t ready = static_cast<event_t>(0);
        SetFlag<HardEvent::V_MTE3>(ready);
        WaitFlag<HardEvent::V_MTE3>(ready);
        if constexpr (outputIsHalf_) {
            if (rows_ * sizeof(half) % 32U == 0U) {
                DataCopy(halfOutGm_[begin_], halfOutUb_, rows_);
            } else {
                const DataCopyExtParams copy{1U, rows_ * static_cast<uint32_t>(sizeof(half)), 0U, 0U, 0U};
                DataCopyPad(halfOutGm_[begin_], halfOutUb_, copy);
            }
        } else {
            if (rows_ * sizeof(float) % 32U == 0U) {
                DataCopy(outGm_[begin_], outUb_, rows_);
            } else {
                const DataCopyExtParams copy{1U, rows_ * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
                DataCopyPad(outGm_[begin_], outUb_, copy);
            }
        }
    }

private:
    __aicore__ inline void CopyIn() {
        const DataCopyExtParams boxesCopy{1U, rows_ * 4U * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        const DataCopyExtParams anchorsCopy{1U, rows_ * 2U * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        const uint8_t boxPad = static_cast<uint8_t>((Align32(boxesCopy.blockLen) - boxesCopy.blockLen) / sizeof(T));
        const uint8_t anchorPad = static_cast<uint8_t>((Align32(anchorsCopy.blockLen) - anchorsCopy.blockLen) / sizeof(T));
        if (boxesCopy.blockLen % 32U == 0U) {
            DataCopy(boxesUb_, boxesGm_[begin_ * 4U], rows_ * 4U);
        } else {
            DataCopyPad(boxesUb_, boxesGm_[begin_ * 4U], boxesCopy,
                        DataCopyPadExtParams<T>{true, 0U, boxPad, static_cast<T>(0)});
        }
        if (anchorsCopy.blockLen % 32U == 0U) {
            DataCopy(anchorsUb_, anchorsGm_[begin_ * 2U], rows_ * 2U);
        } else {
            DataCopyPad(anchorsUb_, anchorsGm_[begin_ * 2U], anchorsCopy,
                        DataCopyPadExtParams<T>{true, 0U, anchorPad, static_cast<T>(0)});
        }
        if constexpr (hasThreshold_) {
            if constexpr (thresholdIsHalf_) {
                const DataCopyExtParams copy{1U, static_cast<uint32_t>(sizeof(half)), 0U, 0U, 0U};
                DataCopyPad(thresholdUb_.ReinterpretCast<half>(), halfThresholdGm_, copy,
                            DataCopyPadExtParams<half>{true, 0U, 0U, static_cast<half>(0)});
            } else {
                const DataCopyExtParams copy{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
                DataCopyPad(thresholdUb_.ReinterpretCast<float>(), thresholdGm_, copy,
                            DataCopyPadExtParams<float>{true, 0U, 0U, 0.0f});
            }
        }
        safety_belt::CopyRatioForKnownShape<DynamicRatio>(tiling_, ratioUb_);
        // Complete compact-column prezero while the independent MTE2 work is
        // in flight. Gather may overwrite the active prefix of these blocks.
        if (rows_ != padded_) {
            PipeBarrier<PIPE_V>();
        }
        constexpr event_t ready = static_cast<event_t>(0);
        SetFlag<HardEvent::MTE2_V>(ready);
        WaitFlag<HardEvent::MTE2_V>(ready);
    }

    __aicore__ inline void InitializeCoordinateTail() {
        // Only the final compact 64-row block can contain inactive values.
        // Start before its active prefix so every Duplicate address remains
        // 32B aligned; subsequent Gather replaces that prefix with real data.
        if (rows_ != padded_) {
            const uint32_t offset = rows_ / kVectorRows * kVectorRows;
            if constexpr (sizeof(T) == sizeof(half)) {
                Duplicate(halfCoordsUb_[offset], static_cast<half>(0), kVectorRows);
                Duplicate(halfCoordsUb_[padded_ + offset], static_cast<half>(0), kVectorRows);
                Duplicate(halfCoordsUb_[2U * padded_ + offset], static_cast<half>(0), kVectorRows);
            } else {
                Duplicate(y1Ub_[offset], 0.0f, kVectorRows);
                Duplicate(y2Ub_[offset], 0.0f, kVectorRows);
                Duplicate(cyUb_[offset], 0.0f, kVectorRows);
            }
        }
    }

    __aicore__ inline void PrepareThreshold() {
        // Documented DataCopyPad dummy behavior: with BOTH explicit padding
        // counts zero, a non-32B input is filled to 32B with its FIRST value.
        // A scalar float DMA therefore already broadcasts threshold eight
        // times. A scalar half DMA does likewise; convert eight copies.
        // Do not set rightPadding on the threshold DMA: that would instead
        // fill with paddingValue and break this broadcast.
        if constexpr (hasThreshold_ && thresholdIsHalf_) {
            Cast(floatThresholdUb_, thresholdUb_.ReinterpretCast<half>(), RoundMode::CAST_NONE, 8U);
        }
    }

    __aicore__ inline void GatherColumns() {
        if (rows_ != padded_) {
            // Counter mode gathers only valid source coordinates. Source UB
            // suffixes are neither read nor initialized on a partial tile.
            const GatherMaskParams gather{1U, 1U, 8U, 0U};
            SetMaskCount();
            SetVectorMask<T, MaskMode::COUNTER>(rows_ * 4U);
            if constexpr (sizeof(T) == sizeof(half)) {
                safety_belt::GatherFixed(halfCoordsUb_, boxesUb_, uint8_t{4U}, true, rows_ * 4U, gather);
                safety_belt::GatherFixed(halfCoordsUb_[padded_], boxesUb_, uint8_t{6U}, true, rows_ * 4U, gather);
                SetVectorMask<T, MaskMode::COUNTER>(rows_ * 2U);
                safety_belt::GatherFixed(halfCoordsUb_[2U * padded_], anchorsUb_, uint8_t{2U}, true, rows_ * 2U, gather);
            } else {
                safety_belt::GatherFixed(y1Ub_, boxesUb_, uint8_t{4U}, true, rows_ * 4U, gather);
                safety_belt::GatherFixed(y2Ub_, boxesUb_, uint8_t{6U}, true, rows_ * 4U, gather);
                SetVectorMask<T, MaskMode::COUNTER>(rows_ * 2U);
                safety_belt::GatherFixed(cyUb_, anchorsUb_, uint8_t{2U}, true, rows_ * 2U, gather);
            }
            SetMaskNorm();
            return;
        }
        SetMaskNorm();
        SetVectorMask<T, MaskMode::NORMAL>(256U / sizeof(T));
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
        if constexpr (sizeof(T) == sizeof(half)) {
            safety_belt::GatherFixed(halfCoordsUb_[offset], boxesUb_[4U * offset], uint8_t{4U}, false, 0U, boxes);
            safety_belt::GatherFixed(halfCoordsUb_[padded_ + offset], boxesUb_[4U * offset], uint8_t{6U}, false, 0U, boxes);
            safety_belt::GatherFixed(halfCoordsUb_[2U * padded_ + offset], anchorsUb_[2U * offset], uint8_t{2U}, false, 0U, anchors);
        } else {
            safety_belt::GatherFixed(y1Ub_[offset], boxesUb_[4U * offset], uint8_t{4U}, false, 0U, boxes);
            safety_belt::GatherFixed(y2Ub_[offset], boxesUb_[4U * offset], uint8_t{6U}, false, 0U, boxes);
            safety_belt::GatherFixed(cyUb_[offset], anchorsUb_[2U * offset], uint8_t{2U}, false, 0U, anchors);
        }
    }

    LocalTensor<T> boxesUb_, anchorsUb_;
    LocalTensor<uint8_t> thresholdUb_, maskUb_;
    LocalTensor<float> coordsUb_, y1Ub_, y2Ub_, cyUb_, onesUb_, zerosUb_;
    LocalTensor<float> floatThresholdUb_, broadcastThresholdUb_, outUb_, ratioUb_;
    GM_ADDR tiling_ = nullptr;
    LocalTensor<half> halfCoordsUb_, halfOutUb_;
    GlobalTensor<T> boxesGm_, anchorsGm_;
    GlobalTensor<float> thresholdGm_, outGm_;
    GlobalTensor<half> halfThresholdGm_, halfOutGm_;
    uint32_t begin_ = 0U, rows_ = 0U, padded_ = 0U;
    float ratio_ = 0.0f, defaultThreshold_ = 0.0f;
    static constexpr bool hasThreshold_ = (IO_MODE & 1U) != 0U;
    static constexpr bool thresholdIsHalf_ = (IO_MODE & 2U) != 0U;
    static constexpr bool outputIsHalf_ = (IO_MODE & 4U) != 0U;
    static_assert(IO_MODE < 8U && (!thresholdIsHalf_ || hasThreshold_),
                  "Invalid optional-threshold template mode");
    static_assert(sizeof(T) == sizeof(half) || (IO_MODE & 6U) == 0U,
                  "Half IO compatibility requires half coordinates");
};

template <class T, uint32_t IO_MODE, bool DynamicRatio = false>
__aicore__ inline void RunSafetyBeltChosenSmall(
    GM_ADDR boxes, GM_ADDR anchors, GM_ADDR threshold, GM_ADDR output,
    const SafetyBeltViolationGeometryTilingData& data, GM_ADDR tiling = nullptr) {
#if SBVG_SMALL_BACKEND == 3
    RunSafetyBeltStagedSmall<T, IO_MODE, DynamicRatio>(boxes, anchors, threshold, output, data, tiling);
#elif SBVG_SMALL_BACKEND == 2
    RunSafetyBeltPairedSmall<T, IO_MODE, DynamicRatio>(boxes, anchors, threshold, output, data, tiling);
#elif SBVG_SMALL_BACKEND == 1
    RunSafetyBeltStridedSmall<T, IO_MODE, DynamicRatio>(boxes, anchors, threshold, output, data, tiling);
#else
    RunSafetyBeltSmall<T, IO_MODE, DynamicRatio>(boxes, anchors, threshold, output, data, tiling);
#endif
}

template <typename DT_PERSON_BOXES, uint32_t IO_MODE, uint32_t STATIC_CASE>
__global__ __aicore__ void safety_belt_violation_geometry(
    GM_ADDR person_boxes, GM_ADDR anchor_points, GM_ADDR threshold,
    GM_ADDR violation_score, GM_ADDR workspace, GM_ADDR tiling) {
    InitSocState();
    REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
    static_assert(STATIC_CASE <= safety_belt::kStaticShapeCount * 4U,
                  "Unknown static metadata case");
    if constexpr (STATIC_CASE > safety_belt::kStaticShapeCount * 3U) {
        // Shape and partition are constants; only the real float32 ratio is
        // read from tiling GM. It shares input DMA's existing MTE2_V event.
        constexpr uint32_t shapeIndex = STATIC_CASE - 1U - safety_belt::kStaticShapeCount * 3U;
        constexpr uint32_t rows = safety_belt::StaticBatchForShape(shapeIndex);
        constexpr uint32_t coreRows = safety_belt::StaticCoreRowsForShape(shapeIndex);
        const SafetyBeltViolationGeometryTilingData data{rows, coreRows, 0.0f, 0.0f};
        if constexpr (rows <= kVectorRows) {
            RunSafetyBeltChosenSmall<DT_PERSON_BOXES, IO_MODE, true>(
                person_boxes, anchor_points, threshold, violation_score, data, tiling);
        } else {
            KernelSafetyBeltViolationGeometry<DT_PERSON_BOXES, IO_MODE, coreRows, rows, true> op;
            op.Init(person_boxes, anchor_points, threshold, violation_score, data, tiling);
            op.Process();
        }
    } else if constexpr (STATIC_CASE != 0U) {
        // Specialize only metadata; always load and compute the actual tensors.
        // No GM Tiling load precedes input DMA on these metadata-specialized kernels.
        constexpr uint32_t shapeIndex = (STATIC_CASE - 1U) / 3U;
        constexpr uint32_t ratioIndex = (STATIC_CASE - 1U) % 3U;
        constexpr uint32_t rows = safety_belt::StaticBatchForShape(shapeIndex);
        constexpr uint32_t coreRows = safety_belt::StaticCoreRowsForShape(shapeIndex);
        constexpr float ratio = ratioIndex == 0U ? 0.8f : (ratioIndex == 1U ? 0.9f : 1.0f);
        const SafetyBeltViolationGeometryTilingData data{rows, coreRows, ratio, 0.0f};
        if constexpr (rows <= kVectorRows) {
            RunSafetyBeltChosenSmall<DT_PERSON_BOXES, IO_MODE>(
                person_boxes, anchor_points, threshold, violation_score, data);
        } else {
            KernelSafetyBeltViolationGeometry<DT_PERSON_BOXES, IO_MODE, coreRows, rows> op;
            op.Init(person_boxes, anchor_points, threshold, violation_score, data);
            op.Process();
        }
    } else {
        GET_TILING_DATA_WITH_STRUCT(SafetyBeltViolationGeometryTilingData, tiling_data, tiling);
        if (tiling_data.batchSize <= kVectorRows) {
            RunSafetyBeltChosenSmall<DT_PERSON_BOXES, IO_MODE>(
                person_boxes, anchor_points, threshold, violation_score, tiling_data);
        } else {
            KernelSafetyBeltViolationGeometry<DT_PERSON_BOXES, IO_MODE> op;
            op.Init(person_boxes, anchor_points, threshold, violation_score, tiling_data);
            op.Process();
        }
    }
    PipeBarrier<PIPE_ALL>();
}
