#pragma once

#include "kernel_operator.h"
#include "safety_belt_violation_geometry_inline.h"
#include "safety_belt_violation_geometry_tiling.h"
#include "safety_belt_violation_geometry_ratio.h"

// Experimental B <= 64 path: MTE2 extracts each coordinate directly into a
// separate 32B UB row. With no explicit left/right padding, DataCopyPad fills
// the remainder of that row with its first element. No GatherMask is needed.
// InitSocState and the final PIPE_ALL barrier belong to the caller.
template <class T, uint32_t IO_MODE, bool DynamicRatio = false>
SBVG_FORCE_INLINE void RunSafetyBeltStridedSmall(
    GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
    GM_ADDR violationScore, const SafetyBeltViolationGeometryTilingData& data,
    GM_ADDR tiling = nullptr) {
    using namespace AscendC;
    constexpr uint32_t kRows = 64U;
    constexpr uint32_t kFloatRowElements = 8U;
    constexpr uint32_t kHalfRowElements = 16U;
    constexpr uint32_t kColumnBytes = kRows * 32U;
    constexpr uint32_t kFloatCoordsAddress = 0U;
    constexpr uint32_t kHalfCoordsAddress = 3U * kColumnBytes;
    constexpr uint32_t kThresholdAddress = kHalfCoordsAddress + 3U * kColumnBytes;
    constexpr uint32_t kFloatThresholdAddress = kThresholdAddress + 32U;
    constexpr uint32_t kOnesAddress = kFloatThresholdAddress + 32U;
    constexpr uint32_t kZerosAddress = kOnesAddress + 32U;
    constexpr uint32_t kMaskAddress = kZerosAddress + 32U;
    constexpr uint32_t kMaskBytes = kRows * kFloatRowElements / 8U;
    constexpr uint32_t kOutputAddress = kMaskAddress + kMaskBytes;
    constexpr uint32_t kHalfOutputAddress = kOutputAddress + kColumnBytes;
    static_assert(kHalfOutputAddress + kColumnBytes == 16576U,
                  "Strided-small UB regions must remain disjoint and 32B aligned");
    constexpr bool hasThreshold = (IO_MODE & 1U) != 0U;
    constexpr bool thresholdIsHalf = (IO_MODE & 2U) != 0U;
    constexpr bool outputIsHalf = (IO_MODE & 4U) != 0U;
    static_assert(IO_MODE < 8U && (!thresholdIsHalf || hasThreshold),
                  "Invalid optional-threshold template mode");
    static_assert(sizeof(T) == sizeof(half) || (IO_MODE & 6U) == 0U,
                  "Half IO compatibility requires half coordinates");

    const uint32_t rows = data.batchSize;
    LocalTensor<float> ratioUb(TPosition::VECIN, 16576U, 8U);
    // Eight 32B rows fill one 256B float vector. The inactive suffix contains
    // UB-only zeros, never extra GM reads, so memory-output Compare is safe.
    const uint32_t paddedRows = (rows + 7U) / 8U * 8U;
    const uint32_t valueCount = paddedRows * kFloatRowElements;
    constexpr uint32_t inputAddress = sizeof(T) == sizeof(half)
                                          ? kHalfCoordsAddress
                                          : kFloatCoordsAddress;
    constexpr uint32_t inputColumnElements = kColumnBytes / sizeof(T);
    LocalTensor<T> inputY1(TPosition::VECIN, inputAddress, inputColumnElements);
    LocalTensor<T> inputY2(TPosition::VECIN, inputAddress + kColumnBytes,
                           inputColumnElements);
    LocalTensor<T> inputCy(TPosition::VECIN, inputAddress + 2U * kColumnBytes,
                           inputColumnElements);
    LocalTensor<float> y1(TPosition::VECCALC, kFloatCoordsAddress,
                           kRows * kFloatRowElements);
    LocalTensor<float> y2(TPosition::VECCALC, kFloatCoordsAddress + kColumnBytes,
                           kRows * kFloatRowElements);
    LocalTensor<float> cy(TPosition::VECCALC, kFloatCoordsAddress + 2U * kColumnBytes,
                           kRows * kFloatRowElements);
    LocalTensor<uint8_t> thresholdBytes(TPosition::VECIN, kThresholdAddress, 32U);
    LocalTensor<float> floatThreshold(TPosition::VECCALC, kFloatThresholdAddress, 8U);
    LocalTensor<float> ones(TPosition::VECCALC, kOnesAddress, 8U);
    LocalTensor<float> zeros(TPosition::VECCALC, kZerosAddress, 8U);
    LocalTensor<uint8_t> comparison(TPosition::VECCALC, kMaskAddress, kMaskBytes);
    LocalTensor<float> output(TPosition::VECOUT, kOutputAddress,
                               kRows * kFloatRowElements);
    LocalTensor<half> halfOutput(TPosition::VECOUT, kHalfOutputAddress,
                                  kRows * kHalfRowElements);

    GlobalTensor<T> boxesGm, anchorsGm;
    boxesGm.SetGlobalBuffer((__gm__ T*)personBoxes, rows * 4U);
    anchorsGm.SetGlobalBuffer((__gm__ T*)anchorPoints, rows * 2U);
    Duplicate(ones, 1.0f, 8U);
    Duplicate(zeros, 0.0f, 8U);

    // GM strides are bytes after a block, UB strides are 32B blocks after its
    // padded extent. Every read below remains inside its own valid input row.
    const DataCopyExtParams boxesCopy{static_cast<uint16_t>(rows),
        static_cast<uint32_t>(sizeof(T)), 3U * static_cast<uint32_t>(sizeof(T)), 0U, 0U};
    const DataCopyExtParams anchorsCopy{static_cast<uint16_t>(rows),
        static_cast<uint32_t>(sizeof(T)), static_cast<uint32_t>(sizeof(T)), 0U, 0U};
    const DataCopyPadExtParams<T> pad{true, 0U, 0U, static_cast<T>(0)};
    DataCopyPad(inputY1, boxesGm[1U], boxesCopy, pad);
    DataCopyPad(inputY2, boxesGm[3U], boxesCopy, pad);
    DataCopyPad(inputCy, anchorsGm[1U], anchorsCopy, pad);
    if constexpr (hasThreshold) {
        if constexpr (thresholdIsHalf) {
            GlobalTensor<half> thresholdGm;
            thresholdGm.SetGlobalBuffer((__gm__ half*)threshold, 1U);
            const DataCopyExtParams copy{1U, static_cast<uint32_t>(sizeof(half)), 0U, 0U, 0U};
            DataCopyPad(thresholdBytes.ReinterpretCast<half>(), thresholdGm, copy,
                        DataCopyPadExtParams<half>{true, 0U, 0U, static_cast<half>(0)});
        } else {
            GlobalTensor<float> thresholdGm;
            thresholdGm.SetGlobalBuffer((__gm__ float*)threshold, 1U);
            const DataCopyExtParams copy{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(thresholdBytes.ReinterpretCast<float>(), thresholdGm, copy,
                        DataCopyPadExtParams<float>{true, 0U, 0U, 0.0f});
        }
    }
    safety_belt::CopyRatioForKnownShape<DynamicRatio>(tiling, ratioUb);
    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));

    if constexpr (hasThreshold && thresholdIsHalf) {
        Cast(floatThreshold, thresholdBytes.ReinterpretCast<half>(), RoundMode::CAST_NONE, 8U);
    }
    if constexpr (sizeof(T) == sizeof(half)) {
        // One half row occupies 16 half elements; use only its first eight.
        // Each becomes one 8-float row, preserving identical replicated lanes.
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(8U);
        const UnaryRepeatParams castRows{1U, 1U, 1U, 1U};
        Cast<float, half, false>(y1, inputY1, RoundMode::CAST_NONE,
                                MASK_PLACEHOLDER, static_cast<uint8_t>(rows), castRows);
        Cast<float, half, false>(y2, inputY2, RoundMode::CAST_NONE,
                                MASK_PLACEHOLDER, static_cast<uint8_t>(rows), castRows);
        Cast<float, half, false>(cy, inputCy, RoundMode::CAST_NONE,
                                MASK_PLACEHOLDER, static_cast<uint8_t>(rows), castRows);
        ResetMask();
    }
    if (rows != paddedRows) {
        const uint32_t offset = rows * kFloatRowElements;
        const uint32_t tail = (paddedRows - rows) * kFloatRowElements;
        // DMA completed before touching these physical coordinate allocations.
        Duplicate(y1[offset], 0.0f, tail);
        Duplicate(y2[offset], 0.0f, tail);
        Duplicate(cy[offset], 0.0f, tail);
    }
    PipeBarrier<PIPE_V>();
    SetCmpMask(zeros);

    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(64U);
    const uint8_t repeats = static_cast<uint8_t>(paddedRows / 8U);
    const BinaryRepeatParams contiguous{1U, 1U, 1U, 8U, 8U, 8U};
    const UnaryRepeatParams unary{1U, 1U, 8U, 8U};
    Sub<float, false>(y2, y2, y1, MASK_PLACEHOLDER, repeats, contiguous);
    PipeBarrier<PIPE_V>();
    safety_belt::MultiplyRatioForKnownShape<DynamicRatio>(
        y2, y2, ratioUb, data.waistRatio, repeats);
    PipeBarrier<PIPE_V>();
    Add<float, false>(y2, y2, y1, MASK_PLACEHOLDER, repeats, contiguous);
    PipeBarrier<PIPE_V>();
    if constexpr (hasThreshold) {
        const LocalTensor<float> value = thresholdIsHalf
            ? floatThreshold : thresholdBytes.ReinterpretCast<float>();
        const BinaryRepeatParams broadcast{1U, 1U, 0U, 8U, 8U, 0U};
        Add<float, false>(y2, y2, value, MASK_PLACEHOLDER, repeats, broadcast);
    } else {
        Adds<float, false>(y2, y2, data.defaultThreshold, MASK_PLACEHOLDER, repeats, unary);
    }
    PipeBarrier<PIPE_V>();
    ResetMask();
    Compare(comparison, cy, y2, CMPMODE::GT, valueCount);
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(valueCount);
    const BinaryRepeatParams selectBroadcast{1U, 0U, 1U, 8U, 0U, 8U};
    Select(output, comparison, ones, uint8_t{1U}, selectBroadcast);
    ResetMask();
    SetMaskNorm();

    if constexpr (outputIsHalf) {
        PipeBarrier<PIPE_V>();
        SetVectorMask<float, MaskMode::NORMAL>(8U);
        const UnaryRepeatParams castRows{1U, 1U, 1U, 1U};
        Cast<half, float, false>(halfOutput, output, RoundMode::CAST_NONE,
                                MASK_PLACEHOLDER, static_cast<uint8_t>(rows), castRows);
        ResetMask();
    }
    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
    WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
    // Compact the first scalar of each 32B UB row directly into GM. Padding is
    // discarded by MTE3, and blockCount never includes the UB-only suffix.
    if constexpr (outputIsHalf) {
        GlobalTensor<half> outputGm;
        outputGm.SetGlobalBuffer((__gm__ half*)violationScore, rows);
        const DataCopyExtParams copy{static_cast<uint16_t>(rows),
            static_cast<uint32_t>(sizeof(half)), 0U, 0U, 0U};
        DataCopyPad(outputGm, halfOutput, copy);
    } else {
        GlobalTensor<float> outputGm;
        outputGm.SetGlobalBuffer((__gm__ float*)violationScore, rows);
        const DataCopyExtParams copy{static_cast<uint16_t>(rows),
            static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(outputGm, output, copy);
    }
}
