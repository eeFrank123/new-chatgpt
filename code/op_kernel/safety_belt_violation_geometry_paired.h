#pragma once

#include "kernel_operator.h"
#include "safety_belt_violation_geometry_inline.h"
#include "safety_belt_violation_geometry_tiling.h"
#include "safety_belt_violation_geometry_ratio.h"

// B <= 64: one strided DMA extracts y1,y2,y1,y2,... as 32B UB rows.
// A second extracts cy. Both box columns have aligned vector addresses, so
// high-dimensional vector strides consume them without GatherMask or shifts.
// InitSocState and the final PIPE_ALL barrier belong to the caller.
template <class T, uint32_t IO_MODE, bool DynamicRatio = false>
SBVG_FORCE_INLINE void RunSafetyBeltPairedSmall(
    GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
    GM_ADDR violationScore, const SafetyBeltViolationGeometryTilingData& data,
    GM_ADDR ratioTiling = nullptr) {
    using namespace AscendC;
    constexpr uint32_t kRows = 64U;
    constexpr uint32_t kFloatRowElements = 8U;
    constexpr uint32_t kHalfRowElements = 16U;
    constexpr uint32_t kColumnBytes = kRows * 32U;
    constexpr uint32_t kFloatBoxesAddress = 0U;
    constexpr uint32_t kFloatCyAddress = 2U * kColumnBytes;
    constexpr uint32_t kHalfBoxesAddress = 3U * kColumnBytes;
    constexpr uint32_t kHalfCyAddress = 5U * kColumnBytes;
    constexpr uint32_t kThresholdAddress = 6U * kColumnBytes;
    constexpr uint32_t kFloatThresholdAddress = kThresholdAddress + 32U;
    constexpr uint32_t kRatioAddress = kFloatThresholdAddress + 32U;
    constexpr uint32_t kOnesAddress = kRatioAddress + 32U;
    constexpr uint32_t kZerosAddress = kOnesAddress + 32U;
    constexpr uint32_t kMaskAddress = kZerosAddress + 32U;
    constexpr uint32_t kMaskBytes = kRows * kFloatRowElements / 8U;
    constexpr uint32_t kCalcAddress = kMaskAddress + kMaskBytes;
    constexpr uint32_t kOutputAddress = kCalcAddress + kColumnBytes;
    constexpr uint32_t kHalfOutputAddress = kOutputAddress + kColumnBytes;
    static_assert(kHalfOutputAddress + kColumnBytes == 18656U,
                  "Paired-small UB regions must be disjoint and 32B aligned");
    static_assert(2U * kRows <= 255U, "Paired box Cast must fit one repeat field");
    constexpr bool hasThreshold = (IO_MODE & 1U) != 0U;
    constexpr bool thresholdIsHalf = (IO_MODE & 2U) != 0U;
    constexpr bool outputIsHalf = (IO_MODE & 4U) != 0U;
    static_assert(IO_MODE < 8U && (!thresholdIsHalf || hasThreshold),
                  "Invalid optional-threshold template mode");
    static_assert(sizeof(T) == sizeof(half) || (IO_MODE & 6U) == 0U,
                  "Half IO compatibility requires half coordinates");

    const uint32_t rows = data.batchSize;
    const uint32_t paddedRows = (rows + 7U) / 8U * 8U;
    const uint32_t valueCount = paddedRows * kFloatRowElements;
    constexpr uint32_t inputBoxesAddress = sizeof(T) == sizeof(half)
        ? kHalfBoxesAddress : kFloatBoxesAddress;
    constexpr uint32_t inputCyAddress = sizeof(T) == sizeof(half)
        ? kHalfCyAddress : kFloatCyAddress;
    LocalTensor<T> inputBoxes(TPosition::VECIN, inputBoxesAddress,
                              2U * kColumnBytes / sizeof(T));
    LocalTensor<T> inputCy(TPosition::VECIN, inputCyAddress,
                           kColumnBytes / sizeof(T));
    LocalTensor<float> floatBoxes(TPosition::VECCALC, kFloatBoxesAddress,
                                   2U * kRows * kFloatRowElements);
    const LocalTensor<float> y1 = floatBoxes;
    const LocalTensor<float> y2 = floatBoxes[kFloatRowElements];
    LocalTensor<float> cy(TPosition::VECCALC, kFloatCyAddress,
                           kRows * kFloatRowElements);
    LocalTensor<uint8_t> thresholdBytes(TPosition::VECIN, kThresholdAddress, 32U);
    LocalTensor<float> floatThreshold(TPosition::VECCALC, kFloatThresholdAddress, 8U);
    LocalTensor<float> ratioUb(TPosition::VECIN, kRatioAddress, 8U);
    LocalTensor<float> ones(TPosition::VECCALC, kOnesAddress, 8U);
    LocalTensor<float> zeros(TPosition::VECCALC, kZerosAddress, 8U);
    LocalTensor<uint8_t> comparison(TPosition::VECCALC, kMaskAddress, kMaskBytes);
    // Keep arithmetic separate from the interleaved inputs: a contiguous
    // destination inside floatBoxes would overwrite later y1/y2 source rows.
    LocalTensor<float> reference(TPosition::VECCALC, kCalcAddress,
                                  kRows * kFloatRowElements);
    LocalTensor<float> output(TPosition::VECOUT, kOutputAddress,
                               kRows * kFloatRowElements);
    LocalTensor<half> halfOutput(TPosition::VECOUT, kHalfOutputAddress,
                                  kRows * kHalfRowElements);

    GlobalTensor<T> boxesGm, anchorsGm;
    boxesGm.SetGlobalBuffer((__gm__ T*)personBoxes, rows * 4U);
    anchorsGm.SetGlobalBuffer((__gm__ T*)anchorPoints, rows * 2U);
    Duplicate(ones, 1.0f, 8U);
    Duplicate(zeros, 0.0f, 8U);

    // Each GM step is 2*T bytes: boxes[1],boxes[3],boxes[5],boxes[7]...
    // The last copied element is boxes[4*B-1], never an extra batch row.
    // Explicit padding is zero; implicit padding repeats the block's scalar.
    const DataCopyExtParams boxesCopy{static_cast<uint16_t>(2U * rows),
        static_cast<uint32_t>(sizeof(T)), static_cast<uint32_t>(sizeof(T)), 0U, 0U};
    const DataCopyExtParams anchorsCopy{static_cast<uint16_t>(rows),
        static_cast<uint32_t>(sizeof(T)), static_cast<uint32_t>(sizeof(T)), 0U, 0U};
    const DataCopyPadExtParams<T> pad{true, 0U, 0U, static_cast<T>(0)};
    DataCopyPad(inputBoxes, boxesGm[1U], boxesCopy, pad);
    DataCopyPad(inputCy, anchorsGm[1U], anchorsCopy, pad);
    safety_belt::CopyRatioForKnownShape<DynamicRatio>(ratioTiling, ratioUb);
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
    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));

    if constexpr (hasThreshold && thresholdIsHalf) {
        Cast(floatThreshold, thresholdBytes.ReinterpretCast<half>(), RoundMode::CAST_NONE, 8U);
    }
    if constexpr (sizeof(T) == sizeof(half)) {
        // Convert the first eight identical half lanes of each 32B UB row.
        // Both interleaved box columns are converted in one 2*B-repeat Cast.
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(8U);
        const UnaryRepeatParams castRows{1U, 1U, 1U, 1U};
        Cast<float, half, false>(floatBoxes, inputBoxes, RoundMode::CAST_NONE,
                                MASK_PLACEHOLDER, static_cast<uint8_t>(2U * rows), castRows);
        Cast<float, half, false>(cy, inputCy, RoundMode::CAST_NONE,
                                MASK_PLACEHOLDER, static_cast<uint8_t>(rows), castRows);
        ResetMask();
    }
    if (rows != paddedRows) {
        const uint32_t tail = (paddedRows - rows) * kFloatRowElements;
        // Only UB tails are initialized. GM extents remain the exact B rows.
        Duplicate(floatBoxes[2U * rows * kFloatRowElements], 0.0f, 2U * tail);
        Duplicate(cy[rows * kFloatRowElements], 0.0f, tail);
    }
    PipeBarrier<PIPE_V>();
    SetCmpMask(zeros);

    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(64U);
    const uint8_t repeats = static_cast<uint8_t>(paddedRows / 8U);
    const BinaryRepeatParams subtractPairs{1U, 2U, 2U, 8U, 16U, 16U};
    const BinaryRepeatParams addY1{1U, 1U, 2U, 8U, 8U, 16U};
    const UnaryRepeatParams unary{1U, 1U, 8U, 8U};
    Sub<float, false>(reference, y2, y1, MASK_PLACEHOLDER, repeats, subtractPairs);
    PipeBarrier<PIPE_V>();
    safety_belt::MultiplyRatioForKnownShape<DynamicRatio>(
        reference, reference, ratioUb, data.waistRatio, repeats);
    PipeBarrier<PIPE_V>();
    Add<float, false>(reference, reference, y1, MASK_PLACEHOLDER, repeats, addY1);
    PipeBarrier<PIPE_V>();
    if constexpr (hasThreshold) {
        const LocalTensor<float> value = thresholdIsHalf
            ? floatThreshold : thresholdBytes.ReinterpretCast<float>();
        const BinaryRepeatParams broadcast{1U, 1U, 0U, 8U, 8U, 0U};
        Add<float, false>(reference, reference, value, MASK_PLACEHOLDER, repeats, broadcast);
    } else {
        Adds<float, false>(reference, reference, data.defaultThreshold, MASK_PLACEHOLDER, repeats, unary);
    }
    PipeBarrier<PIPE_V>();
    ResetMask();
    Compare(comparison, cy, reference, CMPMODE::GT, valueCount);
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
    // MTE3 compacts the first result of every 32B row into contiguous GM.
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
