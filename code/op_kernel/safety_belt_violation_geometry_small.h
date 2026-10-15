#pragma once

#include "kernel_operator.h"
#include "safety_belt_violation_geometry_tiling.h"
#include "safety_belt_violation_geometry_gather.h"

// One vector core handles B <= 64. InitSocState and the final PIPE_ALL barrier
// belong to the caller. Addresses below share the physical UB and never overlap.
template <class T>
__aicore__ inline void RunSafetyBeltSmall(
    GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
    GM_ADDR violationScore, const SafetyBeltViolationGeometryTilingData& data) {
    using namespace AscendC;
    constexpr uint32_t kRows = 64U;
    constexpr uint32_t kBoxesAddress = 0U;
    constexpr uint32_t kAnchorsAddress = 1024U;
    constexpr uint32_t kThresholdAddress = 1536U;
    constexpr uint32_t kCoordsAddress = 1568U;
    constexpr uint32_t kHalfCoordsAddress = 2336U;
    constexpr uint32_t kOnesAddress = 2720U;
    constexpr uint32_t kZerosAddress = 2752U;
    constexpr uint32_t kFloatThresholdAddress = 2784U;
    constexpr uint32_t kOutputAddress = 2816U;
    constexpr uint32_t kHalfOutputAddress = 3072U;
    static_assert(kHalfOutputAddress + kRows * sizeof(half) == 3200U,
                  "Small-shape UB layout must remain within 3200 bytes");

    const uint32_t rows = data.batchSize;
    const bool hasThreshold = data.hasThreshold != 0U;
    const bool thresholdIsHalf = sizeof(T) == sizeof(half) && data.thresholdIsHalf != 0U;
    const bool outputIsHalf = sizeof(T) == sizeof(half) && data.outputIsHalf != 0U;

    LocalTensor<T> boxes(TPosition::VECIN, kBoxesAddress, kRows * 4U);
    LocalTensor<T> anchors(TPosition::VECIN, kAnchorsAddress, kRows * 2U);
    LocalTensor<uint8_t> thresholdBytes(TPosition::VECIN, kThresholdAddress, 32U);
    LocalTensor<float> coords(TPosition::VECCALC, kCoordsAddress, kRows * 3U);
    LocalTensor<float> y1 = coords;
    LocalTensor<float> y2 = coords[kRows];
    LocalTensor<float> cy = coords[2U * kRows];
    LocalTensor<half> halfCoords(TPosition::VECCALC, kHalfCoordsAddress, kRows * 3U);
    LocalTensor<float> ones(TPosition::VECCALC, kOnesAddress, 8U);
    LocalTensor<float> zeros(TPosition::VECCALC, kZerosAddress, 8U);
    LocalTensor<float> convertedThreshold(TPosition::VECCALC, kFloatThresholdAddress, 8U);
    LocalTensor<float> output(TPosition::VECOUT, kOutputAddress, kRows);
    LocalTensor<half> halfOutput(TPosition::VECOUT, kHalfOutputAddress, kRows);

    GlobalTensor<T> boxesGm, anchorsGm;
    boxesGm.SetGlobalBuffer((__gm__ T*)personBoxes, rows * 4U);
    anchorsGm.SetGlobalBuffer((__gm__ T*)anchorPoints, rows * 2U);

    // Constant setup and the single half-coordinate initialization can overlap
    // input DMA. Float coordinates use only active rows and need no tail fill.
    Duplicate(ones, 1.0f, 8U);
    Duplicate(zeros, 0.0f, 8U);
    if constexpr (sizeof(T) == sizeof(half)) {
        if (rows != kRows) {
            Duplicate(halfCoords, static_cast<half>(0), 3U * kRows);
        }
    }

    const DataCopyExtParams boxesCopy{
        1U, rows * 4U * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
    const DataCopyExtParams anchorsCopy{
        1U, rows * 2U * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
    if (boxesCopy.blockLen % 32U == 0U) {
        DataCopy(boxes, boxesGm, rows * 4U);
    } else {
        DataCopyPad(boxes, boxesGm, boxesCopy,
                    DataCopyPadExtParams<T>{true, 0U, 0U, static_cast<T>(0)});
    }
    if (anchorsCopy.blockLen % 32U == 0U) {
        DataCopy(anchors, anchorsGm, rows * 2U);
    } else {
        DataCopyPad(anchors, anchorsGm, anchorsCopy,
                    DataCopyPadExtParams<T>{true, 0U, 0U, static_cast<T>(0)});
    }
    if (hasThreshold) {
        // With both explicit paddings zero, the first scalar fills the dummy
        // bytes through the next 32B boundary, providing eight float copies.
        if (thresholdIsHalf) {
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
    // Finish the prezero before Gather overwrites its active coordinates.
    // The vector barrier runs while MTE2 is fetching the independent inputs.
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));

    if (hasThreshold && thresholdIsHalf) {
        Cast(convertedThreshold, thresholdBytes.ReinterpretCast<half>(), RoundMode::CAST_NONE, 8U);
    }

    // A2 counter mode handles mask source elements in each repeat. One repeat
    // therefore gathers precisely the valid rows without reading an UB suffix.
    const GatherMaskParams gatherParams{1U, 1U, 8U, 0U};
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(rows * 4U);
    if constexpr (sizeof(T) == sizeof(half)) {
        safety_belt::GatherFixed(halfCoords, boxes, uint8_t{4U}, true, rows * 4U, gatherParams);
        safety_belt::GatherFixed(halfCoords[kRows], boxes, uint8_t{6U}, true, rows * 4U, gatherParams);
        SetVectorMask<T, MaskMode::COUNTER>(rows * 2U);
        safety_belt::GatherFixed(halfCoords[2U * kRows], anchors, uint8_t{2U}, true, rows * 2U, gatherParams);
    } else {
        safety_belt::GatherFixed(y1, boxes, uint8_t{4U}, true, rows * 4U, gatherParams);
        safety_belt::GatherFixed(y2, boxes, uint8_t{6U}, true, rows * 4U, gatherParams);
        SetVectorMask<T, MaskMode::COUNTER>(rows * 2U);
        safety_belt::GatherFixed(cy, anchors, uint8_t{2U}, true, rows * 2U, gatherParams);
    }
    SetMaskNorm();
    PipeBarrier<PIPE_V>();
    if constexpr (sizeof(T) == sizeof(half)) {
        // Tail gaps were initialized before DMA, permitting one Cast dispatch.
        Cast(coords, halfCoords, RoundMode::CAST_NONE, 3U * kRows);
        PipeBarrier<PIPE_V>();
    }

    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(rows);
    const BinaryRepeatParams binary{1U, 1U, 1U, 8U, 8U, 8U};
    const UnaryRepeatParams unary{1U, 1U, 8U, 8U};
    Sub<float, false>(y2, y2, y1, MASK_PLACEHOLDER, uint8_t{1U}, binary);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(y2, y2, data.waistRatio, MASK_PLACEHOLDER, uint8_t{1U}, unary);
    PipeBarrier<PIPE_V>();
    Add<float, false>(y2, y2, y1, MASK_PLACEHOLDER, uint8_t{1U}, binary);
    PipeBarrier<PIPE_V>();
    if (hasThreshold) {
        const LocalTensor<float> value = thresholdIsHalf ? convertedThreshold
                                                       : thresholdBytes.ReinterpretCast<float>();
        const BinaryRepeatParams broadcast{1U, 1U, 0U, 8U, 8U, 0U};
        Add<float, false>(y2, y2, value, MASK_PLACEHOLDER, uint8_t{1U}, broadcast);
    } else {
        Adds<float, false>(y2, y2, data.defaultThreshold, MASK_PLACEHOLDER, uint8_t{1U}, unary);
    }
    PipeBarrier<PIPE_V>();

    // Register Compare honors an active mask smaller than 64. Memory-output
    // Compare always consumes whole float vectors on A2 and is unsuitable here.
    Compare(cy, y2, CMPMODE::GT, static_cast<uint64_t>(rows), binary);
    PipeBarrier<PIPE_V>();
    const BinaryRepeatParams selectBroadcast{1U, 0U, 0U, 8U, 0U, 0U};
    Select<float, SELMODE::VSEL_CMPMASK_SPR>(
        output, ones, zeros, uint8_t{1U}, selectBroadcast);
    ResetMask();

    if (outputIsHalf) {
        PipeBarrier<PIPE_V>();
        Cast(halfOutput, output, RoundMode::CAST_NONE, rows);
    }
    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
    WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
    if (outputIsHalf) {
        GlobalTensor<half> outputGm;
        outputGm.SetGlobalBuffer((__gm__ half*)violationScore, rows);
        const DataCopyExtParams copy{1U, rows * static_cast<uint32_t>(sizeof(half)), 0U, 0U, 0U};
        DataCopyPad(outputGm, halfOutput, copy);
    } else {
        GlobalTensor<float> outputGm;
        outputGm.SetGlobalBuffer((__gm__ float*)violationScore, rows);
        const DataCopyExtParams copy{1U, rows * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(outputGm, output, copy);
    }
}
