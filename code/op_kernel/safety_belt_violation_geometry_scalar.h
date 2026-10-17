#pragma once

#include "kernel_operator.h"
#include "safety_belt_violation_geometry_inline.h"
#include "safety_belt_violation_geometry_tiling.h"
#include "safety_belt_violation_geometry_ratio.h"

// Experimental NPU Scalar backend. The caller selects the small-shape limit
// and owns InitSocState/the final PIPE_ALL barrier, as for the Vector backends.
#ifndef SBVG_SCALAR_SMALL_LIMIT
#define SBVG_SCALAR_SMALL_LIMIT 64
#endif
static_assert(SBVG_SCALAR_SMALL_LIMIT >= 1 && SBVG_SCALAR_SMALL_LIMIT <= 64,
              "SBVG_SCALAR_SMALL_LIMIT must be in [1, 64]");

namespace safety_belt {
// Keep the four float32 operations distinct. In particular, this path must not
// contract the multiplication/addition, reassociate, or replace ref with y2.
// Clang/Bisheng accepts these pragmas inside a compound statement; other CPU
// model compilers retain their normal strict floating-point compiler options.
SBVG_FORCE_INLINE bool ScalarGeometryViolation(
    float y1, float y2, float cy, float ratio, float threshold) {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
    const float height = y2 - y1;
    const float scaled = height * ratio;
    const float reference = y1 + scaled;
    const float bound = reference + threshold;
    return cy > bound;
}
}  // namespace safety_belt

// Exactly one vector core handles this entire batch. No scalar GM access is
// used: input/attribute scalars enter UB through DMA, and output leaves via DMA.
// B<=64 requires at most 3.5KiB UB plus the unchanged SDK temporary reserve.
template <class T, uint32_t IO_MODE, bool DynamicRatio = false, bool UnitRatio = false>
SBVG_FORCE_INLINE void RunSafetyBeltScalarSmall(
    GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
    GM_ADDR violationScore, const SafetyBeltViolationGeometryTilingData& data,
    GM_ADDR tiling = nullptr) {
    using namespace AscendC;
    constexpr uint32_t kRows = 64U;
    constexpr uint32_t kInputAddress = 0U;
    constexpr uint32_t kInputBytes = kRows * 6U * sizeof(float);
    constexpr uint32_t kFloatThresholdAddress = kInputBytes;
    constexpr uint32_t kRatioAddress = kFloatThresholdAddress + 32U;
    constexpr uint32_t kConvertedAddress = kRatioAddress + 32U;
    // Half coordinates occupy at most 384 halfs; a half threshold adds one
    // fully initialized 16-half DMA block to the same single Cast operation.
    constexpr uint32_t kConvertedElements = kRows * 6U + 16U;
    constexpr uint32_t kOutputAddress =
        kConvertedAddress + kConvertedElements * sizeof(float);
    static_assert(kOutputAddress % 32U == 0U,
                  "Scalar output must start on a DMA block boundary");
    static_assert(kOutputAddress + kRows * sizeof(float) + 8192U <= 192U * 1024U,
                  "Scalar backend must leave the SDK UB reserve available");
    constexpr bool halfCoordinates = sizeof(T) == sizeof(half);
    constexpr bool hasThreshold = (IO_MODE & 1U) != 0U;
    constexpr bool thresholdIsHalf = (IO_MODE & 2U) != 0U;
    constexpr bool outputIsHalf = (IO_MODE & 4U) != 0U;
    static_assert(sizeof(T) == sizeof(half) || sizeof(T) == sizeof(float),
                  "Scalar coordinates must be float16 or float32");
    static_assert(IO_MODE < 8U && (!thresholdIsHalf || hasThreshold),
                  "Invalid optional-threshold template mode");
    static_assert(halfCoordinates || (IO_MODE & 6U) == 0U,
                  "Half IO compatibility requires half coordinates");
    static_assert(!UnitRatio || !DynamicRatio,
                  "A runtime ratio cannot use the unit-ratio specialization");

    const uint32_t rows = data.batchSize;
    const uint32_t boxesBytes = rows * 4U * sizeof(T);
    const uint32_t anchorsBytes = rows * 2U * sizeof(T);
    const uint32_t paddedBoxesBytes = (boxesBytes + 31U) & ~31U;
    const uint32_t paddedAnchorsBytes = (anchorsBytes + 31U) & ~31U;
    const uint32_t rawEnd = paddedBoxesBytes + paddedAnchorsBytes;
    LocalTensor<T> raw(TPosition::VECIN, kInputAddress, kInputBytes / sizeof(T));
    LocalTensor<T> boxes = raw;
    LocalTensor<T> anchors = raw[paddedBoxesBytes / sizeof(T)];
    LocalTensor<float> floatThreshold(TPosition::VECIN, kFloatThresholdAddress, 8U);
    LocalTensor<float> ratioUb(TPosition::VECIN, kRatioAddress, 8U);
    LocalTensor<float> converted(TPosition::VECCALC, kConvertedAddress, kConvertedElements);
    LocalTensor<float> floatOutput(TPosition::VECOUT, kOutputAddress, kRows);
    LocalTensor<uint16_t> halfOutputBits(TPosition::VECOUT, kOutputAddress, kRows);

    GlobalTensor<T> boxesGm, anchorsGm;
    boxesGm.SetGlobalBuffer((__gm__ T*)personBoxes, rows * 4U);
    anchorsGm.SetGlobalBuffer((__gm__ T*)anchorPoints, rows * 2U);
    const DataCopyExtParams boxesCopy{1U, boxesBytes, 0U, 0U, 0U};
    const DataCopyExtParams anchorsCopy{1U, anchorsBytes, 0U, 0U, 0U};
    if (boxesBytes == paddedBoxesBytes) {
        DataCopy(boxes, boxesGm, rows * 4U);
    } else {
        // Explicit right padding initializes every byte subsequently Cast.
        // It fills only the final 32B block and never extends the GM read.
        const uint8_t pad = static_cast<uint8_t>((paddedBoxesBytes - boxesBytes) / sizeof(T));
        DataCopyPad(boxes, boxesGm, boxesCopy,
                    DataCopyPadExtParams<T>{true, 0U, pad, static_cast<T>(0)});
    }
    if (anchorsBytes == paddedAnchorsBytes) {
        DataCopy(anchors, anchorsGm, rows * 2U);
    } else {
        const uint8_t pad = static_cast<uint8_t>((paddedAnchorsBytes - anchorsBytes) / sizeof(T));
        DataCopyPad(anchors, anchorsGm, anchorsCopy,
                    DataCopyPadExtParams<T>{true, 0U, pad, static_cast<T>(0)});
    }

    if constexpr (hasThreshold) {
        if constexpr (thresholdIsHalf) {
            // Append the half threshold to the contiguous initialized input
            // region, so the same Cast handles coordinates and threshold.
            GlobalTensor<half> thresholdGm;
            thresholdGm.SetGlobalBuffer((__gm__ half*)threshold, 1U);
            LocalTensor<half> halfThreshold(TPosition::VECIN, rawEnd, 16U);
            const DataCopyExtParams copy{1U, sizeof(half), 0U, 0U, 0U};
            DataCopyPad(halfThreshold, thresholdGm, copy,
                        DataCopyPadExtParams<half>{true, 0U, 15U, static_cast<half>(0)});
        } else {
            GlobalTensor<float> thresholdGm;
            thresholdGm.SetGlobalBuffer((__gm__ float*)threshold, 1U);
            const DataCopyExtParams copy{1U, sizeof(float), 0U, 0U, 0U};
            DataCopyPad(floatThreshold, thresholdGm, copy,
                        DataCopyPadExtParams<float>{true, 0U, 7U, 0.0f});
        }
    }
    safety_belt::CopyRatioForKnownShape<DynamicRatio>(tiling, ratioUb);

    constexpr event_t ready = static_cast<event_t>(0);
    // This event also covers dynamic ratio and float threshold scalar reads.
    SetFlag<HardEvent::MTE2_S>(ready);
    if constexpr (halfCoordinates) {
        SetFlag<HardEvent::MTE2_V>(ready);
        WaitFlag<HardEvent::MTE2_V>(ready);
        const uint32_t castElements = rawEnd / sizeof(half) +
            (hasThreshold && thresholdIsHalf ? 16U : 0U);
        Cast(converted, raw.template ReinterpretCast<half>(), RoundMode::CAST_NONE, castElements);
        SetFlag<HardEvent::V_S>(ready);
    }
    WaitFlag<HardEvent::MTE2_S>(ready);

    float ratio = UnitRatio ? 1.0f : data.waistRatio;
    if constexpr (DynamicRatio) {
        ratio = ratioUb.GetValue(0U);
    }
    float thresholdValue = 0.0f;
    if constexpr (hasThreshold && !thresholdIsHalf) {
        thresholdValue = floatThreshold.GetValue(0U);
    }
    if constexpr (halfCoordinates) {
        WaitFlag<HardEvent::V_S>(ready);
        if constexpr (hasThreshold && thresholdIsHalf) {
            thresholdValue = converted.GetValue(rawEnd / sizeof(half));
        }
    }

    const uint32_t anchorBase = paddedBoxesBytes / sizeof(T);
    for (uint32_t row = 0U; row < rows; ++row) {
        float y1, y2, cy;
        if constexpr (halfCoordinates) {
            y1 = converted.GetValue(row * 4U + 1U);
            y2 = converted.GetValue(row * 4U + 3U);
            cy = converted.GetValue(anchorBase + row * 2U + 1U);
        } else {
            y1 = boxes.GetValue(row * 4U + 1U);
            y2 = boxes.GetValue(row * 4U + 3U);
            cy = anchors.GetValue(row * 2U + 1U);
        }
        const bool violation = safety_belt::ScalarGeometryViolation(
            y1, y2, cy, ratio, thresholdValue);
        if constexpr (outputIsHalf) {
            // IEEE binary16 encodings for exactly 0.0 and 1.0 avoid an output
            // Cast and its additional Vector/Scalar dependency transition.
            halfOutputBits.SetValue(row, violation ? uint16_t{0x3c00U} : uint16_t{0U});
        } else {
            floatOutput.SetValue(row, violation ? 1.0f : 0.0f);
        }
    }
    SetFlag<HardEvent::S_MTE3>(ready);
    WaitFlag<HardEvent::S_MTE3>(ready);
    if constexpr (outputIsHalf) {
        GlobalTensor<uint16_t> outputGm;
        outputGm.SetGlobalBuffer((__gm__ uint16_t*)violationScore, rows);
        if (rows * sizeof(uint16_t) % 32U == 0U) {
            DataCopy(outputGm, halfOutputBits, rows);
        } else {
            const DataCopyExtParams copy{
                1U, rows * static_cast<uint32_t>(sizeof(uint16_t)), 0U, 0U, 0U};
            DataCopyPad(outputGm, halfOutputBits, copy);
        }
    } else {
        GlobalTensor<float> outputGm;
        outputGm.SetGlobalBuffer((__gm__ float*)violationScore, rows);
        if (rows * sizeof(float) % 32U == 0U) {
            DataCopy(outputGm, floatOutput, rows);
        } else {
            const DataCopyExtParams copy{
                1U, rows * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(outputGm, floatOutput, copy);
        }
    }
}
