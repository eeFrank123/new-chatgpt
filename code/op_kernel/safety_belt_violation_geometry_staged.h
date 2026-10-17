#pragma once

#include "safety_belt_violation_geometry_small.h"

// The float path can start the person-box calculation while anchors and the
// optional threshold are still being fetched. Half keeps the established
// single Cast of all three packed coordinate columns: splitting that Cast
// would add a second conversion dispatch and two dependent vector barriers.
template <class T, uint32_t IO_MODE, bool DynamicRatio = false>
__aicore__ inline void RunSafetyBeltStagedSmall(
    GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
    GM_ADDR violationScore, const SafetyBeltViolationGeometryTilingData& data,
    GM_ADDR tiling = nullptr) {
    if constexpr (sizeof(T) == sizeof(half)) {
        RunSafetyBeltSmall<T, IO_MODE, DynamicRatio>(
            personBoxes, anchorPoints, threshold, violationScore, data, tiling);
    } else {
        using namespace AscendC;
        constexpr uint32_t kRows = 64U;
        constexpr bool hasThreshold = (IO_MODE & 1U) != 0U;
        static_assert(sizeof(T) == sizeof(float) && (IO_MODE & 6U) == 0U,
                      "Staged float path requires float coordinates and output");

        const uint32_t rows = data.batchSize;
        LocalTensor<float> boxes(TPosition::VECIN, 0U, kRows * 4U);
        LocalTensor<float> anchors(TPosition::VECIN, 1024U, kRows * 2U);
        LocalTensor<float> thresholdUb(TPosition::VECIN, 1536U, 8U);
        LocalTensor<float> coords(TPosition::VECCALC, 1568U, kRows * 3U);
        LocalTensor<float> y1 = coords;
        LocalTensor<float> y2 = coords[kRows];
        LocalTensor<float> cy = coords[2U * kRows];
        LocalTensor<float> ones(TPosition::VECCALC, 2720U, 8U);
        LocalTensor<float> zeros(TPosition::VECCALC, 2752U, 8U);
        LocalTensor<float> output(TPosition::VECOUT, 2816U, kRows);
        LocalTensor<float> ratioUb(TPosition::VECIN, 3200U, 8U);

        GlobalTensor<float> boxesGm, anchorsGm;
        boxesGm.SetGlobalBuffer((__gm__ float*)personBoxes, rows * 4U);
        anchorsGm.SetGlobalBuffer((__gm__ float*)anchorPoints, rows * 2U);
        Duplicate(ones, 1.0f, 8U);
        Duplicate(zeros, 0.0f, 8U);

        const DataCopyExtParams boxesCopy{
            1U, rows * 4U * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        if (boxesCopy.blockLen % 32U == 0U) {
            DataCopy(boxes, boxesGm, rows * 4U);
        } else {
            DataCopyPad(boxes, boxesGm, boxesCopy,
                        DataCopyPadExtParams<float>{true, 0U, 0U, 0.0f});
        }
        // The ratio is needed with the first two box columns. Including its
        // DMA in event 0 lets every supported ratio start the same early math.
        safety_belt::CopyRatioForKnownShape<DynamicRatio>(tiling, ratioUb);
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));

        const DataCopyExtParams anchorsCopy{
            1U, rows * 2U * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        if (anchorsCopy.blockLen % 32U == 0U) {
            DataCopy(anchors, anchorsGm, rows * 2U);
        } else {
            DataCopyPad(anchors, anchorsGm, anchorsCopy,
                        DataCopyPadExtParams<float>{true, 0U, 0U, 0.0f});
        }
        if constexpr (hasThreshold) {
            GlobalTensor<float> thresholdGm;
            thresholdGm.SetGlobalBuffer((__gm__ float*)threshold, 1U);
            const DataCopyExtParams copy{1U, sizeof(float), 0U, 0U, 0U};
            DataCopyPad(thresholdUb, thresholdGm, copy,
                        DataCopyPadExtParams<float>{true, 0U, 0U, 0.0f});
        }
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(1));

        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
        const bool wholeGatherTile = rows == 32U || rows == kRows;
        if (wholeGatherTile) {
            const GatherMaskParams params{
                1U, static_cast<uint16_t>(rows * 4U * sizeof(float) / 256U), 8U, 0U};
            SetMaskNorm();
            SetVectorMask<float, MaskMode::NORMAL>(64U);
            safety_belt::GatherFixed(y1, boxes, uint8_t{4U}, false, 0U, params);
            safety_belt::GatherFixed(y2, boxes, uint8_t{6U}, false, 0U, params);
        } else {
            const GatherMaskParams params{1U, 1U, 8U, 0U};
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(rows * 4U);
            safety_belt::GatherFixed(y1, boxes, uint8_t{4U}, true, rows * 4U, params);
            safety_belt::GatherFixed(y2, boxes, uint8_t{6U}, true, rows * 4U, params);
        }
        SetMaskNorm();
        PipeBarrier<PIPE_V>();

        // Anchors and threshold have no dependency on these three operations.
        // Preserve every separately rounded step, including ratio == 1.0f.
        SetVectorMask<float, MaskMode::NORMAL>(rows);
        const BinaryRepeatParams binary{1U, 1U, 1U, 8U, 8U, 8U};
        const UnaryRepeatParams unary{1U, 1U, 8U, 8U};
        Sub<float, false>(y2, y2, y1, MASK_PLACEHOLDER, uint8_t{1U}, binary);
        PipeBarrier<PIPE_V>();
        safety_belt::MultiplyRatioForKnownShape<DynamicRatio>(
            y2, y2, ratioUb, data.waistRatio, uint8_t{1U});
        PipeBarrier<PIPE_V>();
        Add<float, false>(y2, y2, y1, MASK_PLACEHOLDER, uint8_t{1U}, binary);
        PipeBarrier<PIPE_V>();

        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(1));
        if (wholeGatherTile) {
            const GatherMaskParams params{
                1U, static_cast<uint16_t>(rows * 2U * sizeof(float) / 256U), 8U, 0U};
            SetMaskNorm();
            SetVectorMask<float, MaskMode::NORMAL>(64U);
            safety_belt::GatherFixed(cy, anchors, uint8_t{2U}, false, 0U, params);
        } else {
            const GatherMaskParams params{1U, 1U, 8U, 0U};
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(rows * 2U);
            safety_belt::GatherFixed(cy, anchors, uint8_t{2U}, true, rows * 2U, params);
        }
        // The cy Gather and threshold addition access disjoint vector buffers.
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(rows);
        if constexpr (hasThreshold) {
            const BinaryRepeatParams broadcast{1U, 1U, 0U, 8U, 8U, 0U};
            Add<float, false>(y2, y2, thresholdUb, MASK_PLACEHOLDER, uint8_t{1U}, broadcast);
        } else {
            Adds<float, false>(y2, y2, data.defaultThreshold, MASK_PLACEHOLDER,
                               uint8_t{1U}, unary);
        }
        PipeBarrier<PIPE_V>();
        Compare<float, false>(cy, y2, CMPMODE::GT, MASK_PLACEHOLDER, binary);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams selectBroadcast{1U, 0U, 0U, 8U, 0U, 0U};
        Select<float, SELMODE::VSEL_CMPMASK_SPR>(
            output, ones, zeros, uint8_t{1U}, selectBroadcast);
        ResetMask();

        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
        GlobalTensor<float> outputGm;
        outputGm.SetGlobalBuffer((__gm__ float*)violationScore, rows);
        if (rows * sizeof(float) % 32U == 0U) {
            DataCopy(outputGm, output, rows);
        } else {
            const DataCopyExtParams copy{
                1U, rows * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(outputGm, output, copy);
        }
    }
}
