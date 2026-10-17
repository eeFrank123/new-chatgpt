#pragma once

#include <cstddef>
#include "kernel_operator.h"
#include "safety_belt_violation_geometry_tiling.h"

namespace safety_belt {
static_assert(offsetof(SafetyBeltViolationGeometryTilingData, waistRatio) == 8U,
              "Ratio DMA must follow the shared Host/kernel tiling layout");

// A shape-only kernel knows every input extent before the first DMA. Fetch
// only the actual float32 ratio beside the input tensors, without a scalar
// GM load or an MTE2_S/S_V round trip. The caller shares its MTE2_V event.
template <bool DynamicRatio>
__aicore__ inline void CopyRatioForKnownShape(
    GM_ADDR tiling, const AscendC::LocalTensor<float>& ratioUb) {
    if constexpr (DynamicRatio) {
        AscendC::GlobalTensor<float> ratioGm;
        ratioGm.SetGlobalBuffer((__gm__ float*)(tiling +
            offsetof(SafetyBeltViolationGeometryTilingData, waistRatio)), 1U);
        const AscendC::DataCopyExtParams copy{1U, sizeof(float), 0U, 0U, 0U};
        // With no explicit padding, the first scalar fills all eight UB lanes.
        AscendC::DataCopyPad(ratioUb, ratioGm, copy,
            AscendC::DataCopyPadExtParams<float>{true, 0U, 0U, 0.0f});
    }
}

// Both instructions perform one separately rounded float32 multiplication.
// Retain the caller's active mask and barriers; no formula reassociation.
template <bool DynamicRatio>
__aicore__ inline void MultiplyRatioForKnownShape(
    const AscendC::LocalTensor<float>& dst,
    const AscendC::LocalTensor<float>& src,
    const AscendC::LocalTensor<float>& ratioUb,
    float staticRatio, uint8_t repeats) {
    using namespace AscendC;
    if constexpr (DynamicRatio) {
        const BinaryRepeatParams broadcast{1U, 1U, 0U, 8U, 8U, 0U};
        Mul<float, false>(dst, src, ratioUb, MASK_PLACEHOLDER, repeats, broadcast);
    } else {
        const UnaryRepeatParams contiguous{1U, 1U, 8U, 8U};
        Muls<float, false>(dst, src, staticRatio, MASK_PLACEHOLDER, repeats, contiguous);
    }
}
}  // namespace safety_belt
