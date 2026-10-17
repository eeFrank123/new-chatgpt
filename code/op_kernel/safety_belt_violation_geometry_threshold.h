#pragma once

#include "kernel_operator.h"
#include "safety_belt_violation_geometry_inline.h"

#ifndef SBVG_SCALAR_THRESHOLD
#define SBVG_SCALAR_THRESHOLD 0
#endif

namespace safety_belt {

// Optional device-side scalar load. Refresh on every invocation so reusing a
// threshold GM address with a new value cannot reuse a stale per-core cache line.
// Call only for a present float32 threshold, after independent input DMA issues.
SBVG_FORCE_INLINE float ReadFloatThresholdAndSynchronize(GM_ADDR threshold) {
    using namespace AscendC;
    GlobalTensor<float> thresholdGm;
    thresholdGm.SetGlobalBuffer((__gm__ float*)threshold, 1U);
    DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE,
                            DcciDst::CACHELINE_OUT>(thresholdGm);
    const float value = thresholdGm.GetValue(0U);
    constexpr event_t ready = static_cast<event_t>(2);
    SetFlag<HardEvent::S_V>(ready);
    WaitFlag<HardEvent::S_V>(ready);
    return value;
}

}  // namespace safety_belt
