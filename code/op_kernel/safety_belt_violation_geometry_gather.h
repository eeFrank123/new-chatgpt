#pragma once

#include "kernel_operator.h"

// CANN 8.5 includes the lightweight four-argument overload, although it is
// deprecated. Detect its availability instead of depending on a version macro.
// The caller supplies the vector mask. No retained-element count is needed.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace safety_belt {
template <class T>
__aicore__ inline auto GatherFixedImpl(
    const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src,
    uint8_t pattern, bool /*counter*/, uint32_t /*mask*/,
    const AscendC::GatherMaskParams& params, int)
    -> decltype(AscendC::GatherMask(dst, src, pattern, params), void()) {
    AscendC::GatherMask(dst, src, pattern, params);
}

template <class T>
__aicore__ inline void GatherFixedImpl(
    const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src,
    uint8_t pattern, bool counter, uint32_t mask,
    const AscendC::GatherMaskParams& params, long) {
    // Older or future SDKs may expose only the standard counted interface.
    // It sets its own mask mode, so pass the mode/count on every invocation.
    uint64_t ignoredCount = 0U;
    AscendC::GatherMask(dst, src, pattern, counter, mask, params, ignoredCount);
}

template <class T>
__aicore__ inline void GatherFixed(
    const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src,
    uint8_t pattern, bool counter, uint32_t mask,
    const AscendC::GatherMaskParams& params) {
    GatherFixedImpl(dst, src, pattern, counter, mask, params, 0);
}
}  // namespace safety_belt

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
