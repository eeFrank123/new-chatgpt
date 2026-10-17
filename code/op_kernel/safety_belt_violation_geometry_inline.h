#pragma once

// Match the force-inline spelling used by the CANN 9 kernel/tiling wrappers.
// Set to 0 for a comparison with the original ordinary C++ inline functions.
#ifndef SBVG_FORCE_INLINE_ENABLED
#define SBVG_FORCE_INLINE_ENABLED 1
#endif
static_assert(SBVG_FORCE_INLINE_ENABLED == 0 || SBVG_FORCE_INLINE_ENABLED == 1,
              "SBVG_FORCE_INLINE_ENABLED must be 0 or 1");

#if SBVG_FORCE_INLINE_ENABLED
#define SBVG_FORCE_INLINE __aicore__ inline __attribute__((always_inline))
#else
#define SBVG_FORCE_INLINE __aicore__ inline
#endif
