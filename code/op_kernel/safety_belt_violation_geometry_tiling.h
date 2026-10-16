#pragma once

#include <cstdint>

struct SafetyBeltViolationGeometryTilingData {
    uint32_t batchSize;
    uint32_t rowsPerCore;
    float waistRatio;
    float defaultThreshold;
};

static_assert(sizeof(SafetyBeltViolationGeometryTilingData) == 16,
              "Host and kernel must use the same tiling layout");

namespace safety_belt {
// Preserve the original six shape indices so existing static keys keep their
// meaning. Both the host selector and kernel constants use this one table.
constexpr uint32_t kStaticShapeCount = 11U;

constexpr uint32_t StaticBatchForShape(uint32_t shape) {
    // Opbuild parses this shared header with -std=c++11 independently of the
    // project's CXX_STANDARD. A C++11 constexpr body must be one return.
    return shape == 0U ? 1U : shape == 1U ? 32U : shape == 2U ? 33U :
           shape == 3U ? 256U : shape == 4U ? 1024U : shape == 5U ? 4096U :
           shape == 6U ? 64U : shape == 7U ? 128U : shape == 8U ? 129U :
           shape == 9U ? 257U : shape == 10U ? 512U : 0U;
}

constexpr uint32_t StaticCoreRowsForShape(uint32_t shape) {
    return shape <= 2U ? 64U : shape == 3U ? 256U :
           shape == 4U || shape == 5U ? 512U :
           shape == 6U ? 64U : shape == 7U ? 128U : shape == 8U ? 192U :
           shape == 9U ? 320U : shape == 10U ? 512U : 0U;
}
}  // namespace safety_belt
