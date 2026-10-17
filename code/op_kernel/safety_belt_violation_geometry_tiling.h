#pragma once

#include <cstdint>

#ifndef SBVG_TARGET_ROWS_PER_CORE
#define SBVG_TARGET_ROWS_PER_CORE 512
#endif

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
    // Keep the large aligned shapes specialized in every supported comparison
    // profile. The Host and kernel builds must receive the same target value.
    // Other profiles retain the original table and fall back when it differs
    // from the actual partition. Keep one return for opbuild's C++11 parser.
    return shape <= 2U ? 64U : shape == 3U ? 256U :
           shape == 4U ?
               (SBVG_TARGET_ROWS_PER_CORE == 1024 ||
                SBVG_TARGET_ROWS_PER_CORE == 2048 ||
                SBVG_TARGET_ROWS_PER_CORE == 4096 ? 1024U : 512U) :
           shape == 5U ?
               (SBVG_TARGET_ROWS_PER_CORE == 1024 ? 1024U :
                SBVG_TARGET_ROWS_PER_CORE == 2048 ? 2048U :
                SBVG_TARGET_ROWS_PER_CORE == 4096 ? 4096U : 512U) :
           shape == 6U ? 64U : shape == 7U ? 128U : shape == 8U ? 192U :
           shape == 9U ? 320U : shape == 10U ? 512U : 0U;
}
}  // namespace safety_belt
