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
    constexpr uint32_t batches[kStaticShapeCount] = {
        1U, 32U, 33U, 256U, 1024U, 4096U, 64U, 128U, 129U, 257U, 512U};
    return shape < kStaticShapeCount ? batches[shape] : 0U;
}

constexpr uint32_t StaticCoreRowsForShape(uint32_t shape) {
    constexpr uint32_t rows[kStaticShapeCount] = {
        64U, 64U, 64U, 256U, 512U, 512U, 64U, 128U, 192U, 320U, 512U};
    return shape < kStaticShapeCount ? rows[shape] : 0U;
}
}  // namespace safety_belt
