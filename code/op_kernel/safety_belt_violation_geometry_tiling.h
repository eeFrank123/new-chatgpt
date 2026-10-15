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
