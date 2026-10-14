#pragma once

#include <cstdint>

struct SafetyBeltViolationGeometryTilingData {
    uint32_t batchSize;
    uint32_t rowsPerCore;
    uint32_t hasThreshold;
    float waistRatio;
    float defaultThreshold;
    uint32_t thresholdIsHalf;
    uint32_t outputIsHalf;
};

static_assert(sizeof(SafetyBeltViolationGeometryTilingData) == 28,
              "Host and kernel must use the same tiling layout");
