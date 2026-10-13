#pragma once

#include <cstdint>

struct SafetyBeltViolationGeometryTilingData {
    uint32_t batchSize;
    uint32_t rowsPerCore;
    uint32_t hasThreshold;
    float waistRatio;
    float defaultThreshold;
};

static_assert(sizeof(SafetyBeltViolationGeometryTilingData) == 20,
              "Host and kernel must use the same tiling layout");
