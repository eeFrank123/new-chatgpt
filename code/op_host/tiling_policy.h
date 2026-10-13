#pragma once

#include <cstdint>

namespace safety_belt {
constexpr uint32_t kVectorRows = 64;
constexpr uint32_t kTileRows = 256;

inline uint32_t CeilDiv(uint32_t x, uint32_t y) {
    return (x + y - 1U) / y;
}

struct Partition {
    uint32_t rowsPerCore;
    uint32_t blockDim;
};

// Each core starts at a 64-row boundary. Only the final core can have a tail.
// Avoid launching all available cores for tiny inputs.
inline Partition MakePartition(uint32_t batch, uint32_t availableCores) {
    const uint32_t wanted = CeilDiv(batch, 128U);
    const uint32_t cores = availableCores < wanted ? availableCores : wanted;
    const uint32_t rows = CeilDiv(CeilDiv(batch, cores), kVectorRows) * kVectorRows;
    return {rows, CeilDiv(batch, rows)};
}
}  // namespace safety_belt
