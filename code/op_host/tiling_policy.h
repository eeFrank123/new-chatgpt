#pragma once

#include <cstdint>

namespace safety_belt {
// 单核行数上限：等于 Kernel 侧一个 tile 的容量（op_kernel 里的 kTileRows）。
constexpr uint32_t kMaxRowsPerCore = 256;
// 单核行数的分配粒度。任何一个核的起始行号都是它的整数倍，因此：
//   f32  每行 4 个 float = 16B -> 8 行 = 128B，32B 对齐；
//   f16  每行 4 个 half  =  8B -> 8 行 =  64B，32B 对齐。
// 这样 DataCopyPad 的 GM 地址始终满足对齐要求，同时可以把 batch 比较均匀地铺到所有核上。
constexpr uint32_t kRowGranularity = 8;

inline uint32_t CeilDiv(uint32_t x, uint32_t y) {
    return (x + y - 1U) / y;
}

struct Partition {
    uint32_t rowsPerCore;
    uint32_t blockDim;
};

// 尽量用满可用核数：任务总量很小时按 8 行粒度退化（B=32 只启 1 个核），
// 大 batch 时每个核只分到几十行，单核搬运量小、GM 往返延迟一次就够。
inline Partition MakePartition(uint32_t batch, uint32_t availableCores) {
    uint32_t cores = CeilDiv(batch, kRowGranularity);
    if (cores > availableCores) {
        cores = availableCores;
    }
    if (cores == 0U) {
        cores = 1U;
    }
    uint32_t rows = CeilDiv(CeilDiv(batch, cores), kRowGranularity) * kRowGranularity;
    if (rows > kMaxRowsPerCore) {
        rows = kMaxRowsPerCore;
    }
    if (rows == 0U) {
        rows = kMaxRowsPerCore;
    }
    return {rows, CeilDiv(batch, rows)};
}
}  // namespace safety_belt
