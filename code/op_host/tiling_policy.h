#pragma once

#include <cstdint>

#ifndef SBVG_TARGET_ROWS_PER_CORE
#define SBVG_TARGET_ROWS_PER_CORE 512
#endif

namespace safety_belt {
// Kernel 按实际分配行数申请 UB；最大合法 Batch 可由单核处理，无需多 tile 循环。
constexpr uint32_t kMaxRowsPerCore = 4096;
// 对齐 GatherMask / Compare 的处理粒度，也保证所有核的 GM 起点对齐。
constexpr uint32_t kRowGranularity = 64;
// 待 NPU 实测调优的目标值；可用 CMake -DSBVG_TARGET_ROWS_PER_CORE=128 等覆盖。
// 默认 512 是减少小任务核启动开销的起点，不代表已经测得的最优配置。
constexpr uint32_t kTargetRowsPerCore = SBVG_TARGET_ROWS_PER_CORE;
static_assert(kTargetRowsPerCore >= kRowGranularity &&
              kTargetRowsPerCore <= kMaxRowsPerCore &&
              kTargetRowsPerCore % kRowGranularity == 0U,
              "SBVG_TARGET_ROWS_PER_CORE must be a multiple of 64 in [64, 4096]");

inline uint32_t CeilDiv(uint32_t x, uint32_t y) {
    return (x + y - 1U) / y;
}

struct Partition {
    uint32_t rowsPerCore;
    uint32_t blockDim;
};

inline Partition MakePartition(uint32_t batch, uint32_t availableCores) {
    uint32_t cores = CeilDiv(batch, kTargetRowsPerCore);
    if (cores > availableCores) {
        cores = availableCores;
    }
    if (cores == 0U) {
        cores = 1U;
    }
    uint32_t rows = CeilDiv(CeilDiv(batch, cores), kRowGranularity) * kRowGranularity;
    // 对合法 Batch [1,4096]，rows <= 4096。不要截断 rows 后再增加 blockDim，
    // 否则可用核数较少时会突破 availableCores。
    if (rows == 0U) {
        rows = kRowGranularity;
    }
    return {rows, CeilDiv(batch, rows)};
}
}  // namespace safety_belt
