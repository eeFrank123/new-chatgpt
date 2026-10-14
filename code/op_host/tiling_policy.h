#pragma once

#include <cstdint>

namespace safety_belt {
// 单核处理的行数上限：等于 Kernel 侧一个 tile 的容量（op_kernel 里的 kMaxRows）。
constexpr uint32_t kMaxRowsPerCore = 256;
// 分核粒度。任何一个核的起始行号都是它的整数倍，于是：
//   f32 每行 4 float = 16B -> 8 行 = 128B；f16 每行 4 half = 8B -> 8 行 = 64B，均满足 32B 对齐。
constexpr uint32_t kRowGranularity = 8;
// 单核目标行数：小 shape 不要为了几行数据去启动一堆核。
// 依据 CANN 最佳实践「头尾开销优化」：核启动开销随启动核数近似线性增长
// （空 kernel 约 2 核 0.67us -> 40 核 2.25us），所以计算量小的任务要"够用就好"。
constexpr uint32_t kTargetRowsPerCore = 128;

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
    if (rows > kMaxRowsPerCore) {
        rows = kMaxRowsPerCore;
    }
    if (rows == 0U) {
        rows = kMaxRowsPerCore;
    }
    return {rows, CeilDiv(batch, rows)};
}
}  // namespace safety_belt
