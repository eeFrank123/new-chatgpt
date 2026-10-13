#include <algorithm>
#include <iostream>
#include <vector>
#include "../code/op_kernel/safety_belt_violation_geometry.cpp"
#include "../code/op_host/tiling_policy.h"

struct Header {
    uint32_t batch, bits, cores, hasThreshold;
    float ratio, threshold;
};
static_assert(sizeof(Header) == 24);

template<class T> void Run(const Header& h) {
    std::vector<T> boxes(h.batch * 4), anchors(h.batch * 2);
    std::cin.read(reinterpret_cast<char*>(boxes.data()), boxes.size() * sizeof(T));
    std::cin.read(reinterpret_cast<char*>(anchors.data()), anchors.size() * sizeof(T));
    assert(std::cin.good());
    constexpr float sentinel = -12345.0f;
    std::vector<float> guarded(h.batch + 16, sentinel);
    float* output = guarded.data() + 8;
    const auto p = safety_belt::MakePartition(h.batch, h.cores);
    SafetyBeltViolationGeometryTilingData data{h.batch, p.rowsPerCore, h.hasThreshold, h.ratio, 0.0f};
    float threshold = h.threshold;
    AscendC::thresholdReads = 0;
    for (uint32_t block = 0; block < p.blockDim; ++block) {
        AscendC::blockIndex = block;
        safety_belt_violation_geometry<T>(
            reinterpret_cast<GM_ADDR>(boxes.data()), reinterpret_cast<GM_ADDR>(anchors.data()),
            h.hasThreshold ? reinterpret_cast<GM_ADDR>(&threshold) : nullptr,
            reinterpret_cast<GM_ADDR>(output), nullptr, reinterpret_cast<GM_ADDR>(&data));
    }
    assert(AscendC::thresholdReads == (h.hasThreshold ? p.blockDim : 0));
    for (uint32_t i = 0; i < 8; ++i) {
        assert(guarded[i] == sentinel && guarded[h.batch + 8 + i] == sentinel);
    }
    for (uint32_t i = 0; i < h.batch; ++i) assert(output[i] == 0.0f || output[i] == 1.0f);
    std::cout.write(reinterpret_cast<const char*>(output), h.batch * sizeof(float));
}

int main() {
    // Exhaustively verify every legal B under 1..64 available vector cores.
    for (uint32_t batch = 1; batch <= 4096; ++batch) {
        for (uint32_t cores = 1; cores <= 64; ++cores) {
            const auto p = safety_belt::MakePartition(batch, cores);
            assert(p.blockDim >= 1 && p.blockDim <= cores);
            assert(p.rowsPerCore % 64 == 0);
            assert((p.blockDim - 1) * p.rowsPerCore < batch);
            assert(p.blockDim * p.rowsPerCore >= batch);
        }
    }
    Header h{};
    while (std::cin.read(reinterpret_cast<char*>(&h), sizeof(h))) {
        assert(h.batch >= 1 && h.batch <= 4096 && h.cores >= 1);
        assert(h.hasThreshold <= 1 && h.ratio > 0 && h.ratio <= 1);
        if (h.bits == 16) Run<half>(h);
        else { assert(h.bits == 32); Run<float>(h); }
    }
    return 0;
}
