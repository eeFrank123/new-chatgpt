#include <algorithm>
#include <iostream>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include "../code/op_kernel/safety_belt_violation_geometry.cpp"
#include "../code/op_host/tiling_policy.h"

struct Header {
    uint32_t batch, bits, cores, hasThreshold, thresholdBits, outputBits;
    float ratio, threshold;
};
static_assert(sizeof(Header) == 32);

template<class T> void DispatchKernel(uint32_t mode, GM_ADDR boxes, GM_ADDR anchors,
                                     GM_ADDR threshold, GM_ADDR output, GM_ADDR tiling) {
    // Model the host-selected independent kernel binaries, not a device switch.
    switch (mode) {
        case 0: safety_belt_violation_geometry<T, 0>(boxes, anchors, threshold, output, nullptr, tiling); break;
        case 1: safety_belt_violation_geometry<T, 1>(boxes, anchors, threshold, output, nullptr, tiling); break;
        default:
            if constexpr (sizeof(T) == sizeof(half)) {
                switch (mode) {
                    case 3: safety_belt_violation_geometry<T, 3>(boxes, anchors, threshold, output, nullptr, tiling); break;
                    case 4: safety_belt_violation_geometry<T, 4>(boxes, anchors, threshold, output, nullptr, tiling); break;
                    case 5: safety_belt_violation_geometry<T, 5>(boxes, anchors, threshold, output, nullptr, tiling); break;
                    case 7: safety_belt_violation_geometry<T, 7>(boxes, anchors, threshold, output, nullptr, tiling); break;
                    default: assert(false && "unsupported IO mode");
                }
            } else { assert(false && "unsupported float32 IO mode"); }
    }
}

template<class T, class OutputT> void Run(const Header& h) {
    std::vector<T> boxes(h.batch * 4), anchors(h.batch * 2);
    std::cin.read(reinterpret_cast<char*>(boxes.data()), boxes.size() * sizeof(T));
    std::cin.read(reinterpret_cast<char*>(anchors.data()), anchors.size() * sizeof(T));
    assert(std::cin.good());
    const OutputT sentinel = static_cast<OutputT>(-12344.0f);
    std::vector<OutputT> guarded(h.batch + 16, sentinel);
    OutputT* output = guarded.data() + 8;
    const auto p = safety_belt::MakePartition(h.batch, h.cores);
    SafetyBeltViolationGeometryTilingData data{h.batch, p.rowsPerCore, h.ratio, 0.0f};
    // Exact-sized allocations let ASan detect a 4-byte read of a 2-byte threshold.
    std::vector<float> thresholdFloat(1, h.threshold);
    std::vector<half> thresholdHalf(1, static_cast<half>(h.threshold));
    GM_ADDR threshold = h.thresholdBits == 16 ? reinterpret_cast<GM_ADDR>(thresholdHalf.data())
                                             : reinterpret_cast<GM_ADDR>(thresholdFloat.data());
    AscendC::globalAllocations.clear();
    AscendC::RegisterGlobal(boxes.data(), boxes.size() * sizeof(T));
    AscendC::RegisterGlobal(anchors.data(), anchors.size() * sizeof(T));
    AscendC::RegisterGlobal(output, h.batch * sizeof(OutputT));
    if (h.hasThreshold) AscendC::RegisterGlobal(threshold, h.thresholdBits / 8);
    AscendC::thresholdAddress = h.hasThreshold ? threshold : nullptr;
    AscendC::outputAddress = output;
    AscendC::outputWrites.assign(h.batch, 0);
    AscendC::thresholdReads = 0;
    AscendC::thresholdCopies = 0;
    for (uint32_t block = 0; block < p.blockDim; ++block) {
        AscendC::ResetPipelines();
        AscendC::blockIndex = block;
        const uint32_t mode = h.hasThreshold |
            (static_cast<uint32_t>(h.hasThreshold && h.thresholdBits == 16) << 1U) |
            (static_cast<uint32_t>(h.outputBits == 16) << 2U);
        DispatchKernel<T>(mode,
            reinterpret_cast<GM_ADDR>(boxes.data()), reinterpret_cast<GM_ADDR>(anchors.data()),
            h.hasThreshold ? threshold : nullptr,
            reinterpret_cast<GM_ADDR>(output), reinterpret_cast<GM_ADDR>(&data));
        for (bool pending : AscendC::eventFlags) assert(!pending);
        assert(!AscendC::unifiedMaskEnabled && "unified vector mask must be restored at kernel exit");
        assert(!AscendC::vectorCounterMode && "normal mask mode must be restored at kernel exit");
    }
    assert(AscendC::thresholdReads + AscendC::thresholdCopies == (h.hasThreshold ? p.blockDim : 0));
    for (uint32_t i = 0; i < 8; ++i) {
        assert(guarded[i] == sentinel && guarded[h.batch + 8 + i] == sentinel);
    }
    for (uint32_t i = 0; i < h.batch; ++i) {
        assert(output[i] == 0.0f || output[i] == 1.0f);
        assert(AscendC::outputWrites[i] == 1 && "each output row must be written exactly once");
    }
    std::cout.write(reinterpret_cast<const char*>(output), h.batch * sizeof(OutputT));
}

int main() {
#ifdef _WIN32
    // C/C++ streams default to text mode on Windows, which corrupts binary floats.
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    // Exhaustively verify every legal B under 1..64 available vector cores.
    for (uint32_t batch = 1; batch <= 4096; ++batch) {
        for (uint32_t cores = 1; cores <= 64; ++cores) {
            const auto p = safety_belt::MakePartition(batch, cores);
            assert(p.blockDim >= 1 && p.blockDim <= cores);
            assert(p.rowsPerCore % safety_belt::kRowGranularity == 0);
            assert(p.rowsPerCore >= 1 && p.rowsPerCore <= 4096);
            assert((p.blockDim - 1) * p.rowsPerCore < batch);
            assert(p.blockDim * p.rowsPerCore >= batch);
        }
    }
    Header h{};
    while (std::cin.read(reinterpret_cast<char*>(&h), sizeof(h))) {
        assert(h.batch >= 1 && h.batch <= 4096 && h.cores >= 1);
        assert(h.hasThreshold <= 1 && h.ratio > 0 && h.ratio <= 1);
        assert(h.thresholdBits == 16 || h.thresholdBits == 32);
        assert(h.outputBits == 16 || h.outputBits == 32);
        if (h.bits == 16) {
            if (h.outputBits == 16) Run<half, half>(h);
            else Run<half, float>(h);
        } else {
            assert(h.bits == 32 && h.outputBits == 32 && h.thresholdBits == 32);
            Run<float, float>(h);
        }
    }
    return 0;
}
