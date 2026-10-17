#include <algorithm>
#include <iostream>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include "../code/op_kernel/safety_belt_violation_geometry.cpp"
#include "../code/op_host/tiling_policy.h"

namespace {
// Literal expectations are deliberately independent of the shared metadata.
#if SBVG_TARGET_ROWS_PER_CORE == 1024
constexpr uint32_t kExpectedRowsFor1024 = 1024U;
constexpr uint32_t kExpectedRowsFor4096 = 1024U;
#elif SBVG_TARGET_ROWS_PER_CORE == 2048
constexpr uint32_t kExpectedRowsFor1024 = 1024U;
constexpr uint32_t kExpectedRowsFor4096 = 2048U;
#elif SBVG_TARGET_ROWS_PER_CORE == 4096
constexpr uint32_t kExpectedRowsFor1024 = 1024U;
constexpr uint32_t kExpectedRowsFor4096 = 4096U;
#else
constexpr uint32_t kExpectedRowsFor1024 = 512U;
constexpr uint32_t kExpectedRowsFor4096 = 512U;
#endif
}  // namespace

struct Header {
    uint32_t batch, bits, cores, hasThreshold, thresholdBits, outputBits;
    float ratio, threshold;
};
static_assert(sizeof(Header) == 32);

template<class T, uint32_t IO_MODE>
void DispatchStaticKernel(uint32_t staticCase, GM_ADDR boxes, GM_ADDR anchors,
                          GM_ADDR threshold, GM_ADDR output, GM_ADDR tiling) {
#define SBVG_DISPATCH_CASE(N) \
    case N: safety_belt_violation_geometry<T, IO_MODE, N>(boxes, anchors, threshold, output, nullptr, tiling); break
    switch (staticCase) {
        SBVG_DISPATCH_CASE(0);
        SBVG_DISPATCH_CASE(1);
        SBVG_DISPATCH_CASE(2);
        SBVG_DISPATCH_CASE(3);
        SBVG_DISPATCH_CASE(4);
        SBVG_DISPATCH_CASE(5);
        SBVG_DISPATCH_CASE(6);
        SBVG_DISPATCH_CASE(7);
        SBVG_DISPATCH_CASE(8);
        SBVG_DISPATCH_CASE(9);
        SBVG_DISPATCH_CASE(10);
        SBVG_DISPATCH_CASE(11);
        SBVG_DISPATCH_CASE(12);
        SBVG_DISPATCH_CASE(13);
        SBVG_DISPATCH_CASE(14);
        SBVG_DISPATCH_CASE(15);
        SBVG_DISPATCH_CASE(16);
        SBVG_DISPATCH_CASE(17);
        SBVG_DISPATCH_CASE(18);
        SBVG_DISPATCH_CASE(19);
        SBVG_DISPATCH_CASE(20);
        SBVG_DISPATCH_CASE(21);
        SBVG_DISPATCH_CASE(22);
        SBVG_DISPATCH_CASE(23);
        SBVG_DISPATCH_CASE(24);
        SBVG_DISPATCH_CASE(25);
        SBVG_DISPATCH_CASE(26);
        SBVG_DISPATCH_CASE(27);
        SBVG_DISPATCH_CASE(28);
        SBVG_DISPATCH_CASE(29);
        SBVG_DISPATCH_CASE(30);
        SBVG_DISPATCH_CASE(31);
        SBVG_DISPATCH_CASE(32);
        SBVG_DISPATCH_CASE(33);
        SBVG_DISPATCH_CASE(34);
        SBVG_DISPATCH_CASE(35);
        SBVG_DISPATCH_CASE(36);
        SBVG_DISPATCH_CASE(37);
        SBVG_DISPATCH_CASE(38);
        SBVG_DISPATCH_CASE(39);
        SBVG_DISPATCH_CASE(40);
        SBVG_DISPATCH_CASE(41);
        SBVG_DISPATCH_CASE(42);
        SBVG_DISPATCH_CASE(43);
        SBVG_DISPATCH_CASE(44);
        default: assert(false && "unknown static case");
    }
#undef SBVG_DISPATCH_CASE
}

template<class T> void DispatchKernel(uint32_t mode, uint32_t staticCase, GM_ADDR boxes, GM_ADDR anchors,
                                     GM_ADDR threshold, GM_ADDR output, GM_ADDR tiling) {
    // Model the host-selected independent kernel binaries, not a device switch.
    switch (mode) {
        case 0: DispatchStaticKernel<T, 0>(staticCase, boxes, anchors, threshold, output, tiling); break;
        case 1: DispatchStaticKernel<T, 1>(staticCase, boxes, anchors, threshold, output, tiling); break;
        default:
            if constexpr (sizeof(T) == sizeof(half)) {
                switch (mode) {
                    case 3: DispatchStaticKernel<T, 3>(staticCase, boxes, anchors, threshold, output, tiling); break;
                    case 4: DispatchStaticKernel<T, 4>(staticCase, boxes, anchors, threshold, output, tiling); break;
                    case 5: DispatchStaticKernel<T, 5>(staticCase, boxes, anchors, threshold, output, tiling); break;
                    case 7: DispatchStaticKernel<T, 7>(staticCase, boxes, anchors, threshold, output, tiling); break;
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
    AscendC::RegisterGlobal(&data, sizeof(data));
    if (h.hasThreshold) AscendC::RegisterGlobal(threshold, h.thresholdBits / 8);
    AscendC::thresholdAddress = h.hasThreshold ? threshold : nullptr;
    AscendC::outputAddress = output;
    AscendC::outputWrites.assign(h.batch, 0);
    AscendC::thresholdReads = 0;
    AscendC::thresholdCopies = 0;
    AscendC::scalarCacheInvalidations = 0;
    AscendC::ratioAddress = reinterpret_cast<uint8_t*>(&data.waistRatio);
    AscendC::ratioCopies = 0;
    for (uint32_t block = 0; block < p.blockDim; ++block) {
        AscendC::ResetPipelines();
        AscendC::blockIndex = block;
        const uint32_t mode = h.hasThreshold |
            (static_cast<uint32_t>(h.hasThreshold && h.thresholdBits == 16) << 1U) |
            (static_cast<uint32_t>(h.outputBits == 16) << 2U);
        // Independent expected-key map; do not share the production table.
        constexpr uint32_t staticBatches[] = {
            1U, 32U, 33U, 256U, 1024U, 4096U, 64U, 128U, 129U, 257U, 512U};
        constexpr uint32_t staticCoreRows[] = {
            64U, 64U, 64U, 256U, kExpectedRowsFor1024, kExpectedRowsFor4096,
            64U, 128U, 192U, 320U, 512U};
        constexpr uint32_t staticShapeCount = sizeof(staticBatches) / sizeof(staticBatches[0]);
        uint32_t shape = 0U;
        while (shape < staticShapeCount && staticBatches[shape] != h.batch) ++shape;
        const uint32_t ratio = h.ratio == 0.8f ? 0U : (h.ratio == 0.9f ? 1U : (h.ratio == 1.0f ? 2U : 3U));
        const uint32_t expectedCoreRows = shape < staticShapeCount ? staticCoreRows[shape] : 0U;
        const uint32_t staticCase = shape < staticShapeCount && p.rowsPerCore == expectedCoreRows
            ? (ratio < 3U ? 1U + shape * 3U + ratio : 34U + shape) : 0U;
        AscendC::tilingLoads = 0;
        DispatchKernel<T>(mode, staticCase,
            reinterpret_cast<GM_ADDR>(boxes.data()), reinterpret_cast<GM_ADDR>(anchors.data()),
            h.hasThreshold ? threshold : nullptr,
            reinterpret_cast<GM_ADDR>(output), (staticCase == 0U || staticCase >= 34U)
                ? reinterpret_cast<GM_ADDR>(&data) : nullptr);
        assert(AscendC::tilingLoads == (staticCase == 0U ? 1U : 0U));
        for (bool pending : AscendC::eventFlags) assert(!pending);
        assert(!AscendC::unifiedMaskEnabled && "unified vector mask must be restored at kernel exit");
        assert(!AscendC::vectorCounterMode && "normal mask mode must be restored at kernel exit");
    }
    assert(AscendC::thresholdReads + AscendC::thresholdCopies == (h.hasThreshold ? p.blockDim : 0));
    const bool scalarThreshold = SBVG_SCALAR_THRESHOLD != 0 && h.hasThreshold && h.thresholdBits == 32;
    assert(AscendC::scalarCacheInvalidations == (scalarThreshold ? p.blockDim : 0U));
    const bool knownShape = h.batch == 1 || h.batch == 32 || h.batch == 33 || h.batch == 64 ||
        h.batch == 128 || h.batch == 129 || h.batch == 256 || h.batch == 257 ||
        h.batch == 512 || h.batch == 1024 || h.batch == 4096;
    const uint32_t expectedRows = h.batch == 1024 ? kExpectedRowsFor1024 :
        h.batch == 4096 ? kExpectedRowsFor4096 : h.batch <= 64 ? 64 :
        h.batch == 129 ? 192 : h.batch == 257 ? 320 : h.batch;
    const bool ratioDma = knownShape && p.rowsPerCore == expectedRows &&
        h.ratio != 0.8f && h.ratio != 0.9f && h.ratio != 1.0f;
    assert(AscendC::ratioCopies == (ratioDma ? p.blockDim : 0U));
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
