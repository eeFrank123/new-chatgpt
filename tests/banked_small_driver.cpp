#include <cmath>
#include <iostream>
#include <random>
#include <vector>
// Exercise every small B independently of Host metadata specialization. Exact
// GM extents and initialized-UB checks catch gaps exposed by either layout.
#include "../code/op_kernel/safety_belt_violation_geometry_banked.h"
#ifdef SBVG_REVIEW_SCALAR_SMALL
#include "../code/op_kernel/safety_belt_violation_geometry_scalar.h"
#endif

template<class T, uint32_t Mode, bool DynamicRatio, bool UnitRatio = false>
void Run(uint32_t batch, float ratio, std::mt19937& rng, uint64_t& rowsChecked) {
    constexpr bool hasThreshold = (Mode & 1U) != 0U;
    constexpr bool halfThreshold = (Mode & 2U) != 0U;
    using Output = typename std::conditional<(Mode & 4U) != 0U, half, float>::type;
    std::vector<T> boxes(batch * 4), anchors(batch * 2);
    std::vector<Output> output(batch + 16, Output(-12344.0f));
    const float thresholdValue = (rng() % 2000) / 128.0f;
    float thresholdFloat = thresholdValue;
    half thresholdHalf = half(thresholdValue);
    const float thresholdUsed = hasThreshold ?
        (halfThreshold ? float(thresholdHalf) : thresholdFloat) : 0.0f;
    std::vector<float> expected(batch);
    for (uint32_t row = 0; row < batch; ++row) {
        boxes[4*row] = T(0.0f);
        boxes[4*row+1] = T((int(rng() % 16000) - 8000) / 4.0f);
        boxes[4*row+2] = T(100.0f);
        boxes[4*row+3] = T(float(boxes[4*row+1]) + (rng() % 16000) / 4.0f);
        const float y1 = float(boxes[4*row+1]), y2 = float(boxes[4*row+3]);
        volatile float height = y2 - y1;
        volatile float scaled = float(height) * ratio;
        volatile float reference = y1 + float(scaled);
        volatile float boundary = float(reference) + thresholdUsed;
        float cy = row % 5 == 0 ? float(boundary) :
            (row % 5 == 1 ? std::nextafter(float(boundary), INFINITY) :
             (row % 5 == 2 ? std::nextafter(float(boundary), -INFINITY) :
              float(boundary) + (row % 5 == 3 ? 4.0f : -4.0f)));
        anchors[2*row] = T(50.0f);
        anchors[2*row+1] = T(cy);
        expected[row] = float(anchors[2*row+1]) > float(boundary) ? 1.0f : 0.0f;
    }
    SafetyBeltViolationGeometryTilingData data{batch, 64U, ratio, 0.0f};
    GM_ADDR threshold = reinterpret_cast<GM_ADDR>(halfThreshold ?
        static_cast<void*>(&thresholdHalf) : static_cast<void*>(&thresholdFloat));
    AscendC::globalAllocations.clear();
    AscendC::RegisterGlobal(boxes.data(), boxes.size()*sizeof(T));
    AscendC::RegisterGlobal(anchors.data(), anchors.size()*sizeof(T));
    AscendC::RegisterGlobal(output.data()+8, batch*sizeof(Output));
    AscendC::RegisterGlobal(&data, sizeof(data));
    if (hasThreshold) AscendC::RegisterGlobal(threshold, halfThreshold ? 2U : 4U);
    AscendC::thresholdAddress = hasThreshold ? threshold : nullptr;
    AscendC::ratioAddress = &data.waistRatio;
    AscendC::outputAddress = output.data()+8;
    AscendC::outputWrites.assign(batch, 0U);
    AscendC::ResetPipelines();
    AscendC::InitSocState();
#ifdef SBVG_REVIEW_SCALAR_SMALL
    RunSafetyBeltScalarSmall<T, Mode, DynamicRatio, UnitRatio>(
#else
    RunSafetyBeltBankedSmall<T, Mode, DynamicRatio, UnitRatio>(
#endif
        reinterpret_cast<GM_ADDR>(boxes.data()), reinterpret_cast<GM_ADDR>(anchors.data()),
        hasThreshold ? threshold : nullptr, reinterpret_cast<GM_ADDR>(output.data()+8),
        data, DynamicRatio ? reinterpret_cast<GM_ADDR>(&data) : nullptr);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint32_t row = 0; row < batch; ++row) {
        assert(float(output[row+8]) == expected[row]);
        assert(AscendC::outputWrites[row] == 1U);
    }
    for (bool pending : AscendC::eventFlags) assert(!pending);
    assert(!AscendC::unifiedMaskEnabled && !AscendC::vectorCounterMode);
    for (uint32_t i = 0; i < 8; ++i)
        assert(float(output[i]) == -12344.0f && float(output[batch+8+i]) == -12344.0f);
    rowsChecked += batch;
}

int main() {
    std::mt19937 rng(20261017);
    uint64_t cases = 0U, rows = 0U;
    for (uint32_t batch = 1U; batch <= 64U; ++batch) {
        for (float ratio : {0.8f, 0.9f, 1.0f, 0.6180339f}) {
            for (uint32_t repeat = 0U; repeat < 3U; ++repeat) {
#define REVIEW(T, M, D) Run<T, M, D>(batch, ratio, rng, rows); ++cases
                REVIEW(float, 0U, false); REVIEW(float, 1U, false);
                REVIEW(float, 0U, true); REVIEW(float, 1U, true);
                REVIEW(half, 0U, false); REVIEW(half, 1U, false);
                REVIEW(half, 3U, false); REVIEW(half, 4U, false);
                REVIEW(half, 5U, false); REVIEW(half, 7U, false);
                REVIEW(half, 0U, true); REVIEW(half, 1U, true);
                REVIEW(half, 3U, true); REVIEW(half, 4U, true);
                REVIEW(half, 5U, true); REVIEW(half, 7U, true);
#undef REVIEW
                if (ratio == 1.0f) {
#define REVIEW_UNIT(T, M) Run<T, M, false, true>(batch, ratio, rng, rows); ++cases
                    REVIEW_UNIT(float, 0U); REVIEW_UNIT(float, 1U);
                    REVIEW_UNIT(half, 0U); REVIEW_UNIT(half, 1U);
                    REVIEW_UNIT(half, 3U); REVIEW_UNIT(half, 4U);
                    REVIEW_UNIT(half, 5U); REVIEW_UNIT(half, 7U);
#undef REVIEW_UNIT
                }
            }
        }
    }
    std::cout << cases << " cases, " << rows << " rows passed\n";
}
