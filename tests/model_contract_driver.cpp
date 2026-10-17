// Verify that the functional model actually rejects the hazards it reports.
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdlib>
#endif
#include "kernel_operator.h"
using namespace AscendC;

template<class T> auto HasDirectGather(int) -> decltype(
    GatherMask(std::declval<LocalTensor<T>>(), std::declval<LocalTensor<T>>(), uint8_t{4},
               std::declval<GatherMaskParams>()), std::true_type{});
template<class T> auto HasDirectGather(...) -> std::false_type;

template<class T> void CheckDirectGather() {
    InitSocState();
    LocalTensor<T> input(TPosition::VECIN, 0, 33 * 4);
    LocalTensor<T> y1(TPosition::VECCALC, 1024, 33);
    LocalTensor<T> y2(TPosition::VECCALC, 2048, 33);
    {
        PipeScope scope(Pipe::V);
        for (uint32_t i = 0; i < 33 * 4; ++i) input.SetValue(i, static_cast<T>(i));
    }
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(33 * 4);
#ifndef SBVG_CPU_MODEL_LEGACY_GATHER_ONLY
    GatherMask(y1, input, 4, {1, 1, 8, 0});
    GatherMask(y2, input, 6, {1, 1, 8, 0});
    assert(vectorCounterMode && unifiedVectorMask == 33 * 4);
#else
    uint64_t reserved = 0;
    GatherMask(y1, input, 4, true, 33 * 4, {1, 1, 8, 0}, reserved);
    assert(reserved == 33 && !vectorCounterMode);
    GatherMask(y2, input, 6, true, 33 * 4, {1, 1, 8, 0}, reserved);
    assert(reserved == 33 && !vectorCounterMode);
#endif
    ResetMask();
    SetMaskNorm();
    PipeBarrier<PIPE_V>();
    PipeScope scope(Pipe::V);
    for (uint32_t i = 0; i < 33; ++i) {
        assert(y1.GetValue(i) == T(i * 4 + 1));
        assert(y2.GetValue(i) == T(i * 4 + 3));
    }
}

template<class T> void CheckMultiBlockDma(const std::string& mode, uint32_t rows, bool gaps) {
    InitSocState();
    constexpr uint32_t lanes = 32 / sizeof(T);
    const uint32_t ubStride = gaps ? 1U : 0U;
    const uint32_t ubPitch = lanes * (1U + ubStride);
    const uint32_t viewElements = (rows - 1U) * ubPitch + lanes;
    std::vector<T> input(rows * 4U), output(rows * (gaps ? 2U : 1U), T(-2000));
    for (uint32_t i = 0; i < input.size(); ++i) input[i] = T(float(i) * 0.125f - 9.0f);
    RegisterGlobal(input.data(), input.size() * sizeof(T));
    RegisterGlobal(output.data(), output.size() * sizeof(T));
    GlobalTensor<T> source, dest;
    source.SetGlobalBuffer(input.data() + 3, input.size() - 3);
    dest.SetGlobalBuffer(output.data(), mode == "multiblock_gm_output_tail" ? rows - 1U : output.size());
    LocalTensor<T> expanded(TPosition::VECIN, 0,
                           viewElements - (mode == "multiblock_ub_input_tail" ? 1U : 0U));
    LocalTensor<T> result(TPosition::VECOUT, 8192,
                         viewElements - (mode == "multiblock_ub_output_tail" ? 1U : 0U));
    const uint32_t sourceGap = (mode == "multiblock_wrong_gm_stride" ? 4U : 3U) * sizeof(T);
    const uint32_t destinationGap = mode == "multiblock_wrong_ub_stride" ? 1U : ubStride;
    DataCopyPad(expanded, source,
                {uint16_t(rows + (mode == "multiblock_gm_input_tail" ? 1U : 0U)),
                 uint32_t(sizeof(T)), sourceGap, destinationGap, 0},
                DataCopyPadExtParams<T>{false, 0, 0, T(0)});
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    {
        PipeScope scope(Pipe::V);
        for (uint32_t row = 0; row < rows; ++row) {
            // Every row has a different scalar. This distinguishes per-row
            // implicit padding from accidental replication of input[3].
            for (uint32_t lane = 0; lane < lanes; ++lane)
                assert(expanded.GetValue(row * ubPitch + lane) == input[row * 4U + 3U]);
            result.SetValue(row * ubPitch, expanded.GetValue(row * ubPitch));
        }
    }
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopyPad(dest, result, {uint16_t(rows), uint32_t(sizeof(T)), ubStride,
                              gaps ? uint32_t(sizeof(T)) : 0U, 0});
    for (uint32_t row = 0; row < rows; ++row) {
        assert(output[row * (gaps ? 2U : 1U)] == input[row * 4U + 3U]);
        if (gaps) assert(output[row * 2U + 1U] == T(-2000));
    }
}

void CheckHighDimensionalCast(const std::string& mode) {
    InitSocState();
    constexpr uint32_t rows = 33;
    LocalTensor<half> source(TPosition::VECIN, 0,
                            rows * 16U - (mode == "highdim_cast_short_input" ? 9U : 0U));
    LocalTensor<float> output(TPosition::VECCALC, 2048,
                             rows * 8U - (mode == "highdim_cast_short_output" ? 1U : 0U));
    {
        PipeScope scope(Pipe::V);
        // Keep the final unavailable bytes uninitialized in the short-input
        // rejection test, while initializing every physically available byte.
        const uint32_t available = rows * 16U - (mode == "highdim_cast_short_input" ? 9U : 0U);
        for (uint32_t i = 0; i < available; ++i) source.SetValue(i, half(float(i) * 0.125f - 4.0f));
    }
    PipeBarrier<PIPE_V>();
    const uint8_t sourceStride = mode == "highdim_cast_wrong_source_stride" ? 2U : 1U;
    const uint8_t destinationStride = mode == "highdim_cast_wrong_destination_stride" ? 2U : 1U;
    Cast(output, source, RoundMode::CAST_NONE, uint64_t{8}, uint8_t{rows},
         {1, 1, destinationStride, sourceStride});
    PipeBarrier<PIPE_V>();
    {
        PipeScope scope(Pipe::V);
        for (uint32_t row = 0; row < rows; ++row)
            for (uint32_t lane = 0; lane < 8; ++lane)
                assert(output.GetValue(row * 8U + lane) == float(half(float(row * 16U + lane) * 0.125f - 4.0f)));
    }
}

void CheckDenseHighDimensionalCast() {
    InitSocState();
    LocalTensor<half> source(TPosition::VECIN, 0, 128);
    LocalTensor<float> converted(TPosition::VECCALC, 256, 128);
    LocalTensor<half> roundtrip(TPosition::VECOUT, 768, 128);
    {
        PipeScope scope(Pipe::V);
        for (uint32_t i = 0; i < 128; ++i) source.SetValue(i, half(float(i) * 0.125f - 4.0f));
    }
    PipeBarrier<PIPE_V>();
    Cast(converted, source, RoundMode::CAST_NONE, uint64_t{64}, uint8_t{2}, {1, 1, 8, 4});
    PipeBarrier<PIPE_V>();
    Cast(roundtrip, converted, RoundMode::CAST_NONE, uint64_t{64}, uint8_t{2}, {1, 1, 4, 8});
    PipeBarrier<PIPE_V>();
    PipeScope scope(Pipe::V);
    for (uint32_t i = 0; i < 128; ++i) assert(roundtrip.GetValue(i) == half(float(i) * 0.125f - 4.0f));
}

void CheckMaskedColumnCast(const std::string& mode, uint32_t rows) {
    InitSocState();
    constexpr uint32_t columns = 3U;
    constexpr uint32_t columnRows = 64U;
    LocalTensor<half> source(TPosition::VECIN, 0U, columns * columnRows);
    LocalTensor<float> converted(TPosition::VECCALC, 512U, columns * columnRows);
    {
        PipeScope scope(Pipe::V);
        // Only the values produced by the three Gathers exist. Leave every
        // inter-column tail uninitialized so a full-mask conversion must fail.
        for (uint32_t column = 0U; column < columns; ++column)
            for (uint32_t row = 0U; row < rows; ++row) {
                const float value = float((column + 1U) * 257U + row * 3U) * 0.125f - 64.0f;
                source.SetValue(column * columnRows + row, half(value));
            }
    }
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(mode == "masked_column_cast_full_mask" ? 64U : rows);
    const uint8_t sourceStride = mode == "masked_column_cast_wrong_stride" ? 8U : 4U;
    Cast<float, half, false>(converted, source, RoundMode::CAST_NONE,
                            MASK_PLACEHOLDER, uint8_t{columns}, {1U, 1U, 8U, sourceStride});
    PipeBarrier<PIPE_V>();
    ResetMask();
    assert(!vectorCounterMode);
    PipeScope scope(Pipe::V);
    for (uint32_t column = 0U; column < columns; ++column)
        for (uint32_t row = 0U; row < columnRows; ++row) {
            const uint32_t index = column * columnRows + row;
            // Inspect every destination byte rather than reading inactive
            // values: inactive output lanes must remain uninitialized.
            for (uint32_t byte = 0U; byte < sizeof(float); ++byte)
                assert(converted.memory->initialized[converted.offset + index * sizeof(float) + byte]
                       == (row < rows));
            if (row < rows) {
                const float value = float((column + 1U) * 257U + row * 3U) * 0.125f - 64.0f;
                assert(converted.GetValue(index) == float(half(value)));
            }
        }
}

void CheckBroadcastHighDimensionalMul(const std::string& mode) {
    InitSocState();
    const bool counter = mode == "positive_highdim_mul_counter";
    constexpr uint32_t kRepeatElements = 64U;
    constexpr uint32_t kOutputElements = 384U;
    const uint32_t count = counter ? 137U : 3U * kRepeatElements;
    LocalTensor<float> source(TPosition::VECIN, 0, count);
    // Different values in the eight lanes distinguish block broadcasting
    // from the incorrect broadcast of only the first element.
    LocalTensor<float> factors(TPosition::VECIN, 1024, 8);
    LocalTensor<float> output(TPosition::VECOUT, 1056, kOutputElements);
    {
        PipeScope scope(Pipe::V);
        for (uint32_t i = 0; i < count; ++i)
            source.SetValue(i, (float((i * 17U + 3U) % 127U) - 63.0f) * 0.125f);
        for (uint32_t lane = 0; lane < 8U; ++lane)
            factors.SetValue(lane, (float(lane) - 3.0f) * 0.25f);
    }
    Duplicate(output, -2000.0f, kOutputElements);
    if (mode != "highdim_mul_missing_input_barrier") PipeBarrier<PIPE_V>();
    BinaryRepeatParams strides{2, 1, 0, 16, 8, 0};
    if (mode == "highdim_mul_wrong_broadcast_block_stride") strides.src1BlkStride = 1;
    if (mode == "highdim_mul_wrong_broadcast_repeat_stride") strides.src1RepStride = 1;
    if (mode == "highdim_mul_wrong_source_stride") strides.src0BlkStride = 2;
    if (mode == "highdim_mul_wrong_destination_stride") strides.dstBlkStride = 1;
    if (counter) {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(count);
        Mul<float, false>(output, source, factors, MASK_PLACEHOLDER, uint8_t{1}, strides);
    } else {
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(kRepeatElements);
        Mul<float, false>(output, source, factors, MASK_PLACEHOLDER, uint8_t{3}, strides);
    }
    ResetMask();
    SetMaskNorm();
    if (mode != "highdim_mul_missing_result_barrier") PipeBarrier<PIPE_V>();
    std::array<float, kOutputElements> expected;
    expected.fill(-2000.0f);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t repeat = i / kRepeatElements;
        const uint32_t within = i % kRepeatElements;
        const uint32_t destination = repeat * 128U + (within / 8U) * 16U + within % 8U;
        const float value = (float((i * 17U + 3U) % 127U) - 63.0f) * 0.125f;
        const float factor = (float(i % 8U) - 3.0f) * 0.25f;
        expected[destination] = value * factor;
    }
    PipeScope scope(Pipe::V);
    // Check all gaps as well as computed lanes, so an incorrect destination
    // stride cannot pass merely by writing the right values elsewhere.
    for (uint32_t i = 0; i < kOutputElements; ++i)
        assert(output.GetValue(i) == expected[i]);
}

void CheckRatioFieldDma(const std::string& mode) {
    InitSocState();
    std::array<float, 4> metadata{1001.0f, 2002.0f, 0.625f, -3003.0f};
    RegisterGlobal(metadata.data(), sizeof(metadata));
    ratioAddress = metadata.data() + 2U;
    ratioCopies = 0U;
    GlobalTensor<float> source;
    source.SetGlobalBuffer(metadata.data() + 2U, 2U);
    LocalTensor<float> ratio(TPosition::VECIN, 0, 8U);
    const uint16_t blocks = mode == "ratio_dma_multiple_blocks" ? 2U : 1U;
    const uint32_t bytes = mode == "ratio_dma_extra_field" ? 8U : 4U;
    DataCopyPad(ratio, source, {blocks, bytes, 0U, 0U, 0U},
                DataCopyPadExtParams<float>{true, 0U, 0U, 0.0f});
    assert(ratioCopies == 1U);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    PipeScope scope(Pipe::V);
    for (uint32_t i = 0; i < 8U; ++i) assert(ratio.GetValue(i) == 0.625f);
}

// Separate event IDs expose only copies preceding the matching SetFlag.
// Waiting an older event after a newer one must not revoke visibility.
void CheckSplitEvents(const std::string& mode) {
    InitSocState();
    LocalTensor<float> first(TPosition::VECIN, 0U, 8U);
    LocalTensor<float> second(TPosition::VECIN, 32U, 8U);
    {
        PipeScope scope(Pipe::MTE2);
        first.SetValue(0U, 3.0f);
    }
    SetFlag<HardEvent::MTE2_V>(0);
    {
        PipeScope scope(Pipe::MTE2);
        second.SetValue(0U, 7.0f);
    }
    SetFlag<HardEvent::MTE2_V>(1);
    if (mode == "split_event_wrong_id") WaitFlag<HardEvent::MTE2_V>(2);
    if (mode == "split_event_duplicate_set") SetFlag<HardEvent::MTE2_V>(1);
    if (mode == "split_event_invalid_id") SetFlag<HardEvent::MTE2_V>(8);
    if (mode == "positive_reverse_events") {
        WaitFlag<HardEvent::MTE2_V>(1);
        WaitFlag<HardEvent::MTE2_V>(0);
    } else {
        WaitFlag<HardEvent::MTE2_V>(0);
        {
            PipeScope scope(Pipe::V);
            assert(first.GetValue(0U) == 3.0f);
            if (mode == "split_event_late_copy") (void)second.GetValue(0U);
        }
        WaitFlag<HardEvent::MTE2_V>(1);
    }
    PipeScope scope(Pipe::V);
    assert(first.GetValue(0U) == 3.0f && second.GetValue(0U) == 7.0f);
    for (bool pending : eventFlags) assert(!pending);
}

void CheckScalarCacheRefresh(const std::string& mode) {
    // Register only actual scalar tensors. A model that reads a physical 64B
    // line eagerly would over-read these exact four-byte registrations.
    alignas(64) std::array<float, 17> storage{};
    storage[1] = 3.5f;
    storage[16] = 23.0f;
    RegisterGlobal(storage.data(), sizeof(float));
    RegisterGlobal(storage.data() + 1U, sizeof(float));
    RegisterGlobal(storage.data() + 16U, sizeof(float));
    GlobalTensor<float> source, neighbor, otherLine;
    source.SetGlobalBuffer(storage.data(), 1U);
    neighbor.SetGlobalBuffer(storage.data() + 1U, 1U);
    otherLine.SetGlobalBuffer(storage.data() + 16U, 1U);
    thresholdAddress = storage.data();
    scalarCacheInvalidations = 0U;
    scalarGlobalCache.clear();
    blockIndex = 0U;
    uint32_t iteration = 0U;
    for (float expected : {0.0f, 7.5f, 0.0f}) {
        storage[0] = expected;
        if (iteration == 1U) {
            storage[1] = 6.25f;
            storage[16] = 47.0f;
        }
        // Reset the execution state while deliberately retaining per-core
        // cached elements from the preceding invocation at the same address.
        ResetPipelines();
        InitSocState();
        LocalTensor<float> input(TPosition::VECCALC, 0U, 8U);
        LocalTensor<float> output(TPosition::VECCALC, 32U, 8U);
        Duplicate(input, 2.0f, 8U);
        PipeBarrier<PIPE_V>();
        if (!(mode == "scalar_cache_missing_refresh" && iteration == 1U))
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(source);
        const float scalar = source.GetValue(0U);
        assert(scalar == expected && "scalar GM read reused a stale per-core cache entry");
        SetFlag<HardEvent::S_V>(2);
        WaitFlag<HardEvent::S_V>(2);
        Adds(output, input, scalar, 8U);
        PipeBarrier<PIPE_V>();
        {
            PipeScope scope(Pipe::V);
            for (uint32_t i = 0U; i < 8U; ++i)
                assert(output.GetValue(i) == 2.0f + expected);
        }
        if (iteration == 0U) {
            assert(neighbor.GetValue(0U) == 3.5f);
            assert(otherLine.GetValue(0U) == 23.0f);
            blockIndex = 1U;
            assert(source.GetValue(0U) == 0.0f);
            blockIndex = 0U;
        } else if (iteration == 1U) {
            // Invalidating source also refreshes other cached elements in its
            // line, but preserves the other line and other physical cores.
            assert(neighbor.GetValue(0U) == 6.25f);
            assert(otherLine.GetValue(0U) == 23.0f);
            blockIndex = 1U;
            assert(source.GetValue(0U) == 0.0f);
            blockIndex = 0U;
        }
        for (bool pending : eventFlags) assert(!pending);
        ++iteration;
    }
    assert(scalarCacheInvalidations == 3U);
    storage[0] = 9.0f;
    DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(source);
    assert(source.GetValue(0U) == 9.0f);
    blockIndex = 1U;
    assert(source.GetValue(0U) == 0.0f);
    DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(source);
    assert(source.GetValue(0U) == 9.0f);
    assert(scalarCacheInvalidations == 5U);
    blockIndex = 0U;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    // Intentional assertion failures are subprocess rejection tests. Keep
    // Windows crash-report UI from delaying each expected rejection.
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    _set_abort_behavior(0, _CALL_REPORTFAULT);
    _set_error_mode(_OUT_TO_STDERR);
#endif
    assert(argc == 2);
    const std::string mode(argv[1]);
    ResetPipelines();
    if (mode == "positive_scalar_cache_refresh" || mode == "scalar_cache_missing_refresh") {
        CheckScalarCacheRefresh(mode);
        return 0;
    }
    if (mode == "positive_masked_column_cast" || mode.rfind("masked_column_cast_", 0) == 0) {
        if (mode == "positive_masked_column_cast") {
            for (uint32_t rows = 1U; rows <= 64U; ++rows) {
                ResetPipelines();
                CheckMaskedColumnCast(mode, rows);
            }
        } else CheckMaskedColumnCast(mode, 33U);
        return 0;
    }
    if (mode == "positive_split_events" || mode == "positive_reverse_events" ||
        mode.rfind("split_event_", 0) == 0) {
        CheckSplitEvents(mode);
        return 0;
    }
    if (mode == "positive_ratio_field_dma" || mode.rfind("ratio_dma_", 0) == 0) {
        CheckRatioFieldDma(mode);
        return 0;
    }
    if (mode == "positive_highdim_mul_broadcast" || mode == "positive_highdim_mul_counter" ||
        mode.rfind("highdim_mul_", 0) == 0) {
        CheckBroadcastHighDimensionalMul(mode);
        return 0;
    }
    if (mode.rfind("positive_multiblock", 0) == 0 || mode.rfind("multiblock_", 0) == 0) {
        if (mode.rfind("positive_multiblock", 0) == 0) {
            for (const uint32_t rows : {1U, 32U, 33U}) {
                CheckMultiBlockDma<float>(mode, rows, mode == "positive_multiblock_gaps");
                ResetPipelines();
                CheckMultiBlockDma<half>(mode, rows, mode == "positive_multiblock_gaps");
                ResetPipelines();
            }
        } else CheckMultiBlockDma<float>(mode, 33, false);
        return 0;
    }
    if (mode == "positive_highdim_cast_rows" || mode.rfind("highdim_cast_", 0) == 0) {
        CheckHighDimensionalCast(mode);
        return 0;
    }
    if (mode == "positive_highdim_cast_dense") {
        CheckDenseHighDimensionalCast();
        return 0;
    }
    if (mode == "positive_counter_arithmetic_65" || mode == "positive_counter_arithmetic_4096") {
        InitSocState();
        const uint32_t rows = mode == "positive_counter_arithmetic_65" ? 65U : 4096U;
        LocalTensor<float> input(TPosition::VECCALC, 0, rows);
        LocalTensor<float> scalar(TPosition::VECCALC, 16384, 8);
        LocalTensor<float> output(TPosition::VECOUT, 16416, rows);
        Duplicate(input, 3.0f, rows);
        Duplicate(scalar, 0.25f, 8);
        PipeBarrier<PIPE_V>();
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(rows);
        Muls<float, false>(input, input, 2.0f, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Add<float, false>(output, input, scalar, MASK_PLACEHOLDER, 1, {1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        Sub<float, false>(output, output, scalar, MASK_PLACEHOLDER, 1, {1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        Adds<float, false>(output, output, 0.5f, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        ResetMask();
        SetMaskNorm();
        PipeScope scope(Pipe::V);
        for (uint32_t i = 0; i < rows; ++i) assert(output.GetValue(i) == 6.5f);
        return 0;
    }
    if (mode == "positive_vector_barriers" || mode == "positive_all_barrier" ||
        mode == "missing_vector_raw" || mode == "missing_vector_chain_barrier") {
        InitSocState();
        LocalTensor<float> input(TPosition::VECCALC, 0, 64);
        LocalTensor<float> alias(TPosition::VECIN, 0, 64);
        LocalTensor<float> output(TPosition::VECOUT, 256, 64);
        Duplicate(input, 3.0f, 64);
        if (mode == "positive_all_barrier") PipeBarrier<PIPE_ALL>();
        else if (mode != "missing_vector_raw") PipeBarrier<PIPE_V>();
        // The first multiply reads and writes the same instruction's input.
        // In-place operations are legal; a later instruction needs a barrier.
        Muls(alias, input, 2.0f, 64);
        if (mode != "missing_vector_chain_barrier") PipeBarrier<PIPE_V>();
        Muls(output, alias, 2.0f, 64);
        PipeBarrier<PIPE_V>();
        PipeScope scope(Pipe::V);
        for (uint32_t i = 0; i < 64; ++i) assert(output.GetValue(i) == 12.0f);
        return 0;
    }
    if (mode == "positive_counter_mask_reset" || mode == "counter_reset_without_norm") {
        InitSocState();
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(512);
        ResetMask();
        assert(!unifiedMaskEnabled && vectorCounterMode &&
               "ResetMask must leave counter mode active");
        if (mode == "positive_counter_mask_reset") SetMaskNorm();
        assert(!vectorCounterMode && "kernel exit requires normal mask mode");
        return 0;
    }
    if (mode == "positive_gather_overload_detection") {
#ifndef SBVG_CPU_MODEL_LEGACY_GATHER_ONLY
        static_assert(decltype(HasDirectGather<float>(0))::value);
        static_assert(decltype(HasDirectGather<half>(0))::value);
#else
        static_assert(!decltype(HasDirectGather<float>(0))::value);
        static_assert(!decltype(HasDirectGather<half>(0))::value);
#endif
        CheckDirectGather<float>();
        CheckDirectGather<half>();
        return 0;
    }
#ifndef SBVG_CPU_MODEL_LEGACY_GATHER_ONLY
    if (mode == "counter_gather_without_mask") {
        InitSocState();
        LocalTensor<float> input(TPosition::VECIN, 0, 64);
        LocalTensor<float> output(TPosition::VECCALC, 256, 16);
        Duplicate(input, 3.0f, 64);
        SetMaskCount();
        GatherMask(output, input, 4, {1, 1, 8, 0});
        return 0;
    }
#endif
    if (mode == "positive_direct_select_128" || mode == "positive_direct_select_512" ||
        mode == "positive_direct_select_65" || mode == "missing_cmpmask_scalar") {
        InitSocState();
        const uint32_t rows = mode == "positive_direct_select_128" ? 128U :
                              mode == "positive_direct_select_65" ? 65U : 512U;
        const uint32_t padded = (rows + 63) / 64 * 64;
        LocalTensor<float> source(TPosition::VECCALC, 0, padded);
        LocalTensor<float> boundary(TPosition::VECCALC, 2048, padded);
        LocalTensor<float> zeros(TPosition::VECCALC, 4096, 8);
        LocalTensor<float> ones(TPosition::VECCALC, 4128, padded);
        LocalTensor<uint8_t> mask(TPosition::VECCALC, 6176, padded / 8);
        LocalTensor<float> output(TPosition::VECOUT, 6272, padded);
        Duplicate(boundary, 0.0f, padded);
        Duplicate(zeros, -7.25f, 8);
        {
            PipeScope scope(Pipe::V);
            for (uint32_t i = 0; i < padded; ++i) {
                // Distinct bit patterns in successive repeats catch mask
                // reuse, a 32 B mask repeat stride, or byte/bits confusion.
                const bool selected = ((i * 17U + i / 64U * 13U) % 31U) < 11U;
                source.SetValue(i, selected ? 1.0f : -1.0f);
                ones.SetValue(i, float(i) + 0.5f);
            }
        }
        PipeBarrier<PIPE_V>();
        if (mode != "missing_cmpmask_scalar") SetCmpMask(zeros);
        // Memory-output Compare must preserve the scalar stored in CMPMASK.
        Compare(mask, source, boundary, CMPMODE::GT, padded);
        PipeBarrier<PIPE_V>();
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(rows);
        Select(output, mask, ones, uint8_t{1}, {1, 1, 1, 8, 8, 8});
        ResetMask();
        SetMaskNorm();
        PipeBarrier<PIPE_V>();
        {
            PipeScope scope(Pipe::V);
            for (uint32_t i = 0; i < rows; ++i) {
                const bool selected = ((i * 17U + i / 64U * 13U) % 31U) < 11U;
                assert(output.GetValue(i) == (selected ? float(i) + 0.5f : -7.25f));
            }
        }
        if (rows % 64U == 0U) {
            // Normal-mask repeats exercise the same compressed mask strides
            // without counter mode, broadcasting one 32 B true-data block.
            SetMaskNorm();
            SetVectorMask<float, MaskMode::NORMAL>(64);
            Select(output, mask, ones, uint8_t(rows / 64U), {1, 0, 1, 8, 0, 8});
            ResetMask();
            PipeBarrier<PIPE_V>();
            PipeScope scope(Pipe::V);
            for (uint32_t i = 0; i < rows; ++i) {
                const bool selected = ((i * 17U + i / 64U * 13U) % 31U) < 11U;
                assert(output.GetValue(i) == (selected ? float(i % 8U) + 0.5f : -7.25f));
            }
        }
        return 0;
    }
    if (mode == "positive_counter_gather") {
        InitSocState();
        LocalTensor<float> input(TPosition::VECIN, 0, 33 * 4);
        LocalTensor<float> column(TPosition::VECCALC, 1024, 33);
        {
            PipeScope scope(Pipe::V);
            for (uint32_t i = 0; i < 33 * 4; ++i) input.SetValue(i, static_cast<float>(i));
        }
        PipeBarrier<PIPE_V>();
        uint64_t count = 0;
        GatherMask(column, input, 4, true, 33 * 4, {1, 1, 8, 0}, count);
        assert(count == 33);
        PipeBarrier<PIPE_V>();
        PipeScope scope(Pipe::V);
        for (uint32_t i = 0; i < 33; ++i) assert(column.GetValue(i) == i * 4 + 1);
        return 0;
    }
    if (mode == "positive_cmpmask_spr" || mode == "missing_cmpmask_spr" ||
        mode == "missing_cmpmask_barrier") {
        InitSocState();
        LocalTensor<float> source(TPosition::VECCALC, 0, 33);
        LocalTensor<float> boundary(TPosition::VECCALC, 256, 33);
        LocalTensor<float> output(TPosition::VECOUT, 512, 33);
        LocalTensor<float> ones(TPosition::VECCALC, 512, 8);
        LocalTensor<float> zeros(TPosition::VECCALC, 768, 8);
        Duplicate(boundary, 0.0f, 33);
        Duplicate(ones, 1.0f, 8);
        Duplicate(zeros, 0.0f, 8);
        {
            PipeScope scope(Pipe::V);
            for (uint32_t i = 0; i < 33; ++i) source.SetValue(i, i % 3 == 1 ? 1.0f : 0.0f);
        }
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(33);
        if (mode != "missing_cmpmask_spr") Compare(source, boundary, CMPMODE::GT, 33, {});
        if (mode != "missing_cmpmask_barrier") PipeBarrier<PIPE_V>();
        Select<float, SELMODE::VSEL_CMPMASK_SPR>(output, ones, zeros, 1, {1, 0, 0, 8, 0, 0});
        ResetMask();
        PipeBarrier<PIPE_V>();
        PipeScope scope(Pipe::V);
        for (uint32_t i = 0; i < 33; ++i) assert(output.GetValue(i) == (i % 3 == 1 ? 1.0f : 0.0f));
        return 0;
    }
    if (mode.rfind("raw_", 0) == 0 || mode == "positive_raw_views") {
        InitSocState();
        LocalTensor<float> input(TPosition::VECIN, 0, 64);
        LocalTensor<float> alias(TPosition::VECCALC, 0, 64);
        LocalTensor<float> result(TPosition::VECOUT, 256, 64);
        if (mode == "raw_uninitialized") Muls(result, input, 1.0f, 64);
        if (mode == "raw_view_bounds") {
            LocalTensor<float> shortView(TPosition::VECCALC, 0, 8);
            Duplicate(shortView, 1.0f, 9);
        }
        if (mode == "raw_ub_bounds") {
            LocalTensor<float> outside(TPosition::VECCALC, kUnifiedBufferBytes, 1);
        }
        Duplicate(input, 3.0f, 64);
        PipeBarrier<PIPE_V>();
        if (mode != "raw_missing_mask") {
            SetMaskNorm();
            SetVectorMask<float, MaskMode::NORMAL>(64);
        }
        Muls<float, false>(alias, input, 2.0f, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        // alias and input deliberately share physical UB despite logical position.
        Add<float, false>(result, alias, input, MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
        ResetMask();
        PipeBarrier<PIPE_V>();
        PipeScope scope(Pipe::V);
        for (uint32_t i = 0; i < 64; ++i) assert(result.GetValue(i) == 12.0f);
        return 0;
    }
    std::array<float, 8> input{3, 9, 11, 13, 17, 19, 23, 29}, output{};
    RegisterGlobal(input.data(), sizeof(input));
    RegisterGlobal(output.data(), sizeof(output));
    GlobalTensor<float> source, dest;
    source.SetGlobalBuffer(input.data(), mode == "gm_bounds" ? 9 : 8);
    dest.SetGlobalBuffer(output.data(), 8);
    TPipe pipe;
    TBuf<TPosition::VECIN> in;
    TBuf<TPosition::VECOUT> out;
    TBuf<TPosition::VECCALC> calc;
    pipe.InitBuffer(in, 32);
    pipe.InitBuffer(out, 32);
    pipe.InitBuffer(calc, 32);
    auto x = in.Get<float>(), y = out.Get<float>();
    if (mode == "uninitialized") Muls(y, x, 1.0f, 8);
    if (mode == "illegal_dma_position") x = calc.Get<float>();
    const bool implicitPadding = mode == "positive";
    const bool explicitPadding = mode == "positive_explicit_padding";
    const DataCopyExtParams inputCopy{1, implicitPadding || explicitPadding ? 4U : 32U, 0, 0, 0};
    const DataCopyExtParams outputCopy{1, 32, 0, 0, 0};
    if (mode == "positive_aligned_copy" || mode == "unaligned_copy_count") {
        DataCopy(x, source, mode == "unaligned_copy_count" ? 7U : 8U);
    } else {
        DataCopyPad(x, source, inputCopy,
                    DataCopyPadExtParams<float>{true, 0, uint8_t(explicitPadding ? 7 : 0),
                                               explicitPadding ? 0.0f : -123.0f});
    }
    if (mode == "wait_without_set") WaitFlag<HardEvent::MTE2_V>(0);
    if (mode != "missing_input_sync") {
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    }
    Muls(y, x, 1.0f, 8);
    if (mode != "missing_output_sync") {
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
    }
    if (mode == "positive_aligned_copy") DataCopy(dest, y, 8U);
    else DataCopyPad(dest, y, outputCopy);
    if (mode == "positive_aligned_copy") assert(output == input);
    if (implicitPadding) for (float v : output) assert(v == 3);
    if (explicitPadding) {
        assert(output[0] == 3);
        for (size_t i = 1; i < output.size(); ++i) assert(output[i] == 0);
    }
    return 0;
}
