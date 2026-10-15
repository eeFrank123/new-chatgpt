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
    PipeScope scope(Pipe::V);
    for (uint32_t i = 0; i < 33; ++i) {
        assert(y1.GetValue(i) == T(i * 4 + 1));
        assert(y2.GetValue(i) == T(i * 4 + 3));
    }
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
        if (mode != "missing_cmpmask_scalar") SetCmpMask(zeros);
        // Memory-output Compare must preserve the scalar stored in CMPMASK.
        Compare(mask, source, boundary, CMPMODE::GT, padded);
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(rows);
        Select(output, mask, ones, uint8_t{1}, {1, 1, 1, 8, 8, 8});
        ResetMask();
        SetMaskNorm();
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
        uint64_t count = 0;
        GatherMask(column, input, 4, true, 33 * 4, {1, 1, 8, 0}, count);
        assert(count == 33);
        PipeScope scope(Pipe::V);
        for (uint32_t i = 0; i < 33; ++i) assert(column.GetValue(i) == i * 4 + 1);
        return 0;
    }
    if (mode == "positive_cmpmask_spr" || mode == "missing_cmpmask_spr") {
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
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(33);
        if (mode != "missing_cmpmask_spr") Compare(source, boundary, CMPMODE::GT, 33, {});
        Select<float, SELMODE::VSEL_CMPMASK_SPR>(output, ones, zeros, 1, {1, 0, 0, 8, 0, 0});
        ResetMask();
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
        if (mode != "raw_missing_mask") {
            SetMaskNorm();
            SetVectorMask<float, MaskMode::NORMAL>(64);
        }
        Muls<float, false>(alias, input, 2.0f, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
        // alias and input deliberately share physical UB despite logical position.
        Add<float, false>(result, alias, input, MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
        ResetMask();
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
