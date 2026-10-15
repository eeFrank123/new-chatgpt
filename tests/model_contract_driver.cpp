// Verify that the functional model actually rejects the hazards it reports.
#include <string>
#include "kernel_operator.h"
using namespace AscendC;

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode(argv[1]);
    ResetPipelines();
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
    DataCopyPad(x, source, inputCopy,
                DataCopyPadExtParams<float>{true, 0, uint8_t(explicitPadding ? 7 : 0),
                                           explicitPadding ? 0.0f : -123.0f});
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
    DataCopyPad(dest, y, outputCopy);
    if (implicitPadding) for (float v : output) assert(v == 3);
    if (explicitPadding) {
        assert(output[0] == 3);
        for (size_t i = 1; i < output.size(); ++i) assert(output[i] == 0);
    }
    return 0;
}
