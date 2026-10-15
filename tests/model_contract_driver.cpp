// Verify that the functional model actually rejects the hazards it reports.
#include <string>
#include "kernel_operator.h"
using namespace AscendC;

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode(argv[1]);
    ResetPipelines();
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
