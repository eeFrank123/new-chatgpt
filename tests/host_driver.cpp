#include <iostream>
#include <limits>
#include "../code/op_host/safety_belt_violation_geometry.cpp"

bool Supports(const ops::OpDef& op, ge::DataType input, ge::DataType threshold,
              ge::DataType output, bool present) {
    const auto& a = op.inputs.at("person_boxes");
    const auto& b = op.inputs.at("anchor_points");
    const auto& t = op.inputs.at("threshold");
    const auto& o = op.outputs.at("violation_score");
    assert(a.types.size() == b.types.size() && a.types.size() == t.types.size() && a.types.size() == o.types.size());
    for (const auto* p : {&a, &b, &t, &o}) assert(p->types.size() == p->formats.size());
    for (size_t i = 0; i < a.types.size(); ++i) {
        if (a.types[i] == input && b.types[i] == input &&
            (!present || t.types[i] == threshold) && o.types[i] == output) return true;
    }
    return false;
}

int main() {
    ops::SafetyBeltViolationGeometry op("SafetyBeltViolationGeometry");
    // Original supplied template's float16 ABI; the old implementation rejects it.
    if (!Supports(op, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, true)) {
        std::cerr << "FAIL: original-template float16 input/threshold/output ABI is not registered\n";
        return 1;
    }
    uint32_t cases = 0;
    for (auto input : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        for (auto threshold : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
            for (auto output : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
                for (bool present : {false, true}) {
                    const bool supported = input == ge::DT_FLOAT16 ||
                        (output == ge::DT_FLOAT && (!present || threshold == ge::DT_FLOAT));
                    assert(Supports(op, input, threshold, output, present) == supported);
                    if (!supported) continue;
                    gert::TilingContext ctx;
                    ctx.inputs[0].dtype = ctx.inputs[1].dtype = input;
                    ctx.inputs[2].dtype = threshold;
                    ctx.output.dtype = output;
                    ctx.present = present;
                    assert(optiling::TilingFunc(&ctx) == ge::GRAPH_SUCCESS);
                    assert(ctx.workspace == 0 && ctx.blocks == 1);
                    const auto* data = ctx.GetTilingData<SafetyBeltViolationGeometryTilingData>();
                    assert(data->batchSize == 32 && data->rowsPerCore == 64);
                    assert(data->hasThreshold == static_cast<uint32_t>(present));
                    assert(data->defaultThreshold == 0.0f);
#ifdef CHECK_STORAGE_FLAGS
                    assert(data->thresholdIsHalf == static_cast<uint32_t>(present && threshold == ge::DT_FLOAT16));
                    assert(data->outputIsHalf == static_cast<uint32_t>(output == ge::DT_FLOAT16));
#endif
                    ++cases;
                }
            }
        }
    }
    gert::InferShapeContext shape;
    assert(ge::InferShape(&shape) == ge::GRAPH_SUCCESS);
    assert(shape.output.dims == std::vector<int64_t>({32, 1}));
    gert::InferDataTypeContext dtype;
    assert(ge::InferDataType(&dtype) == ge::GRAPH_SUCCESS);
    assert(dtype.output == ge::DT_FLOAT); // Standard inferred output remains FP32.
    uint32_t rejected = 0;
    const auto Reject = [&rejected](gert::TilingContext ctx) {
        assert(optiling::TilingFunc(&ctx) == ge::GRAPH_FAILED);
        ++rejected;
    };
    for (int64_t batch : {0, 4097}) {
        gert::TilingContext ctx;
        ctx.shapes[0].shape.dims[0] = ctx.shapes[1].shape.dims[0] = batch;
        Reject(ctx);
    }
    for (size_t input = 0; input < 2; ++input) {
        gert::TilingContext ctx;
        ctx.shapes[input].shape.dims = {32};
        Reject(ctx);
        ctx = {};
        ctx.shapes[input].shape.dims[1] = 3;
        Reject(ctx);
    }
    {
        gert::TilingContext ctx;
        ctx.shapes[1].shape.dims[0] = 31;
        Reject(ctx);
        ctx = {};
        ctx.inputs[1].dtype = ge::DT_FLOAT16;
        Reject(ctx);
        ctx = {};
        ctx.inputs[0].dtype = ctx.inputs[1].dtype = ge::DT_INT32;
        Reject(ctx);
        ctx = {};
        ctx.output.dtype = ge::DT_FLOAT16;
        Reject(ctx);
        ctx = {};
        ctx.inputs[2].dtype = ge::DT_INT32;
        Reject(ctx);
        ctx = {};
        ctx.shapes[2].shape.dims = {};
        Reject(ctx);
        ctx = {};
        ctx.shapes[2].shape.dims = {2};
        Reject(ctx);
        ctx = {};
        ctx.cores = 0;
        Reject(ctx);
    }
    for (float ratio : {0.0f, -0.5f, 1.01f, std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
        gert::TilingContext ctx;
        ctx.attrs.ratio = ratio;
        Reject(ctx);
    }
    for (uint32_t batch : {1U, 33U, 512U, 1024U, 4096U}) {
        for (uint32_t cores : {1U, 8U, 48U}) {
            gert::TilingContext ctx;
            ctx.present = false;
            // An absent optional input's metadata must not affect acceptance.
            ctx.inputs[2].dtype = ge::DT_INT32;
            ctx.shapes[2].shape.dims = {2, 2};
            ctx.shapes[0].shape.dims[0] = ctx.shapes[1].shape.dims[0] = batch;
            ctx.cores = cores;
            assert(optiling::TilingFunc(&ctx) == ge::GRAPH_SUCCESS);
            const auto* t = ctx.GetTilingData<SafetyBeltViolationGeometryTilingData>();
            assert(t->batchSize == batch && t->hasThreshold == 0 && t->defaultThreshold == 0);
            assert(ctx.blocks >= 1 && ctx.blocks <= cores);
            assert(t->rowsPerCore * ctx.blocks >= batch);
            assert(t->rowsPerCore * (ctx.blocks - 1) < batch);
            ++cases;
        }
    }
    std::cout << "Host registration/tiling profiles passed: " << cases << '\n';
    std::cout << "Invalid host metadata cases rejected: " << rejected << '\n';
}
