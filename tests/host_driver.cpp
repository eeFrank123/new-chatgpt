#include <iostream>
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
    std::cout << "Host registration/tiling profiles passed: " << cases << '\n';
}
