#include <iostream>
#include <cmath>
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
                    assert(data->defaultThreshold == 0.0f);
                    const uint32_t expectedMode = static_cast<uint32_t>(present) |
                        (static_cast<uint32_t>(present && threshold == ge::DT_FLOAT16) << 1U) |
                        (static_cast<uint32_t>(output == ge::DT_FLOAT16) << 2U);
                    assert(ctx.key == (static_cast<uint64_t>(input) |
                                       (static_cast<uint64_t>(expectedMode) << 8U) |
                                       (4ULL << 16U)));
                    assert(ctx.GetRawTilingData()->size == 16U);
                    ++cases;
                }
            }
        }
    }
    // Every specialized key still carries the complete general tiling data;
    // only the selected kernel may eliminate unused metadata reads.
    uint32_t staticCases = 0, genericCases = 0;
    const uint32_t staticBatches[] = {1U, 32U, 33U, 256U, 1024U, 4096U};
    const uint32_t staticCoreRows[] = {64U, 64U, 64U, 256U, 512U, 512U};
    for (auto input : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        for (uint32_t mode : {0U, 1U, 3U, 4U, 5U, 7U}) {
            if (input == ge::DT_FLOAT && mode > 1U) continue;
            for (uint32_t shapeIndex = 0; shapeIndex < 6U; ++shapeIndex) {
                const uint32_t batch = staticBatches[shapeIndex];
                for (uint32_t ratioIndex = 0; ratioIndex < 3U; ++ratioIndex) {
                    gert::TilingContext ctx;
                    ctx.inputs[0].dtype = ctx.inputs[1].dtype = input;
                    ctx.inputs[2].dtype = (mode & 2U) ? ge::DT_FLOAT16 : ge::DT_FLOAT;
                    ctx.output.dtype = (mode & 4U) ? ge::DT_FLOAT16 : ge::DT_FLOAT;
                    ctx.present = (mode & 1U) != 0U;
                    ctx.shapes[0].shape.dims[0] = ctx.shapes[1].shape.dims[0] = batch;
                    ctx.attrs.ratio = ratioIndex == 0U ? 0.8f : ratioIndex == 1U ? 0.9f : 1.0f;
                    assert(optiling::TilingFunc(&ctx) == ge::GRAPH_SUCCESS);
                    const auto* t = ctx.GetTilingData<SafetyBeltViolationGeometryTilingData>();
                    const bool matchingPartition = t->rowsPerCore == staticCoreRows[shapeIndex];
                    const uint32_t staticCase = matchingPartition ? 1U + shapeIndex * 3U + ratioIndex : 0U;
                    assert(ctx.key == (static_cast<uint64_t>(input) |
                                       (static_cast<uint64_t>(mode) << 8U) |
                                       (static_cast<uint64_t>(staticCase) << 16U)));
                    assert(t->batchSize == batch);
                    assert(t->waistRatio == ctx.attrs.ratio && t->defaultThreshold == 0.0f);
                    assert(ctx.GetRawTilingData()->size == 16U);
                    assert(ctx.blocks == safety_belt::CeilDiv(batch, t->rowsPerCore));
                    if (matchingPartition) ++staticCases;
                    else ++genericCases;
                }
            }
            // Adjacent float32 values must not be rounded onto a static ratio.
            for (uint32_t batch : {1U, 32U, 33U, 2U, 31U, 34U, 64U, 256U, 1024U, 4096U}) {
                for (float ratio : {0.8f, 0.9f, 1.0f, 0.5f, 0.33333334f,
                                    std::nextafter(0.8f, 0.0f), std::nextafter(0.8f, 1.0f),
                                    std::nextafter(0.9f, 0.0f), std::nextafter(0.9f, 1.0f),
                                    std::nextafter(1.0f, 0.0f)}) {
                    if ((batch == 1U || batch == 32U || batch == 33U ||
                         batch == 256U || batch == 1024U || batch == 4096U) &&
                        (ratio == 0.8f || ratio == 0.9f || ratio == 1.0f)) continue;
                    gert::TilingContext ctx;
                    ctx.inputs[0].dtype = ctx.inputs[1].dtype = input;
                    ctx.inputs[2].dtype = (mode & 2U) ? ge::DT_FLOAT16 : ge::DT_FLOAT;
                    ctx.output.dtype = (mode & 4U) ? ge::DT_FLOAT16 : ge::DT_FLOAT;
                    ctx.present = (mode & 1U) != 0U;
                    ctx.shapes[0].shape.dims[0] = ctx.shapes[1].shape.dims[0] = batch;
                    ctx.attrs.ratio = ratio;
                    assert(optiling::TilingFunc(&ctx) == ge::GRAPH_SUCCESS);
                    assert(ctx.key == (static_cast<uint64_t>(input) |
                                       (static_cast<uint64_t>(mode) << 8U)));
                    const auto* t = ctx.GetTilingData<SafetyBeltViolationGeometryTilingData>();
                    assert(t->batchSize == batch && t->waistRatio == ratio);
                    ++genericCases;
                }
            }
            // If fewer vector cores change the real partition, a large shape
            // must use the general kernel instead of launching a static one
            // with an incorrect rows-per-core constant.
            for (uint32_t batch : {1024U, 4096U}) {
                for (uint32_t cores : {1U, 2U, 3U, 4U, 7U, 8U}) {
                    gert::TilingContext ctx;
                    ctx.inputs[0].dtype = ctx.inputs[1].dtype = input;
                    ctx.inputs[2].dtype = (mode & 2U) ? ge::DT_FLOAT16 : ge::DT_FLOAT;
                    ctx.output.dtype = (mode & 4U) ? ge::DT_FLOAT16 : ge::DT_FLOAT;
                    ctx.present = (mode & 1U) != 0U;
                    ctx.shapes[0].shape.dims[0] = ctx.shapes[1].shape.dims[0] = batch;
                    ctx.cores = cores;
                    ctx.attrs.ratio = 0.9f;
                    assert(optiling::TilingFunc(&ctx) == ge::GRAPH_SUCCESS);
                    const auto* t = ctx.GetTilingData<SafetyBeltViolationGeometryTilingData>();
                    const uint32_t staticCase = t->rowsPerCore == 512U ? (batch == 1024U ? 14U : 17U) : 0U;
                    assert(ctx.key == (static_cast<uint64_t>(input) |
                                       (static_cast<uint64_t>(mode) << 8U) |
                                       (static_cast<uint64_t>(staticCase) << 16U)));
                    assert(ctx.blocks <= cores && ctx.blocks * t->rowsPerCore >= batch);
                    if (staticCase) ++staticCases;
                    else ++genericCases;
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
            assert(t->batchSize == batch && t->defaultThreshold == 0);
            assert(ctx.blocks >= 1 && ctx.blocks <= cores);
            assert(t->rowsPerCore * ctx.blocks >= batch);
            assert(t->rowsPerCore * (ctx.blocks - 1) < batch);
            ++cases;
        }
    }
    std::cout << "Host registration/tiling profiles passed: " << cases << '\n';
    std::cout << "Static metadata dispatch cases passed: " << staticCases << '\n';
    std::cout << "Generic metadata fallback cases passed: " << genericCases << '\n';
    std::cout << "Invalid host metadata cases rejected: " << rejected << '\n';
}
