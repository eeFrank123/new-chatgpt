#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/safety_belt_violation_geometry_tiling.h"
#include "../op_kernel/tiling_key_safety_belt_violation_geometry.h"
#include "tiling_policy.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto* boxesShape = context->GetInputShape(0);
    const auto* anchorsShape = context->GetInputShape(1);
    const auto* boxesDesc = context->GetInputDesc(0);
    const auto* anchorsDesc = context->GetInputDesc(1);
    const auto* outputDesc = context->GetOutputDesc(0);
    const auto* attrs = context->GetAttrs();
    if (boxesShape == nullptr || anchorsShape == nullptr || boxesDesc == nullptr ||
        anchorsDesc == nullptr || outputDesc == nullptr || attrs == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& boxes = boxesShape->GetStorageShape();
    const auto& anchors = anchorsShape->GetStorageShape();
    if (boxes.GetDimNum() != 2U || anchors.GetDimNum() != 2U ||
        boxes.GetDim(1) != 4 || anchors.GetDim(1) != 2 ||
        boxes.GetDim(0) < 1 || boxes.GetDim(0) > 4096 ||
        boxes.GetDim(0) != anchors.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dtype = boxesDesc->GetDataType();
    if ((dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) ||
        anchorsDesc->GetDataType() != dtype) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType outputDtype = outputDesc->GetDataType();
    if (outputDtype != ge::DT_FLOAT &&
        !(dtype == ge::DT_FLOAT16 && outputDtype == ge::DT_FLOAT16)) {
        return ge::GRAPH_FAILED;
    }
    const float* ratio = attrs->GetFloat(0);
    if (ratio == nullptr || !(*ratio > 0.0f && *ratio <= 1.0f)) {
        return ge::GRAPH_FAILED;
    }

    // Inspect presence and metadata only. A device tensor's value is not
    // necessarily accessible during Host tiling without a value dependency.
    const auto* thresholdShape = context->GetOptionalInputShape(2);
    const auto* thresholdDesc = context->GetOptionalInputDesc(2);
    const bool hasThreshold = thresholdShape != nullptr;
    if (hasThreshold) {
        const auto& shape = thresholdShape->GetStorageShape();
        if (thresholdDesc == nullptr ||
            (thresholdDesc->GetDataType() != ge::DT_FLOAT &&
             !(dtype == ge::DT_FLOAT16 && thresholdDesc->GetDataType() == ge::DT_FLOAT16)) ||
            shape.GetDimNum() != 1U || shape.GetDim(0) != 1) {
            return ge::GRAPH_FAILED;
        }
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t coreCount = platform.GetCoreNumAiv();
    if (coreCount == 0U) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t batch = static_cast<uint32_t>(boxes.GetDim(0));
    const auto partition = safety_belt::MakePartition(batch, coreCount);
    auto* tiling = context->GetTilingData<SafetyBeltViolationGeometryTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->batchSize = batch;
    tiling->rowsPerCore = partition.rowsPerCore;
    tiling->hasThreshold = hasThreshold ? 1U : 0U;
    tiling->waistRatio = *ratio;
    tiling->defaultThreshold = 0.0f;
    tiling->thresholdIsHalf = hasThreshold && thresholdDesc->GetDataType() == ge::DT_FLOAT16 ? 1U : 0U;
    tiling->outputIsHalf = outputDtype == ge::DT_FLOAT16 ? 1U : 0U;
    context->GetRawTilingData()->SetDataSize(sizeof(SafetyBeltViolationGeometryTilingData));

    const uint32_t DT_PERSON_BOXES = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_PERSON_BOXES);
    context->SetBlockDim(partition.blockDim);
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    const auto* boxes = context->GetInputShape(0);
    const auto* anchors = context->GetInputShape(1);
    auto* output = context->GetOutputShape(0);
    if (boxes == nullptr || anchors == nullptr || output == nullptr ||
        boxes->GetDimNum() != 2U || anchors->GetDimNum() != 2U) {
        return GRAPH_FAILED;
    }
    const int64_t batch = boxes->GetDim(0);
    const int64_t anchorBatch = anchors->GetDim(0);
    // Allow symbolic dimensions during inference; concrete shapes are checked in tiling.
    if ((boxes->GetDim(1) >= 0 && boxes->GetDim(1) != 4) ||
        (anchors->GetDim(1) >= 0 && anchors->GetDim(1) != 2) ||
        (batch >= 0 && (batch < 1 || batch > 4096)) ||
        (batch >= 0 && anchorBatch >= 0 && batch != anchorBatch)) {
        return GRAPH_FAILED;
    }
    output->SetDimNum(2);
    output->SetDim(0, batch);
    output->SetDim(1, 1);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    return context->SetOutputDataType(0, DT_FLOAT);
}
}  // namespace ge

namespace ops {
class SafetyBeltViolationGeometry : public OpDef {
public:
    explicit SafetyBeltViolationGeometry(const char* name) : OpDef(name) {
        // Columns are complete IO profiles, not independent lists:
        //   inputs    threshold    output
        //   float     float        float     (problem contract)
        //   half      float        float     (problem contract)
        //   half      half         float     (template compatibility)
        //   half      float        half      (template compatibility)
        //   half      half         half      (original supplied template)
        // InferDataType still defaults to float32. Half output is used only when
        // an ACLNN caller explicitly supplies a half output tensor.
        this->Input("person_boxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("anchor_points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("threshold")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("violation_score")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("waist_ratio").AttrType(REQUIRED).Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(SafetyBeltViolationGeometry);
}  // namespace ops
