#pragma once
// Standalone test doubles for Host metadata/registration, not a CANN runtime.
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>
namespace ge {
using graphStatus = int;
constexpr graphStatus GRAPH_SUCCESS = 0, GRAPH_FAILED = 1;
enum DataType { DT_FLOAT, DT_FLOAT16, DT_INT32 };
enum Format { FORMAT_ND };
}
namespace gert {
struct Shape {
    std::vector<int64_t> dims;
    size_t GetDimNum() const { return dims.size(); }
    int64_t GetDim(size_t i) const { return dims.at(i); }
    void SetDimNum(size_t n) { dims.resize(n); }
    void SetDim(size_t i, int64_t n) { dims.at(i) = n; }
};
struct StorageShape {
    Shape shape;
    const Shape& GetStorageShape() const { return shape; }
};
struct Desc {
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType GetDataType() const { return dtype; }
};
struct RuntimeAttrs {
    float ratio = 0.8f;
    const float* GetFloat(size_t i) const { assert(i == 0); return &ratio; }
};
struct TilingData {
    alignas(8) uint8_t bytes[128]{};
    size_t size = 0;
    void SetDataSize(size_t n) { assert(n <= sizeof(bytes)); size = n; }
};
struct TilingContext {
    StorageShape shapes[3]{{{{32, 4}}}, {{{32, 2}}}, {{{1}}}};
    Desc inputs[3], output;
    RuntimeAttrs attrs;
    bool present = true;
    uint32_t cores = 32, blocks = 0;
    uint64_t key = 0;
    TilingData tiling;
    size_t workspace = 99;
    const StorageShape* GetInputShape(size_t i) const { return &shapes[i]; }
    const Desc* GetInputDesc(size_t i) const { return &inputs[i]; }
    const Desc* GetOutputDesc(size_t i) const { assert(i == 0); return &output; }
    const StorageShape* GetOptionalInputShape(size_t i) const { return present ? &shapes[i] : nullptr; }
    const Desc* GetOptionalInputDesc(size_t i) const { return present ? &inputs[i] : nullptr; }
    const RuntimeAttrs* GetAttrs() const { return &attrs; }
    const void* GetPlatformInfo() const { return &cores; }
    template<class T> T* GetTilingData() { tiling.SetDataSize(sizeof(T)); return reinterpret_cast<T*>(tiling.bytes); }
    TilingData* GetRawTilingData() { return &tiling; }
    void SetBlockDim(uint32_t n) { blocks = n; }
    void SetTilingKey(uint64_t n) { key = n; }
    size_t* GetWorkspaceSizes(size_t n) { assert(n == 1); return &workspace; }
};
struct InferShapeContext {
    Shape inputs[2]{{{32, 4}}, {{32, 2}}}, output;
    const Shape* GetInputShape(size_t i) const { return &inputs[i]; }
    Shape* GetOutputShape(size_t i) { assert(i == 0); return &output; }
};
struct InferDataTypeContext {
    ge::DataType output = ge::DT_INT32;
    ge::graphStatus SetOutputDataType(size_t i, ge::DataType t) {
        assert(i == 0); output = t; return ge::GRAPH_SUCCESS;
    }
};
}
namespace ops {
enum ParamType { REQUIRED, OPTIONAL };
struct Param {
    std::vector<ge::DataType> types;
    std::vector<ge::Format> formats;
    Param& ParamType(ops::ParamType) { return *this; }
    Param& DataType(std::initializer_list<ge::DataType> x) { types = x; return *this; }
    Param& Format(std::initializer_list<ge::Format> x) { formats = x; return *this; }
};
struct AttrDef {
    AttrDef& AttrType(ops::ParamType) { return *this; }
    AttrDef& Float() { return *this; }
};
struct CoreDef {
    template<class T> CoreDef& SetTiling(T) { return *this; }
    CoreDef& AddConfig(const char*) { return *this; }
};
struct OpDef {
    explicit OpDef(const char*) {}
    std::map<std::string, Param> inputs, outputs;
    AttrDef attr;
    CoreDef core;
    Param& Input(const char* name) { return inputs[name]; }
    Param& Output(const char* name) { return outputs[name]; }
    AttrDef& Attr(const char*) { return attr; }
    template<class T> OpDef& SetInferShape(T) { return *this; }
    template<class T> OpDef& SetInferDataType(T) { return *this; }
    CoreDef& AICore() { return core; }
};
}
#define OP_ADD(...)
