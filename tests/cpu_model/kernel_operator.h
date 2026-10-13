#pragma once
// Minimal, bounds-checked CPU model of the APIs used by this kernel.
// This is NOT CANN simulation and does not validate device compilation or timing.
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#define __aicore__
#define __global__
#define __gm__
#define REGISTER_TILING_DEFAULT(...)
#define GET_TILING_DATA_WITH_STRUCT(Type, name, address) \
    Type name; std::memcpy(&name, address, sizeof(Type))
using GM_ADDR = uint8_t*;
using half = _Float16;
using event_t = int;
constexpr int PIPE_V = 0;

namespace AscendC {
inline uint32_t blockIndex = 0;
inline uint32_t thresholdReads = 0;
inline uint32_t GetBlockIdx() { return blockIndex; }
enum class QuePosition { VECIN, VECOUT };
enum class TPosition { VECCALC };
enum class HardEvent { V_MTE2 };
enum class RoundMode { CAST_NONE };
enum class CMPMODE { GT };
enum class SELMODE { VSEL_TENSOR_SCALAR_MODE };
template<int> inline void PipeBarrier() {}
template<HardEvent> inline void SetFlag(event_t) {}
template<HardEvent> inline void WaitFlag(event_t) {}

struct Memory {
    explicit Memory(size_t n) : bytes(n), initialized(n, false) {}
    std::vector<uint8_t> bytes;
    std::vector<bool> initialized;
};

template<class T> struct LocalTensor {
    std::shared_ptr<Memory> memory;
    size_t offset = 0;
    void Check(size_t i, size_t count = 1) const {
        assert(memory && offset + (i + count) * sizeof(T) <= memory->bytes.size());
    }
    T GetValue(size_t i) const {
        Check(i);
        const size_t p = offset + i * sizeof(T);
        for (size_t j = 0; j < sizeof(T); ++j) assert(memory->initialized[p + j]);
        T value;
        std::memcpy(&value, memory->bytes.data() + p, sizeof(T));
        return value;
    }
    void SetValue(size_t i, T value) const {
        Check(i);
        const size_t p = offset + i * sizeof(T);
        std::memcpy(memory->bytes.data() + p, &value, sizeof(T));
        for (size_t j = 0; j < sizeof(T); ++j) memory->initialized[p + j] = true;
    }
    LocalTensor operator[](size_t i) const {
        Check(i, 0);
        return {memory, offset + i * sizeof(T)};
    }
    void Aligned() const { assert(offset % 32 == 0); }
};

template<class T> struct GlobalTensor {
    T* data = nullptr;
    size_t size = 0;
    void SetGlobalBuffer(T* p, size_t n) { assert(p != nullptr); data = p; size = n; }
    T GetValue(size_t i) const {
        assert(data && i < size);
        ++thresholdReads;
        return data[i];
    }
    GlobalTensor operator[](size_t i) const {
        assert(i <= size);
        return {data + i, size - i};
    }
};

template<QuePosition, int> struct TQue {
    std::shared_ptr<Memory> memory;
    int state = 0;
    template<class T> LocalTensor<T> AllocTensor() {
        assert(state == 0); state = 1; return {memory, 0};
    }
    template<class T> void EnQue(LocalTensor<T>) { assert(state == 1); state = 2; }
    template<class T> LocalTensor<T> DeQue() {
        assert(state == 2); state = 3; return {memory, 0};
    }
    template<class T> void FreeTensor(LocalTensor<T>) { assert(state == 3); state = 0; }
};
template<TPosition> struct TBuf {
    std::shared_ptr<Memory> memory;
    template<class T> LocalTensor<T> Get() { return {memory, 0}; }
};
struct TPipe {
    template<QuePosition P, int N> void InitBuffer(TQue<P, N>& q, int buffers, size_t n) {
        assert(buffers == 1); q.memory = std::make_shared<Memory>(n);
    }
    template<TPosition P> void InitBuffer(TBuf<P>& b, size_t n) {
        b.memory = std::make_shared<Memory>(n);
    }
    int FetchEventID(HardEvent) { return 0; }
};

struct DataCopyExtParams {
    uint16_t blockCount;
    uint32_t blockLen, srcStride, dstStride;
    uint16_t rsv;
};
template<class T> struct DataCopyPadExtParams {
    bool isPad;
    uint8_t leftPadding, rightPadding;
    T paddingValue;
};
struct GatherMaskParams {
    uint8_t src0BlockStride;
    uint16_t repeatTimes, src0RepeatStride;
    uint8_t src1RepeatStride;
};
template<class T> void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src,
                                  DataCopyExtParams copy, DataCopyPadExtParams<T> pad) {
    assert(copy.blockCount == 1 && copy.srcStride == 0 && copy.dstStride == 0);
    assert(pad.leftPadding == 0 && pad.rightPadding == 0);
    assert(copy.blockLen % sizeof(T) == 0);
    dst.Aligned();
    const size_t n = copy.blockLen / sizeof(T);
    assert(n <= src.size);
    for (size_t i = 0; i < n; ++i) dst.SetValue(i, src.data[i]);
    // CANN's implicit dummy bytes fill to a 32-byte boundary.
    const size_t padded = (copy.blockLen + 31) / 32 * 32 / sizeof(T);
    for (size_t i = n; i < padded; ++i) dst.SetValue(i, src.data[0]);
}
template<class T> void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src,
                                  DataCopyExtParams copy) {
    assert(copy.blockCount == 1 && copy.srcStride == 0 && copy.dstStride == 0);
    assert(copy.blockLen % sizeof(T) == 0);
    src.Aligned();
    const size_t n = copy.blockLen / sizeof(T);
    assert(n <= dst.size);
    for (size_t i = 0; i < n; ++i) dst.data[i] = src.GetValue(i);
}
template<class T> void Duplicate(LocalTensor<T> dst, T x, uint32_t n) {
    dst.Aligned();
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, x);
}
template<class T> void GatherMask(LocalTensor<T> dst, LocalTensor<T> src,
                                  uint8_t pattern, bool counter, uint32_t mask,
                                  GatherMaskParams params, uint64_t& reserved) {
    assert(!counter && mask == 0 && params.src0BlockStride == 1);
    dst.Aligned(); src.Aligned();
    const uint8_t patterns[] = {0, 0x55, 0xaa, 0x11, 0x22, 0x44, 0x88, 0xff};
    assert(pattern >= 1 && pattern <= 7);
    reserved = 0;
    for (uint32_t r = 0; r < params.repeatTimes; ++r) {
        for (uint32_t i = 0; i < 256 / sizeof(T); ++i) {
            if ((patterns[pattern] >> (i % 8)) & 1U) {
                dst.SetValue(reserved++, src.GetValue(r * params.src0RepeatStride * 32 / sizeof(T) + i));
            }
        }
    }
}
template<class T, class S> void Cast(LocalTensor<T> dst, LocalTensor<S> src,
                                    RoundMode, uint32_t n) {
    dst.Aligned(); src.Aligned();
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, static_cast<T>(src.GetValue(i)));
}
inline void Sub(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, a.GetValue(i) - b.GetValue(i));
}
inline void Add(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, a.GetValue(i) + b.GetValue(i));
}
inline void Muls(LocalTensor<float> dst, LocalTensor<float> a, float b, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, a.GetValue(i) * b);
}
inline void Adds(LocalTensor<float> dst, LocalTensor<float> a, float b, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, a.GetValue(i) + b);
}
inline void Compare(LocalTensor<uint8_t> dst, LocalTensor<float> a,
                    LocalTensor<float> b, CMPMODE, uint32_t n) {
    assert(n % 64 == 0);
    dst.Aligned(); a.Aligned(); b.Aligned();
    for (uint32_t i = 0; i < n; i += 8) {
        uint8_t bits = 0;
        for (uint32_t j = 0; j < 8; ++j) bits |= uint8_t(a.GetValue(i + j) > b.GetValue(i + j)) << j;
        dst.SetValue(i / 8, bits);
    }
}
inline void Select(LocalTensor<float> dst, LocalTensor<uint8_t> mask,
                   LocalTensor<float> one, float zero, SELMODE, uint32_t n) {
    dst.Aligned(); mask.Aligned(); one.Aligned();
    for (uint32_t i = 0; i < n; ++i) {
        dst.SetValue(i, ((mask.GetValue(i / 8) >> (i % 8)) & 1U) ? one.GetValue(i) : zero);
    }
}
}  // namespace AscendC
