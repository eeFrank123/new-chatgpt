#pragma once
// Minimal, bounds-checked CPU model of the APIs used by this kernel.
// This is NOT CANN simulation and does not validate device compilation or timing.
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <array>
#include <utility>
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
#define KERNEL_TYPE_AIV_ONLY 1
#define KERNEL_TASK_TYPE_DEFAULT(type) static_assert((type) == KERNEL_TYPE_AIV_ONLY)

namespace AscendC {
enum class Pipe : size_t { S, V, MTE2, MTE3 };
inline constexpr size_t kPipes = 4;
inline Pipe currentPipe = Pipe::S;
inline std::array<uint64_t, kPipes> epochs{};
inline std::array<std::array<uint64_t, kPipes>, kPipes> visible{};
struct PipeScope {
    Pipe previous;
    explicit PipeScope(Pipe pipe) : previous(currentPipe) {
        currentPipe = pipe;
        ++epochs[static_cast<size_t>(pipe)];
    }
    ~PipeScope() { currentPipe = previous; }
};
struct GlobalAllocation { uintptr_t begin, end; };
inline std::vector<GlobalAllocation> globalAllocations;
inline void RegisterGlobal(const void* p, size_t n) {
    const auto begin = reinterpret_cast<uintptr_t>(p);
    assert(p && n > 0);
    globalAllocations.push_back({begin, begin + n});
}
inline void CheckGlobal(const void* p, size_t n) {
    const auto begin = reinterpret_cast<uintptr_t>(p);
    for (const auto& a : globalAllocations)
        if (begin >= a.begin && begin + n <= a.end) return;
    assert(false && "GM tensor exceeds the registered physical allocation");
}
inline uint32_t blockIndex = 0;
inline uint32_t thresholdReads = 0;
inline uint32_t thresholdCopies = 0;
inline const void* thresholdAddress = nullptr;
inline const void* outputAddress = nullptr;
inline std::vector<uint32_t> outputWrites;
inline uint32_t GetBlockIdx() { return blockIndex; }
enum class QuePosition { VECIN, VECOUT };
enum class TPosition { VECIN, VECCALC, VECOUT };
enum class HardEvent { V_MTE2, MTE2_V, V_MTE3, MTE2_S, S_V };
enum class RoundMode { CAST_NONE };
enum class CMPMODE { GT };
enum class SELMODE { VSEL_TENSOR_SCALAR_MODE };
template<int P> inline void PipeBarrier() { static_assert(P == PIPE_V); }
inline std::array<bool, 5> eventFlags{};
inline std::array<uint64_t, 5> eventEpochs{};
inline std::pair<Pipe, Pipe> EventPipes(HardEvent e) {
    switch (e) {
        case HardEvent::V_MTE2: return {Pipe::V, Pipe::MTE2};
        case HardEvent::MTE2_V: return {Pipe::MTE2, Pipe::V};
        case HardEvent::V_MTE3: return {Pipe::V, Pipe::MTE3};
        case HardEvent::MTE2_S: return {Pipe::MTE2, Pipe::S};
        case HardEvent::S_V: return {Pipe::S, Pipe::V};
    }
    assert(false); return {Pipe::S, Pipe::S};
}
template<HardEvent E> inline void SetFlag(event_t) {
    assert(!eventFlags[static_cast<size_t>(E)]);
    eventFlags[static_cast<size_t>(E)] = true;
    eventEpochs[static_cast<size_t>(E)] = epochs[static_cast<size_t>(EventPipes(E).first)];
}
template<HardEvent E> inline void WaitFlag(event_t) {
    assert(eventFlags[static_cast<size_t>(E)]);
    eventFlags[static_cast<size_t>(E)] = false;
    const auto pipes = EventPipes(E);
    visible[static_cast<size_t>(pipes.second)][static_cast<size_t>(pipes.first)] =
        eventEpochs[static_cast<size_t>(E)];
}
inline void ResetPipelines() {
    currentPipe = Pipe::S; epochs.fill(0); eventFlags.fill(false); eventEpochs.fill(0);
    for (auto& row : visible) row.fill(0);
}

struct Memory {
    explicit Memory(size_t n, TPosition p)
        : bytes(n), initialized(n, false), writer(n), writtenAt(n), position(p) {}
    std::vector<uint8_t> bytes;
    std::vector<bool> initialized;
    std::vector<Pipe> writer;
    std::vector<uint64_t> writtenAt;
    TPosition position;
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
        for (size_t j = 0; j < sizeof(T); ++j) {
            assert(memory->initialized[p + j] && "read of uninitialized UB byte");
            const auto from = memory->writer[p + j];
            assert((from == currentPipe ||
                    visible[static_cast<size_t>(currentPipe)][static_cast<size_t>(from)] >= memory->writtenAt[p + j]) &&
                   "missing cross-pipeline producer/consumer synchronization");
        }
        T value;
        std::memcpy(&value, memory->bytes.data() + p, sizeof(T));
        return value;
    }
    void SetValue(size_t i, T value) const {
        Check(i);
        const size_t p = offset + i * sizeof(T);
        std::memcpy(memory->bytes.data() + p, &value, sizeof(T));
        for (size_t j = 0; j < sizeof(T); ++j) {
            memory->initialized[p + j] = true;
            memory->writer[p + j] = currentPipe;
            memory->writtenAt[p + j] = epochs[static_cast<size_t>(currentPipe)];
        }
    }
    LocalTensor operator[](size_t i) const {
        Check(i, 0);
        return {memory, offset + i * sizeof(T)};
    }
    template<class S> LocalTensor<S> ReinterpretCast() const {
        assert(offset % alignof(S) == 0);
        return {memory, offset};
    }
    void Aligned() const { assert(offset % 32 == 0); }
};

template<class T> struct GlobalTensor {
    T* data = nullptr;
    size_t size = 0;
    void SetGlobalBuffer(T* p, size_t n) {
        CheckGlobal(p, n * sizeof(T)); data = p; size = n;
    }
    T GetValue(size_t i) const {
        assert(data && i < size);
        if (data == thresholdAddress) ++thresholdReads;
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
        assert(buffers == 1);
        q.memory = std::make_shared<Memory>(n, P == QuePosition::VECIN ? TPosition::VECIN : TPosition::VECOUT);
    }
    template<TPosition P> void InitBuffer(TBuf<P>& b, size_t n) {
        b.memory = std::make_shared<Memory>(n, P);
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
struct BrcbRepeatParams {
    uint16_t dstBlkStride, dstRepStride;
};
struct BinaryRepeatParams {
    uint8_t dstBlkStride, src0BlkStride, src1BlkStride;
    uint8_t dstRepStride, src0RepStride, src1RepStride;
};
template<class T> void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src,
                                  DataCopyExtParams copy, DataCopyPadExtParams<T> pad) {
    PipeScope scope(Pipe::MTE2);
    assert(copy.blockCount == 1 && copy.srcStride == 0 && copy.dstStride == 0);
    assert(pad.leftPadding == 0);
    assert(pad.rightPadding <= 32 / sizeof(T));
    assert(copy.blockLen % sizeof(T) == 0);
    dst.Aligned();
    assert(dst.memory->position == TPosition::VECIN);
    const size_t n = copy.blockLen / sizeof(T);
    assert(n <= src.size);
    if (src.data == thresholdAddress) ++thresholdCopies;
    for (size_t i = 0; i < n; ++i) dst.SetValue(i, src.data[i]);
    // Both explicit paddings zero: CANN uses the first source element for
    // implicit dummy bytes. Explicit right padding uses paddingValue instead.
    const size_t totalBytes = copy.blockLen + pad.rightPadding * sizeof(T);
    const size_t padded = (totalBytes + 31) / 32 * 32 / sizeof(T);
    for (size_t i = n; i < padded; ++i)
        dst.SetValue(i, pad.isPad && pad.rightPadding ? pad.paddingValue : src.data[0]);
}
template<class T> void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src,
                                  DataCopyExtParams copy) {
    PipeScope scope(Pipe::MTE3);
    assert(copy.blockCount == 1 && copy.srcStride == 0 && copy.dstStride == 0);
    assert(copy.blockLen % sizeof(T) == 0);
    src.Aligned();
    assert(src.memory->position == TPosition::VECOUT);
    const size_t n = copy.blockLen / sizeof(T);
    assert(n <= dst.size);
    for (size_t i = 0; i < n; ++i) {
        dst.data[i] = src.GetValue(i);
        if (outputAddress) {
            const auto first = (reinterpret_cast<uintptr_t>(dst.data) -
                                reinterpret_cast<uintptr_t>(outputAddress)) / sizeof(T);
            assert(first + i < outputWrites.size());
            ++outputWrites[first + i];
        }
    }
}
template<class T> void Duplicate(LocalTensor<T> dst, T x, uint32_t n) {
    PipeScope scope(Pipe::V);
    dst.Aligned();
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, x);
}
template<class T> void GatherMask(LocalTensor<T> dst, LocalTensor<T> src,
                                  uint8_t pattern, bool counter, uint32_t mask,
                                  GatherMaskParams params, uint64_t& reserved) {
    PipeScope scope(Pipe::V);
    assert(!counter && mask == 0 && params.src0BlockStride == 1);
    assert(params.repeatTimes <= 255);
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
    PipeScope scope(Pipe::V);
    dst.Aligned(); src.Aligned();
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, static_cast<T>(src.GetValue(i)));
}
inline void Sub(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t n) {
    PipeScope scope(Pipe::V);
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, a.GetValue(i) - b.GetValue(i));
}
inline void Add(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t n) {
    PipeScope scope(Pipe::V);
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, a.GetValue(i) + b.GetValue(i));
}
inline void Add(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b,
                uint64_t mask, uint8_t repeats, BinaryRepeatParams p) {
    PipeScope scope(Pipe::V);
    assert(mask >= 1 && mask <= 64);
    dst.Aligned(); a.Aligned(); b.Aligned();
    for (uint32_t r = 0; r < repeats; ++r) {
        for (uint32_t i = 0; i < mask; ++i) {
            const uint32_t block = i / 8, lane = i % 8;
            const size_t d = r * p.dstRepStride * 8 + block * p.dstBlkStride * 8 + lane;
            const size_t x = r * p.src0RepStride * 8 + block * p.src0BlkStride * 8 + lane;
            const size_t y = r * p.src1RepStride * 8 + block * p.src1BlkStride * 8 + lane;
            dst.SetValue(d, a.GetValue(x) + b.GetValue(y));
        }
    }
}
inline void Brcb(LocalTensor<float> dst, LocalTensor<float> src,
                 uint8_t repeats, BrcbRepeatParams p) {
    PipeScope scope(Pipe::V);
    dst.Aligned(); src.Aligned();
    for (uint32_t r = 0; r < repeats; ++r) {
        for (uint32_t i = 0; i < 8; ++i) {
            const float x = src.GetValue(r * 8 + i);
            for (uint32_t j = 0; j < 8; ++j)
                dst.SetValue(r * p.dstRepStride * 8 + i * p.dstBlkStride * 8 + j, x);
        }
    }
}
inline void Muls(LocalTensor<float> dst, LocalTensor<float> a, float b, uint32_t n) {
    PipeScope scope(Pipe::V);
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, a.GetValue(i) * b);
}
inline void Adds(LocalTensor<float> dst, LocalTensor<float> a, float b, uint32_t n) {
    PipeScope scope(Pipe::V);
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, a.GetValue(i) + b);
}
inline void Compare(LocalTensor<uint8_t> dst, LocalTensor<float> a,
                    LocalTensor<float> b, CMPMODE, uint32_t n) {
    PipeScope scope(Pipe::V);
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
    PipeScope scope(Pipe::V);
    dst.Aligned(); mask.Aligned(); one.Aligned();
    for (uint32_t i = 0; i < n; ++i) {
        dst.SetValue(i, ((mask.GetValue(i / 8) >> (i % 8)) & 1U) ? one.GetValue(i) : zero);
    }
}
}  // namespace AscendC
