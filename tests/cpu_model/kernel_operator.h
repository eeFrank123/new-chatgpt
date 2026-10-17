#pragma once
// Minimal, bounds-checked CPU model of the APIs used by this kernel.
// This is NOT CANN simulation and does not validate device compilation or timing.
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <array>
#include <utility>
#include <type_traits>
#include <vector>

#define __aicore__
#define __global__
#define __gm__
#define REGISTER_TILING_DEFAULT(...)
#define GET_TILING_DATA_WITH_STRUCT(Type, name, address) \
    Type name; assert((address) != nullptr); ++AscendC::tilingLoads; \
    std::memcpy(&name, address, sizeof(Type))
using GM_ADDR = uint8_t*;
using half = _Float16;
using event_t = int;
constexpr int PIPE_V = 0;
constexpr int PIPE_ALL = 1;
#define KERNEL_TYPE_AIV_ONLY 1
#define KERNEL_TASK_TYPE_DEFAULT(type) static_assert((type) == KERNEL_TYPE_AIV_ONLY)

namespace AscendC {
enum class Pipe : size_t { S, V, MTE2, MTE3 };
inline constexpr size_t kPipes = 4;
inline Pipe currentPipe = Pipe::S;
inline std::array<uint64_t, kPipes> epochs{};
inline std::array<std::array<uint64_t, kPipes>, kPipes> visible{};
// Vector instructions are asynchronous even within PIPE_V. A completed
// barrier (or a waited outgoing event) makes earlier vector UB writes safe
// for a later vector read. This is a dependency check, not a timing model.
inline uint64_t completedVectorEpoch = 0;
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
inline uint32_t ratioCopies = 0;
inline uint32_t tilingLoads = 0;
inline const void* thresholdAddress = nullptr;
inline const void* ratioAddress = nullptr;
inline const void* outputAddress = nullptr;
inline std::vector<uint32_t> outputWrites;
inline uint32_t GetBlockIdx() { return blockIndex; }
enum class QuePosition { VECIN, VECOUT };
enum class TPosition { VECIN, VECCALC, VECOUT };
enum class HardEvent { V_MTE2, MTE2_V, V_MTE3, MTE2_S, S_V };
enum class RoundMode { CAST_NONE };
enum class CMPMODE { GT };
enum class SELMODE { VSEL_CMPMASK_SPR, VSEL_TENSOR_SCALAR_MODE, VSEL_TENSOR_TENSOR_MODE };
enum class MaskMode { NORMAL, COUNTER };
inline constexpr uint64_t MASK_PLACEHOLDER = 0;
inline uint64_t unifiedVectorMask = 64;
inline bool unifiedMaskEnabled = false;
inline bool vectorCounterMode = false;
inline uint64_t registerCompareMask = 0;
inline std::array<uint8_t, 16> compareRegisterBytes{};
inline bool registerCompareMaskInitialized = false;
inline uint64_t compareRegisterWrittenAt = 0;
inline void SetMaskNorm() { vectorCounterMode = false; }
inline void SetMaskCount() { vectorCounterMode = true; }
template<class T, MaskMode M = MaskMode::NORMAL> inline void SetVectorMask(uint64_t n) {
    if constexpr (M == MaskMode::NORMAL) assert(n >= 1 && n <= 256 / sizeof(T));
    else assert(n >= 1 && n <= UINT32_MAX);
    unifiedVectorMask = n;
    unifiedMaskEnabled = true;
}
// CANN 8.5 ResetMask restores the vector mask bits only. The independent
// counter/normal control bit requires SetMaskNorm after counter-mode work.
inline void ResetMask() { unifiedVectorMask = 64; unifiedMaskEnabled = false; }
template<int P> inline void PipeBarrier() {
    static_assert(P == PIPE_V || P == PIPE_ALL);
    completedVectorEpoch = epochs[static_cast<size_t>(Pipe::V)];
    if constexpr (P == PIPE_ALL) {
        for (auto& row : visible)
            for (size_t from = 0; from < kPipes; ++from) row[from] = epochs[from];
    }
}
inline constexpr size_t kEventIds = 8;
inline std::array<bool, 5 * kEventIds> eventFlags{};
inline std::array<uint64_t, 5 * kEventIds> eventEpochs{};
inline size_t EventSlot(HardEvent event, event_t id) {
    assert(id >= 0 && id < static_cast<event_t>(kEventIds));
    return static_cast<size_t>(event) * kEventIds + static_cast<size_t>(id);
}
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
template<HardEvent E> inline void SetFlag(event_t id) {
    const size_t slot = EventSlot(E, id);
    assert(!eventFlags[slot]);
    eventFlags[slot] = true;
    eventEpochs[slot] = epochs[static_cast<size_t>(EventPipes(E).first)];
}
template<HardEvent E> inline void WaitFlag(event_t id) {
    const size_t slot = EventSlot(E, id);
    assert(eventFlags[slot]);
    eventFlags[slot] = false;
    const auto pipes = EventPipes(E);
    auto& seen = visible[static_cast<size_t>(pipes.second)][static_cast<size_t>(pipes.first)];
    if (seen < eventEpochs[slot]) seen = eventEpochs[slot];
    if (pipes.first == Pipe::V && completedVectorEpoch < eventEpochs[slot])
        completedVectorEpoch = eventEpochs[slot];
}
inline void ResetRawLocalMemory();
inline void ResetPipelines() {
    currentPipe = Pipe::S; epochs.fill(0); eventFlags.fill(false); eventEpochs.fill(0);
    for (auto& row : visible) row.fill(0);
    completedVectorEpoch = 0;
    ResetRawLocalMemory();
    ResetMask();
    SetMaskNorm();
    registerCompareMaskInitialized = false;
    compareRegisterWrittenAt = 0;
}

struct Memory {
    explicit Memory(size_t n, TPosition p)
        : bytes(n), initialized(n, false), writer(n), writtenAt(n), position(p) {}
    std::vector<uint8_t> bytes;
    std::vector<bool> initialized;
    std::vector<Pipe> writer;
    std::vector<uint64_t> writtenAt;
    TPosition position;
    void Grow(size_t n) {
        if (n <= bytes.size()) return;
        bytes.resize(n); initialized.resize(n, false); writer.resize(n); writtenAt.resize(n);
    }
};

// Raw LocalTensor views address the same physical UB, including when their
// logical positions differ. Keep per-view bounds and positions on each tensor.
inline constexpr size_t kUnifiedBufferBytes = 192U * 1024U;
inline std::shared_ptr<Memory> rawLocalMemory;
inline bool socStateInitialized = false;
inline void ResetRawLocalMemory() { rawLocalMemory.reset(); socStateInitialized = false; }
inline void InitSocState() {
    ResetRawLocalMemory(); ResetMask(); SetMaskNorm();
    registerCompareMaskInitialized = false; compareRegisterWrittenAt = 0;
    socStateInitialized = true;
}

template<class T> struct LocalTensor {
    std::shared_ptr<Memory> memory;
    size_t offset = 0;
    size_t viewEnd = 0;
    TPosition viewPosition = TPosition::VECCALC;
    bool hasViewPosition = false;
    LocalTensor() = default;
    LocalTensor(std::shared_ptr<Memory> backing, size_t address)
        : memory(std::move(backing)), offset(address) {}
    LocalTensor(TPosition position, uint32_t address, uint32_t count)
        : offset(address), viewEnd(size_t(address) + size_t(count) * sizeof(T)),
          viewPosition(position), hasViewPosition(true) {
        assert(socStateInitialized && "raw UB tensors require InitSocState at kernel entry");
        assert(address % 32 == 0 && viewEnd <= kUnifiedBufferBytes);
        if (!rawLocalMemory)
            rawLocalMemory = std::make_shared<Memory>(0, TPosition::VECCALC);
        memory = rawLocalMemory;
        memory->Grow(viewEnd);
    }
    TPosition Position() const { return hasViewPosition ? viewPosition : memory->position; }
    void Check(size_t i, size_t count = 1) const {
        assert(memory && offset + (i + count) * sizeof(T) <= memory->bytes.size());
        assert(viewEnd == 0 || offset + (i + count) * sizeof(T) <= viewEnd);
    }
    T GetValue(size_t i) const {
        Check(i);
        const size_t p = offset + i * sizeof(T);
        for (size_t j = 0; j < sizeof(T); ++j) {
            assert(memory->initialized[p + j] && "read of uninitialized UB byte");
            const auto from = memory->writer[p + j];
            if (from == Pipe::V && currentPipe == Pipe::V)
                assert((memory->writtenAt[p + j] == epochs[static_cast<size_t>(Pipe::V)] ||
                        memory->writtenAt[p + j] <= completedVectorEpoch) &&
                       "missing PIPE_V barrier for a vector read-after-write dependency");
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
        auto result = *this;
        result.offset += i * sizeof(T);
        return result;
    }
    template<class S> LocalTensor<S> ReinterpretCast() const {
        assert(offset % alignof(S) == 0);
        LocalTensor<S> result(memory, offset);
        result.viewEnd = viewEnd;
        result.viewPosition = viewPosition;
        result.hasViewPosition = hasViewPosition;
        return result;
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
    uint32_t rsv;
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
    uint8_t dstBlkStride = 1, src0BlkStride = 1, src1BlkStride = 1;
    uint8_t dstRepStride = 8, src0RepStride = 8, src1RepStride = 8;
};
struct UnaryRepeatParams {
    uint8_t dstBlkStride = 1, srcBlkStride = 1, dstRepStride = 8, srcRepStride = 8;
};
template<class T> void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src,
                                  DataCopyExtParams copy, DataCopyPadExtParams<T> pad) {
    PipeScope scope(Pipe::MTE2);
    assert(copy.blockCount >= 1 && copy.blockCount <= 4095 && copy.blockLen > 0);
    assert(pad.leftPadding == 0);
    assert(pad.rightPadding <= 32 / sizeof(T));
    assert(copy.blockLen % sizeof(T) == 0);
    assert(copy.srcStride % sizeof(T) == 0);
    dst.Aligned();
    assert(dst.Position() == TPosition::VECIN);
    const size_t n = copy.blockLen / sizeof(T);
    if (src.data == thresholdAddress) ++thresholdCopies;
    if (src.data == ratioAddress) {
        assert(copy.blockCount == 1U && copy.blockLen == sizeof(float) &&
               "dynamic ratio DMA must read exactly its one float32 field");
        ++ratioCopies;
    }
    // DataCopyExtParams uses bytes for GM gaps and 32 B blocks for UB gaps.
    // CANN 8.5 kernel_check_data_copy_overflow.h also rounds EACH UB row up
    // independently; short GM rows must never cause an aligned GM over-read.
    const size_t totalBytes = copy.blockLen + pad.rightPadding * sizeof(T);
    const size_t padded = (totalBytes + 31) / 32 * 32 / sizeof(T);
    const size_t sourcePitch = n + copy.srcStride / sizeof(T);
    const size_t destinationPitch = padded + copy.dstStride * (32 / sizeof(T));
    for (size_t row = 0; row < copy.blockCount; ++row) {
        const size_t sourceStart = row * sourcePitch;
        const size_t destinationStart = row * destinationPitch;
        assert(sourceStart + n <= src.size);
        CheckGlobal(src.data + sourceStart, copy.blockLen);
        dst.Check(destinationStart, padded);
        for (size_t i = 0; i < n; ++i)
            dst.SetValue(destinationStart + i, src.data[sourceStart + i]);
        // With both explicit paddings zero, the implicit dummy bytes repeat
        // this row's FIRST input element, rather than the first entire input.
        for (size_t i = n; i < padded; ++i)
            dst.SetValue(destinationStart + i,
                         pad.isPad && pad.rightPadding ? pad.paddingValue : src.data[sourceStart]);
    }
}
template<class T> void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src,
                                  DataCopyExtParams copy) {
    PipeScope scope(Pipe::MTE3);
    assert(copy.blockCount >= 1 && copy.blockCount <= 4095 && copy.blockLen > 0);
    assert(copy.blockLen % sizeof(T) == 0);
    assert(copy.dstStride % sizeof(T) == 0);
    src.Aligned();
    assert(src.Position() == TPosition::VECOUT);
    const size_t n = copy.blockLen / sizeof(T);
    const size_t padded = (copy.blockLen + 31) / 32 * 32 / sizeof(T);
    const size_t sourcePitch = padded + copy.srcStride * (32 / sizeof(T));
    const size_t destinationPitch = n + copy.dstStride / sizeof(T);
    for (size_t row = 0; row < copy.blockCount; ++row) {
        const size_t sourceStart = row * sourcePitch;
        const size_t destinationStart = row * destinationPitch;
        assert(destinationStart + n <= dst.size);
        CheckGlobal(dst.data + destinationStart, copy.blockLen);
        src.Check(sourceStart, padded);
        for (size_t i = 0; i < n; ++i) {
            dst.data[destinationStart + i] = src.GetValue(sourceStart + i);
            if (outputAddress) {
                const auto first = (reinterpret_cast<uintptr_t>(dst.data) -
                                    reinterpret_cast<uintptr_t>(outputAddress)) / sizeof(T);
                assert(first + destinationStart + i < outputWrites.size());
                ++outputWrites[first + destinationStart + i];
            }
        }
    }
}
template<class T> void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t n) {
    assert(n > 0 && size_t(n) * sizeof(T) % 32 == 0);
    DataCopyPad(dst, src, {1U, n * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U},
                DataCopyPadExtParams<T>{false, 0U, 0U, T(0)});
}
template<class T> void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t n) {
    assert(n > 0 && size_t(n) * sizeof(T) % 32 == 0);
    DataCopyPad(dst, src, {1U, n * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U});
}
template<class T> void Duplicate(LocalTensor<T> dst, T x, uint32_t n) {
    PipeScope scope(Pipe::V);
    dst.Aligned();
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, x);
}
template<class T> uint64_t GatherMaskModel(LocalTensor<T> dst, LocalTensor<T> src,
                                         uint8_t pattern, bool counter, uint32_t mask,
                                         GatherMaskParams params) {
    PipeScope scope(Pipe::V);
    assert((counter || mask == 0) && params.src0BlockStride == 1);
    assert(params.repeatTimes <= 255);
    dst.Aligned(); src.Aligned();
    const uint8_t patterns[] = {0, 0x55, 0xaa, 0x11, 0x22, 0x44, 0x88, 0xff};
    assert(pattern >= 1 && pattern <= 7);
    uint64_t reserved = 0;
    if (counter) {
        assert(mask > 0 && params.src0RepeatStride == 8);
        for (uint32_t r = 0; r < params.repeatTimes; ++r)
            for (uint32_t i = 0; i < mask; ++i)
                if ((patterns[pattern] >> (i % 8)) & 1U)
                    dst.SetValue(reserved++, src.GetValue(r * params.src0RepeatStride * 32 / sizeof(T) + i));
        return reserved;
    }
    for (uint32_t r = 0; r < params.repeatTimes; ++r) {
        for (uint32_t i = 0; i < 256 / sizeof(T); ++i) {
            if ((patterns[pattern] >> (i % 8)) & 1U) {
                dst.SetValue(reserved++, src.GetValue(r * params.src0RepeatStride * 32 / sizeof(T) + i));
            }
        }
    }
    return reserved;
}
template<class T> void GatherMask(LocalTensor<T> dst, LocalTensor<T> src,
                                  uint8_t pattern, bool counter, uint32_t mask,
                                  GatherMaskParams params, uint64_t& reserved) {
    reserved = GatherMaskModel(dst, src, pattern, counter, mask, params);
    // The legacy overload sets its own mask mode and resets to normal after
    // reading the reserved count. It does not preserve a caller's counter mode.
    unifiedVectorMask = mask;
    unifiedMaskEnabled = true;
    SetMaskNorm();
}
#ifndef SBVG_CPU_MODEL_LEGACY_GATHER_ONLY
template<class T> void GatherMask(LocalTensor<T> dst, LocalTensor<T> src,
                                  uint8_t pattern, GatherMaskParams params) {
    if (vectorCounterMode) assert(unifiedMaskEnabled && "counter GatherMask needs a vector count");
    (void)GatherMaskModel(dst, src, pattern, vectorCounterMode,
                          vectorCounterMode ? static_cast<uint32_t>(unifiedVectorMask) : 0U,
                          params);
}
#endif
template<class T, class S> void Cast(LocalTensor<T> dst, LocalTensor<S> src,
                                    RoundMode, uint32_t n) {
    PipeScope scope(Pipe::V);
    dst.Aligned(); src.Aligned();
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, static_cast<T>(src.GetValue(i)));
}
template<class T, class S, bool isSetMask = true>
void Cast(LocalTensor<T> dst, LocalTensor<S> src, RoundMode,
          uint64_t mask, uint8_t repeats, UnaryRepeatParams p) {
    static_assert((std::is_same_v<T, float> && std::is_same_v<S, half>) ||
                  (std::is_same_v<T, half> && std::is_same_v<S, float>),
                  "This high-dimensional model covers half/float conversion only");
    PipeScope scope(Pipe::V);
    if constexpr (!isSetMask) {
        assert(unifiedMaskEnabled && mask == MASK_PLACEHOLDER);
        mask = unifiedVectorMask;
    }
    constexpr uint32_t dstBlockElements = 32 / sizeof(T);
    constexpr uint32_t srcBlockElements = 32 / sizeof(S);
    constexpr uint32_t repeatElements = 256 / (sizeof(T) > sizeof(S) ? sizeof(T) : sizeof(S));
    assert(mask >= 1 && (vectorCounterMode || mask <= repeatElements));
    assert(repeats >= 1);
    if (vectorCounterMode) assert(repeats == 1);
    const uint32_t iterations = vectorCounterMode ? (mask + repeatElements - 1U) / repeatElements : repeats;
    dst.Aligned(); src.Aligned();
    for (uint32_t row = 0; row < iterations; ++row) {
        const uint32_t active = vectorCounterMode
            ? static_cast<uint32_t>(mask - uint64_t(row) * repeatElements < repeatElements
                ? mask - uint64_t(row) * repeatElements : repeatElements) : mask;
        for (uint32_t i = 0; i < active; ++i) {
            const size_t d = (row * p.dstRepStride + (i / dstBlockElements) * p.dstBlkStride) *
                             dstBlockElements + i % dstBlockElements;
            const size_t s = (row * p.srcRepStride + (i / srcBlockElements) * p.srcBlkStride) *
                             srcBlockElements + i % srcBlockElements;
            dst.SetValue(d, static_cast<T>(src.GetValue(s)));
        }
    }
}
inline void Sub(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t n) {
    PipeScope scope(Pipe::V);
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, a.GetValue(i) - b.GetValue(i));
}
inline void Add(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t n) {
    PipeScope scope(Pipe::V);
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, a.GetValue(i) + b.GetValue(i));
}
template<class T, bool isSetMask = true, class Op>
inline void BinaryHighDim(LocalTensor<T> dst, LocalTensor<T> a, LocalTensor<T> b,
                          uint64_t mask, uint8_t repeats, BinaryRepeatParams p, Op op) {
    PipeScope scope(Pipe::V);
    if constexpr (!isSetMask) {
        assert(unifiedMaskEnabled && mask == MASK_PLACEHOLDER);
        mask = unifiedVectorMask;
    }
    const uint32_t blockElements = 32 / sizeof(T);
    const uint32_t repeatElements = 256 / sizeof(T);
    assert(mask >= 1 && (vectorCounterMode || mask <= repeatElements));
    if (vectorCounterMode) assert(repeats == 1);
    const uint32_t iterations = vectorCounterMode ? (mask + repeatElements - 1U) / repeatElements : repeats;
    dst.Aligned(); a.Aligned(); b.Aligned();
    for (uint32_t r = 0; r < iterations; ++r) {
        const uint32_t active = vectorCounterMode
            ? static_cast<uint32_t>(mask - uint64_t(r) * repeatElements < repeatElements
                ? mask - uint64_t(r) * repeatElements : repeatElements) : mask;
        for (uint32_t i = 0; i < active; ++i) {
            const uint32_t block = i / blockElements, lane = i % blockElements;
            const size_t d = (r * p.dstRepStride + block * p.dstBlkStride) * blockElements + lane;
            const size_t x = (r * p.src0RepStride + block * p.src0BlkStride) * blockElements + lane;
            const size_t y = (r * p.src1RepStride + block * p.src1BlkStride) * blockElements + lane;
            dst.SetValue(d, op(a.GetValue(x), b.GetValue(y)));
        }
    }
}
template<class T, bool isSetMask = true>
inline void Add(LocalTensor<T> dst, LocalTensor<T> a, LocalTensor<T> b,
                uint64_t mask, uint8_t repeats, BinaryRepeatParams p) {
    BinaryHighDim<T, isSetMask>(dst, a, b, mask, repeats, p,
                               [](T x, T y) { return T(x + y); });
}
template<class T, bool isSetMask = true>
inline void Sub(LocalTensor<T> dst, LocalTensor<T> a, LocalTensor<T> b,
                uint64_t mask, uint8_t repeats, BinaryRepeatParams p) {
    BinaryHighDim<T, isSetMask>(dst, a, b, mask, repeats, p,
                               [](T x, T y) { return T(x - y); });
}
template<class T, bool isSetMask = true>
inline void Mul(LocalTensor<T> dst, LocalTensor<T> a, LocalTensor<T> b,
                uint64_t mask, uint8_t repeats, BinaryRepeatParams p) {
    BinaryHighDim<T, isSetMask>(dst, a, b, mask, repeats, p,
                               [](T x, T y) { return T(x * y); });
}
template<class T, bool isSetMask = true, class Op>
inline void UnaryHighDim(LocalTensor<T> dst, LocalTensor<T> a, T scalar,
                         uint64_t mask, uint8_t repeats, UnaryRepeatParams p, Op op) {
    PipeScope scope(Pipe::V);
    if constexpr (!isSetMask) {
        assert(unifiedMaskEnabled && mask == MASK_PLACEHOLDER);
        mask = unifiedVectorMask;
    }
    const uint32_t blockElements = 32 / sizeof(T);
    const uint32_t repeatElements = 256 / sizeof(T);
    assert(mask >= 1 && (vectorCounterMode || mask <= repeatElements));
    if (vectorCounterMode) assert(repeats == 1);
    const uint32_t iterations = vectorCounterMode ? (mask + repeatElements - 1U) / repeatElements : repeats;
    dst.Aligned(); a.Aligned();
    for (uint32_t r = 0; r < iterations; ++r) {
        const uint32_t active = vectorCounterMode
            ? static_cast<uint32_t>(mask - uint64_t(r) * repeatElements < repeatElements
                ? mask - uint64_t(r) * repeatElements : repeatElements) : mask;
        for (uint32_t i = 0; i < active; ++i) {
            const uint32_t block = i / blockElements, lane = i % blockElements;
            const size_t d = (r * p.dstRepStride + block * p.dstBlkStride) * blockElements + lane;
            const size_t x = (r * p.srcRepStride + block * p.srcBlkStride) * blockElements + lane;
            dst.SetValue(d, op(a.GetValue(x), scalar));
        }
    }
}
template<class T, bool isSetMask = true>
inline void Muls(LocalTensor<T> dst, LocalTensor<T> a, T scalar,
                 uint64_t mask, uint8_t repeats, UnaryRepeatParams p) {
    UnaryHighDim<T, isSetMask>(dst, a, scalar, mask, repeats, p,
                              [](T x, T y) { return T(x * y); });
}
template<class T, bool isSetMask = true>
inline void Adds(LocalTensor<T> dst, LocalTensor<T> a, T scalar,
                 uint64_t mask, uint8_t repeats, UnaryRepeatParams p) {
    UnaryHighDim<T, isSetMask>(dst, a, scalar, mask, repeats, p,
                              [](T x, T y) { return T(x + y); });
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
template<class T, bool isSetMask = true>
inline void Compare(LocalTensor<T> a, LocalTensor<T> b, CMPMODE,
                    uint64_t mask, BinaryRepeatParams p) {
    PipeScope scope(Pipe::V);
    static_assert(sizeof(T) == sizeof(float));
    if constexpr (!isSetMask) {
        assert(mask == MASK_PLACEHOLDER && unifiedMaskEnabled && !vectorCounterMode);
        mask = unifiedVectorMask;
    }
    assert(mask >= 1 && mask <= 64);
    a.Aligned(); b.Aligned();
    registerCompareMask = 0;
    for (uint32_t i = 0; i < mask; ++i) {
        const uint32_t block = i / 8, lane = i % 8;
        const size_t x = block * p.src0BlkStride * 8 + lane;
        const size_t y = block * p.src1BlkStride * 8 + lane;
        if (a.GetValue(x) > b.GetValue(y)) registerCompareMask |= uint64_t{1} << i;
    }
    std::memcpy(compareRegisterBytes.data(), &registerCompareMask, sizeof(registerCompareMask));
    registerCompareMaskInitialized = true;
    compareRegisterWrittenAt = epochs[static_cast<size_t>(Pipe::V)];
}
template<class T> inline void SetCmpMask(LocalTensor<T> src) {
    PipeScope scope(Pipe::V);
    assert(src.offset % 16 == 0);
    const auto bytes = src.template ReinterpretCast<uint8_t>();
    for (size_t i = 0; i < compareRegisterBytes.size(); ++i)
        compareRegisterBytes[i] = bytes.GetValue(i);
    std::memcpy(&registerCompareMask, compareRegisterBytes.data(), sizeof(registerCompareMask));
    registerCompareMaskInitialized = true;
    compareRegisterWrittenAt = epochs[static_cast<size_t>(Pipe::V)];
}
template<class T, SELMODE Mode>
inline void Select(LocalTensor<T> dst, LocalTensor<T> whenTrue, LocalTensor<T> whenFalse,
                   uint8_t repeats, BinaryRepeatParams p) {
    PipeScope scope(Pipe::V);
    static_assert(Mode == SELMODE::VSEL_CMPMASK_SPR && sizeof(T) == sizeof(float));
    assert(registerCompareMaskInitialized && unifiedMaskEnabled);
    assert(compareRegisterWrittenAt <= completedVectorEpoch &&
           "missing PIPE_V barrier between register Compare/SetCmpMask and Select");
    assert(unifiedVectorMask >= 1 && unifiedVectorMask <= 64);
    dst.Aligned(); whenTrue.Aligned(); whenFalse.Aligned();
    // Mode 0 can broadcast a 32B input block while the destination aliases it.
    // Snapshot sources before writing so reads model one vector instruction.
    std::vector<T> trueValues, falseValues;
    for (uint32_t r = 0; r < repeats; ++r) {
        for (uint32_t i = 0; i < unifiedVectorMask; ++i) {
            const uint32_t block = i / 8, lane = i % 8;
            trueValues.push_back(whenTrue.GetValue((r * p.src0RepStride + block * p.src0BlkStride) * 8 + lane));
            falseValues.push_back(whenFalse.GetValue((r * p.src1RepStride + block * p.src1BlkStride) * 8 + lane));
        }
    }
    for (uint32_t r = 0; r < repeats; ++r) {
        for (uint32_t i = 0; i < unifiedVectorMask; ++i) {
            const uint32_t block = i / 8, lane = i % 8;
            const auto value = ((registerCompareMask >> i) & 1U)
                ? trueValues[r * unifiedVectorMask + i] : falseValues[r * unifiedVectorMask + i];
            dst.SetValue((r * p.dstRepStride + block * p.dstBlkStride) * 8 + lane, value);
        }
    }
}
inline void Select(LocalTensor<float> dst, LocalTensor<uint8_t> mask,
                   LocalTensor<float> one, float zero, SELMODE, uint32_t n) {
    // CANN 8.5's scalar-value overload duplicates its scalar and inserts two
    // PIPE_V barriers before vsel. Direct Select overloads do not do so.
    PipeBarrier<PIPE_V>();
    PipeScope scope(Pipe::V);
    dst.Aligned(); mask.Aligned(); one.Aligned();
    for (uint32_t i = 0; i < n; ++i) {
        dst.SetValue(i, ((mask.GetValue(i / 8) >> (i % 8)) & 1U) ? one.GetValue(i) : zero);
    }
}
// Public direct mode-1 Select: CMPMASK contains the scalar source bits, while
// the second vector source is the compressed per-element selection mask.
// Its stride unit is one group of (32 / sizeof(T)) selection bits, so the
// ordinary {src1BlkStride=1, src1RepStride=8} consumes consecutive mask bits.
template<class T, class U>
inline void Select(LocalTensor<T> dst, LocalTensor<U> mask, LocalTensor<T> whenTrue,
                   uint8_t repeats, BinaryRepeatParams p) {
    PipeScope scope(Pipe::V);
    static_assert(sizeof(T) == sizeof(float));
    assert(registerCompareMaskInitialized && "mode-1 Select requires SetCmpMask scalar setup");
    assert(compareRegisterWrittenAt <= completedVectorEpoch &&
           "missing PIPE_V barrier between SetCmpMask and direct Select");
    assert(unifiedMaskEnabled);
    assert(unifiedVectorMask >= 1 && (vectorCounterMode || unifiedVectorMask <= 64));
    if (vectorCounterMode) assert(repeats == 1);
    dst.Aligned(); mask.Aligned(); whenTrue.Aligned();
    T whenFalse;
    std::memcpy(&whenFalse, compareRegisterBytes.data(), sizeof(T));
    const auto maskBytes = mask.template ReinterpretCast<uint8_t>();
    std::vector<T> values;
    const uint32_t iterations = vectorCounterMode ? static_cast<uint32_t>((unifiedVectorMask + 63U) / 64U) : repeats;
    const auto activeCount = [&](uint32_t r) {
        return static_cast<uint32_t>(vectorCounterMode
            ? (unifiedVectorMask - uint64_t(r) * 64U < 64U ? unifiedVectorMask - uint64_t(r) * 64U : 64U)
            : unifiedVectorMask);
    };
    for (uint32_t r = 0; r < iterations; ++r) {
        for (uint32_t i = 0; i < activeCount(r); ++i) {
            const uint32_t block = i / 8, lane = i % 8;
            const size_t selectedByte = r * p.src1RepStride + block * p.src1BlkStride;
            const bool selected = ((maskBytes.GetValue(selectedByte) >> lane) & 1U) != 0;
            values.push_back(selected ? whenTrue.GetValue((r * p.src0RepStride + block * p.src0BlkStride) * 8 + lane)
                                      : whenFalse);
        }
    }
    size_t valueIndex = 0;
    for (uint32_t r = 0; r < iterations; ++r)
        for (uint32_t i = 0; i < activeCount(r); ++i) {
            const uint32_t block = i / 8, lane = i % 8;
            dst.SetValue((r * p.dstRepStride + block * p.dstBlkStride) * 8 + lane,
                         values[valueIndex++]);
        }
}
}  // namespace AscendC
