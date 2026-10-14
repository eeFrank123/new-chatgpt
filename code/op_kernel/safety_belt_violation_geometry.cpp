/**
 * 安全带低挂高用几何判定融合算子 —— Kernel 侧（瘦身版 v3）
 *
 * 判定语义完全不变：
 *   Y_ref = y1 + waist_ratio * (y2 - y1)
 *   violation_score = (cy > Y_ref + threshold) ? 1.0 : 0.0
 *
 * v3 修掉了 v2 在平台上直接报 Runtime Error 的原因：
 *   DataCopyPad 的 GM -> UB 支持通路上，GM -> VECCALC 只在 Ascend 950PR/950DT 上支持，
 *   910B（Atlas A2）不支持。v2 把 boxes/anchors 的搬运目标放在 TBuf<TPosition::VECCALC>
 *   里，等于把数据搬进了位置不合法的缓冲区，上板直接报错。
 *   现在按位置分开三块 UB（都不用 TQue）：
 *     TBuf<TPosition::VECIN>    输入搬运目标（boxes / anchors / threshold）  —— GM -> VECIN 支持
 *     TBuf<TPosition::VECCALC>  纯计算中间量（y1/y2/cy、f16 暂存、1.0、mask）
 *     TBuf<TPosition::VECOUT>   输出搬运源（float / half 结果）              —— UB -> GM 支持
 *
 * 相对原始提交，本版只改开销、不改语义：
 *   1. 去掉全部 TQue：不再有 AllocTensor/EnQue/DeQue/FreeTensor（每次都要取 eventID +
 *      一组 SetFlag/WaitFlag，都是标量流水上的活）。同步只保留 4 组跨流水 flag。
 *   2. 单核只处理一个 tile（行数上限 256，Host 侧保证），kernel 内没有 tile 循环。
 *   3. boxes/anchors/threshold 三个搬运一次性发出，显存往返延迟互相重叠；
 *      threshold 的标量读放在取列之后，与 GatherMask 的执行重叠。
 *   4. 1.0 常量向量只在 Init 里建一次。
 *   5. 不做任何 UB 别名复用（v2 让输出复用了 boxes 的空间，会让自动同步分析到同一地址
 *      被不同 LocalTensor 读写；这里每段独立）。
 *
 * 两个开关只用改 0/1，不动别处：
 *   SBVG_AIV_ONLY   = 1 时声明 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)（只启 AIV）
 *   SBVG_THR_VIA_GM = 1 时 threshold 直接读 GM 标量（对照"从 UB 读"这条路径）
 */
#define SBVG_AIV_ONLY 0
#define SBVG_THR_VIA_GM 0

#include "kernel_operator.h"
#include "safety_belt_violation_geometry_tiling.h"
#include "tiling_key_safety_belt_violation_geometry.h"

using namespace AscendC;

namespace {
// 单核处理的行数上限（一个 tile），必须与 Host 侧 tiling_policy.h 的 kMaxRowsPerCore 一致。
constexpr uint32_t kMaxRows = 256U;
// GatherMask 以 256B 为一个 repeat；64 行对 f32/f16 都是整数个 repeat。
constexpr uint32_t kGroupRows = 64U;

// ---- 输入段（TPosition::VECIN，DataCopyPad 的目标）----
constexpr uint32_t kBoxesOff      = 0U;      constexpr uint32_t kBoxesBytes   = kMaxRows * 4U * 4U;   // 4096
constexpr uint32_t kAnchorsOff    = 4096U;   constexpr uint32_t kAnchorsBytes = kMaxRows * 2U * 4U;   // 2048
constexpr uint32_t kThrOff        = 6144U;   constexpr uint32_t kThrBytes     = 32U;
constexpr uint32_t kInBytes       = kThrOff + kThrBytes;                                            // 6176

// ---- 计算段（TPosition::VECCALC，只被向量指令读写）----
constexpr uint32_t kCoordsOff     = 0U;      constexpr uint32_t kCoordsBytes  = kMaxRows * 3U * 4U;   // y1/y2/cy 各 1024
constexpr uint32_t kHalfCoordsOff = 3072U;   constexpr uint32_t kHalfCoordsBytes = kMaxRows * 3U * 2U;  // 1536
constexpr uint32_t kOnesOff       = 4608U;   constexpr uint32_t kOnesBytes    = kMaxRows * 4U;       // 1024
constexpr uint32_t kMaskOff       = 5632U;   constexpr uint32_t kMaskBytes    = kMaxRows / 8U;       // 32
constexpr uint32_t kCalcBytes     = kMaskOff + kMaskBytes;                                          // 5664

// ---- 输出段（TPosition::VECOUT，UB -> GM 的源）----
constexpr uint32_t kOutOff        = 0U;      constexpr uint32_t kOutBytes     = kMaxRows * 4U;       // 1024
constexpr uint32_t kHalfOutOff    = 1024U;   constexpr uint32_t kHalfOutBytes = kMaxRows * 2U;       // 512
constexpr uint32_t kOutBytesTotal = kHalfOutOff + kHalfOutBytes;                                    // 1536
}  // namespace

template <class T>
class KernelSafetyBeltViolationGeometry {
public:
    __aicore__ inline void Init(
        GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
        GM_ADDR violationScore, const SafetyBeltViolationGeometryTilingData& data) {
        const uint32_t start = GetBlockIdx() * data.rowsPerCore;
        begin_ = start;
        rows_ = start < data.batchSize ? data.batchSize - start : 0U;
        if (rows_ > data.rowsPerCore) {
            rows_ = data.rowsPerCore;
        }
        ratio_ = data.waistRatio;
        thr_ = data.defaultThreshold;  // 未传 threshold 时 Host 给 0.0
        hasThr_ = data.hasThreshold != 0U;
        thrHalf_ = data.thresholdIsHalf != 0U;
        outHalf_ = data.outputIsHalf != 0U;

        boxesGm_.SetGlobalBuffer((__gm__ T*)personBoxes, data.batchSize * 4U);
        anchorsGm_.SetGlobalBuffer((__gm__ T*)anchorPoints, data.batchSize * 2U);
        if (outHalf_) {
            halfOutGm_.SetGlobalBuffer((__gm__ half*)violationScore, data.batchSize);
        } else {
            outGm_.SetGlobalBuffer((__gm__ float*)violationScore, data.batchSize);
        }
        if (hasThr_) {
            if (thrHalf_) {
                thrHalfGm_.SetGlobalBuffer((__gm__ half*)threshold, 1U);
            } else {
                thrGm_.SetGlobalBuffer((__gm__ float*)threshold, 1U);
            }
        }

        pipe_.InitBuffer(inBuf_, kInBytes);
        pipe_.InitBuffer(calcBuf_, kCalcBytes);
        pipe_.InitBuffer(outBuf_, kOutBytesTotal);
        LocalTensor<uint8_t> inBase = inBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> calcBase = calcBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> outBase = outBuf_.Get<uint8_t>();
        boxesUb_ = inBase[kBoxesOff].ReinterpretCast<T>();
        anchorsUb_ = inBase[kAnchorsOff].ReinterpretCast<T>();
        thrUb_ = inBase[kThrOff];
        coordsUb_ = calcBase[kCoordsOff].ReinterpretCast<float>();
        y1Ub_ = coordsUb_;
        y2Ub_ = coordsUb_[kMaxRows];
        cyUb_ = coordsUb_[2U * kMaxRows];
        halfCoordsUb_ = calcBase[kHalfCoordsOff].ReinterpretCast<half>();
        onesUb_ = calcBase[kOnesOff].ReinterpretCast<float>();
        maskUb_ = calcBase[kMaskOff].ReinterpretCast<uint8_t>();
        outUb_ = outBase[kOutOff].ReinterpretCast<float>();
        halfOutUb_ = outBase[kHalfOutOff].ReinterpretCast<half>();

        // Select 需要一条全 1.0 的向量，只建一次。
        Duplicate(onesUb_, 1.0f, kMaxRows);
    }

    __aicore__ inline void Process() {
        if (rows_ == 0U) {
            return;
        }
        const uint32_t valid = rows_;
        const uint32_t padded = ((valid + kGroupRows - 1U) / kGroupRows) * kGroupRows;

        // 未传 threshold 时 Host 已经把 0.0 放进 tiling，这里完全不碰 GM。
        CopyIn(valid);
        GatherColumns(padded);

        // threshold 的标量读取排在取列之后，MTE2 早已落地，这里只付 flag 往返 + 标量读，
        // 且与上面已经发出的向量指令重叠。
        if (hasThr_) {
            ReadThreshold();
        }

        // 公式顺序与参考实现一致：不要合并成 FMA、不要改写成 cy - Y_ref > threshold、
        // 也不要对 waist_ratio == 1 特判（fp32 下 y1+1.0*(y2-y1) 不保证逐位等于 y2）。
        Sub(y2Ub_, y2Ub_, y1Ub_, padded);
        Muls(y2Ub_, y2Ub_, ratio_, padded);
        Add(y2Ub_, y2Ub_, y1Ub_, padded);
        Adds(y2Ub_, y2Ub_, thr_, padded);

        Compare(maskUb_, cyUb_, y2Ub_, CMPMODE::GT, padded);

        // 结果写回：V -> MTE3 同步只需一次。
        if (outHalf_) {
            Select(cyUb_, maskUb_, onesUb_, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, padded);
            Cast(halfOutUb_, cyUb_, RoundMode::CAST_NONE, padded);
            const event_t toMte3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(toMte3);
            WaitFlag<HardEvent::V_MTE3>(toMte3);
            const DataCopyExtParams copy{1U, valid * static_cast<uint32_t>(sizeof(half)), 0U, 0U, 0U};
            DataCopyPad(halfOutGm_[begin_], halfOutUb_, copy);
        } else {
            Select(outUb_, maskUb_, onesUb_, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, padded);
            const event_t toMte3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(toMte3);
            WaitFlag<HardEvent::V_MTE3>(toMte3);
            const DataCopyExtParams copy{1U, valid * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(outGm_[begin_], outUb_, copy);
        }
    }

private:
    // 三个搬运一次性发出：显存往返延迟互相重叠，只付一次。
    __aicore__ inline void CopyIn(uint32_t valid) {
        const DataCopyExtParams boxCopy{
            1U, valid * 4U * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        const DataCopyExtParams anchorCopy{
            1U, valid * 2U * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(boxesUb_, boxesGm_[begin_ * 4U], boxCopy,
                    DataCopyPadExtParams<T>{true, 0U, 0U, static_cast<T>(0)});
        DataCopyPad(anchorsUb_, anchorsGm_[begin_ * 2U], anchorCopy,
                    DataCopyPadExtParams<T>{true, 0U, 0U, static_cast<T>(0)});
#if !SBVG_THR_VIA_GM
        if (hasThr_) {
            if (thrHalf_) {
                const DataCopyExtParams copy{1U, static_cast<uint32_t>(sizeof(half)), 0U, 0U, 0U};
                DataCopyPad(thrUb_.ReinterpretCast<half>(), thrHalfGm_, copy,
                            DataCopyPadExtParams<half>{true, 0U, 0U, static_cast<half>(0)});
            } else {
                const DataCopyExtParams copy{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
                DataCopyPad(thrUb_.ReinterpretCast<float>(), thrGm_, copy,
                            DataCopyPadExtParams<float>{true, 0U, 0U, 0.0f});
            }
        }
#endif
        // 一次 MTE2 -> V 同步覆盖上面所有搬运。
        const event_t toVector = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(toVector);
        WaitFlag<HardEvent::MTE2_V>(toVector);
    }

    // 取列：pattern 4/6 从 [x1,y1,x2,y2] 取 y1/y2，pattern 2 从 [cx,cy] 取 cy。
    // 不用标量循环，也不依赖任何特殊 stride；f16 输入先转 f32 再参与公式与比较。
    __aicore__ inline void GatherColumns(uint32_t padded) {
        const GatherMaskParams boxGather{
            1U, static_cast<uint16_t>(padded * 4U * sizeof(T) / 256U), 8U, 0U};
        const GatherMaskParams anchorGather{
            1U, static_cast<uint16_t>(padded * 2U * sizeof(T) / 256U), 8U, 0U};
        uint64_t reserved = 0U;
        if constexpr (sizeof(T) == sizeof(half)) {
            LocalTensor<half> h1 = halfCoordsUb_;
            LocalTensor<half> h2 = halfCoordsUb_[kMaxRows];
            LocalTensor<half> hc = halfCoordsUb_[2U * kMaxRows];
            GatherMask(h1, boxesUb_, static_cast<uint8_t>(4), false, 0, boxGather, reserved);
            GatherMask(h2, boxesUb_, static_cast<uint8_t>(6), false, 0, boxGather, reserved);
            GatherMask(hc, anchorsUb_, static_cast<uint8_t>(2), false, 0, anchorGather, reserved);
            Cast(y1Ub_, h1, RoundMode::CAST_NONE, padded);
            Cast(y2Ub_, h2, RoundMode::CAST_NONE, padded);
            Cast(cyUb_, hc, RoundMode::CAST_NONE, padded);
        } else {
            GatherMask(y1Ub_, boxesUb_, static_cast<uint8_t>(4), false, 0, boxGather, reserved);
            GatherMask(y2Ub_, boxesUb_, static_cast<uint8_t>(6), false, 0, boxGather, reserved);
            GatherMask(cyUb_, anchorsUb_, static_cast<uint8_t>(2), false, 0, anchorGather, reserved);
        }
    }

    __aicore__ inline void ReadThreshold() {
#if SBVG_THR_VIA_GM
        // 直接读 GM 标量：不需要 flag 与搬运，但标量流水要等一次显存往返。
        thr_ = thrHalf_ ? static_cast<float>(thrHalfGm_.GetValue(0)) : thrGm_.GetValue(0);
#else
        // 从 UB 读标量：MTE2_S 等搬运完成，S_V 让后面的向量指令等标量就位。
        const event_t mte2ToScalar = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(mte2ToScalar);
        WaitFlag<HardEvent::MTE2_S>(mte2ToScalar);
        thr_ = thrHalf_ ? static_cast<float>(thrUb_.ReinterpretCast<half>().GetValue(0))
                        : thrUb_.ReinterpretCast<float>().GetValue(0);
        const event_t scalarToVector = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(scalarToVector);
        WaitFlag<HardEvent::S_V>(scalarToVector);
#endif
    }

    TPipe pipe_;
    TBuf<TPosition::VECIN> inBuf_;
    TBuf<TPosition::VECCALC> calcBuf_;
    TBuf<TPosition::VECOUT> outBuf_;
    LocalTensor<uint8_t> thrUb_;
    LocalTensor<T> boxesUb_;
    LocalTensor<T> anchorsUb_;
    LocalTensor<float> coordsUb_;
    LocalTensor<float> y1Ub_;
    LocalTensor<float> y2Ub_;
    LocalTensor<float> cyUb_;
    LocalTensor<half> halfCoordsUb_;
    LocalTensor<float> onesUb_;
    LocalTensor<uint8_t> maskUb_;
    LocalTensor<float> outUb_;
    LocalTensor<half> halfOutUb_;
    GlobalTensor<T> boxesGm_;
    GlobalTensor<T> anchorsGm_;
    GlobalTensor<float> outGm_;
    GlobalTensor<half> halfOutGm_;
    GlobalTensor<float> thrGm_;
    GlobalTensor<half> thrHalfGm_;
    uint32_t begin_ = 0U;
    uint32_t rows_ = 0U;
    float ratio_ = 0.0f;
    float thr_ = 0.0f;
    bool hasThr_ = false;
    bool thrHalf_ = false;
    bool outHalf_ = false;
};

template <typename DT_PERSON_BOXES>
__global__ __aicore__ void safety_belt_violation_geometry(
    GM_ADDR person_boxes, GM_ADDR anchor_points, GM_ADDR threshold,
    GM_ADDR violation_score, GM_ADDR workspace, GM_ADDR tiling) {
#if SBVG_AIV_ONLY && defined(KERNEL_TASK_TYPE_DEFAULT)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
    GET_TILING_DATA_WITH_STRUCT(SafetyBeltViolationGeometryTilingData, tiling_data, tiling);
    KernelSafetyBeltViolationGeometry<DT_PERSON_BOXES> op;
    op.Init(person_boxes, anchor_points, threshold, violation_score, tiling_data);
    op.Process();
}
