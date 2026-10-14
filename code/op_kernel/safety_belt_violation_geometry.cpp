/**
 * 安全带低挂高用几何判定融合算子 —— Kernel 侧（瘦身版 / lean）
 *
 * 判定语义完全不变：
 *   Y_ref = y1 + waist_ratio * (y2 - y1)
 *   violation_score = (cy > Y_ref + threshold) ? 1.0 : 0.0
 *
 * 本版针对"小算子被固定开销支配"这一事实做结构瘦身（上一次提交约 4.0us，
 * 平台最优 1.5~2.2us，而空 kernel 的核启动开销就有 0.6~2.2us，见 CANN 最佳实践
 * 「头尾开销优化」：核启动开销随启动核数近似线性增长，约 2 核 0.67us -> 40 核 2.25us）：
 *
 *   1. KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)：算子只启 AIV，不让 Cube 核空转。
 *   2. 去掉 TQue：不再用 AllocTensor/EnQue/DeQue/FreeTensor（每次都要取 eventID + 一组
 *      SetFlag/WaitFlag，且是标量流水上的活）。改用 1 个 TBuf + 编译期常量偏移组成
 *      LocalTensor，同步只保留 4 组跨流水 flag：MTE2_V、MTE2_S、S_V、V_MTE3。
 *   3. 单核只处理一个 tile（行数上限 256，Host 侧保证），因此没有 tile 循环。
 *   4. 三个搬运（boxes/anchors/threshold）一次性发出，显存往返延迟互相重叠；
 *      threshold 的标量读放在取列之后，与 GatherMask 的执行重叠。
 *   5. 1.0 常量向量只在 Init 里建一次。
 *
 * 想验证"threshold 到底该从 UB 读还是直接读 GM"时，把下面这个宏改成 1 再提交一次即可，
 * 两种写法都正确，只是性能路径不同。
 */
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

// ---- UB 手工布局：偏移全部是编译期常量，每段都 32B 对齐；按 4B/元素（f32 上界）预留 ----
constexpr uint32_t kBoxesOff     = 0U;
constexpr uint32_t kBoxesBytes   = kMaxRows * 4U * 4U;              // 4096
constexpr uint32_t kAnchorsOff   = kBoxesOff + kBoxesBytes;         // 4096
constexpr uint32_t kAnchorsBytes = kMaxRows * 2U * 4U;              // 2048
constexpr uint32_t kCoordsOff    = kAnchorsOff + kAnchorsBytes;     // 6144
constexpr uint32_t kCoordsBytes  = kMaxRows * 3U * 4U;              // y1/y2/cy 各一段
constexpr uint32_t kHalfCoordsOff = kCoordsOff + kCoordsBytes;      // 9216
constexpr uint32_t kHalfCoordsBytes = kMaxRows * 3U * 2U;           // f16 取列暂存
constexpr uint32_t kOnesOff      = kHalfCoordsOff + kHalfCoordsBytes;  // 10752
constexpr uint32_t kOnesBytes    = kMaxRows * 4U;
constexpr uint32_t kMaskOff      = kOnesOff + kOnesBytes;           // 11776
constexpr uint32_t kMaskBytes    = kMaxRows / 8U;
constexpr uint32_t kThrOff       = kMaskOff + kMaskBytes;           // 11808
constexpr uint32_t kThrBytes     = 32U;
constexpr uint32_t kUbBytes      = kThrOff + kThrBytes;             // 11840
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
#if SBVG_THR_VIA_GM
        if (hasThr_) {
            if (thrHalf_) {
                thrHalfGm_.SetGlobalBuffer((__gm__ half*)threshold, 1U);
            } else {
                thrGm_.SetGlobalBuffer((__gm__ float*)threshold, 1U);
            }
        }
#endif

        pipe_.InitBuffer(ub_, kUbBytes);
        LocalTensor<uint8_t> base = ub_.Get<uint8_t>();
        boxesUb_ = base[kBoxesOff].ReinterpretCast<T>();
        anchorsUb_ = base[kAnchorsOff].ReinterpretCast<T>();
        coordsUb_ = base[kCoordsOff].ReinterpretCast<float>();
        y1Ub_ = coordsUb_;
        y2Ub_ = coordsUb_[kMaxRows];
        cyUb_ = coordsUb_[2U * kMaxRows];
        if constexpr (sizeof(T) == sizeof(half)) {
            halfCoordsUb_ = base[kHalfCoordsOff].ReinterpretCast<half>();
        }
        onesUb_ = base[kOnesOff].ReinterpretCast<float>();
        maskUb_ = base[kMaskOff].ReinterpretCast<uint8_t>();
        thrUb_ = base[kThrOff];
        outUb_ = base[kBoxesOff].ReinterpretCast<float>();  // boxes 取列后即无用，复用其空间
        if constexpr (sizeof(T) == sizeof(half)) {
            halfOutUb_ = base[kBoxesOff].ReinterpretCast<half>();
        }

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
    TBuf<TPosition::VECCALC> ub_;
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
#ifdef KERNEL_TASK_TYPE_DEFAULT
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
    GET_TILING_DATA_WITH_STRUCT(SafetyBeltViolationGeometryTilingData, tiling_data, tiling);
    KernelSafetyBeltViolationGeometry<DT_PERSON_BOXES> op;
    op.Init(person_boxes, anchor_points, threshold, violation_score, tiling_data);
    op.Process();
}
