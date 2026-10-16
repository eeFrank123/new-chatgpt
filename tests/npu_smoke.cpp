// Real-device correctness checks. Requires the installed custom ACLNN package.
// This program deliberately does not report host-clock timings as kernel latency.
#include "acl/acl.h"
#include "aclnn_safety_belt_violation_geometry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Check(int64_t status, const char* operation) {
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
    }
}

uint32_t Bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float FromBits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// IEEE binary16 round-to-nearest, ties-to-even. Test coordinates are finite.
uint16_t ToHalf(float value) {
    const uint32_t bits = Bits(value);
    const uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
    const uint32_t fraction = bits & 0x7fffffU;
    const int exponent = static_cast<int>((bits >> 23U) & 0xffU) - 127 + 15;
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        const uint32_t mantissa = fraction | 0x800000U;
        const unsigned shift = static_cast<unsigned>(14 - exponent);
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1U << shift) - 1U);
        const uint32_t halfway = 1U << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U))) ++rounded;
        return static_cast<uint16_t>(sign | rounded);
    }
    uint32_t rounded = (static_cast<uint32_t>(exponent) << 10U) | (fraction >> 13U);
    const uint32_t remainder = fraction & 0x1fffU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (rounded & 1U))) ++rounded;
    return static_cast<uint16_t>(sign | rounded);
}

float FromHalf(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
    const uint32_t exponent = (bits >> 10U) & 31U;
    const uint32_t fraction = bits & 1023U;
    if (exponent == 0U) {
        const float magnitude = std::ldexp(static_cast<float>(fraction), -24);
        return (sign != 0U) ? -magnitude : magnitude;
    }
    return FromBits(sign | ((exponent + 112U) << 23U) | (fraction << 13U));
}

float Quantize(float value, bool half) { return half ? FromHalf(ToHalf(value)) : value; }

float ReferenceLine(float y1, float y2, float ratio, float threshold) {
    // Each volatile assignment enforces a separate binary32 rounding step.
    volatile float difference = y2 - y1;
    volatile float product = difference * ratio;
    volatile float reference = y1 + product;
    volatile float limit = reference + threshold;
    return limit;
}

float AdjacentPositive(float value, bool half, bool above) {
    if (!half) return std::nextafter(value, above ? std::numeric_limits<float>::infinity() : 0.0f);
    uint16_t bits = ToHalf(value);
    if (above) ++bits;
    else if (bits != 0U) --bits;
    return FromHalf(bits);
}

std::vector<uint8_t> Encode(const std::vector<float>& values, bool half) {
    std::vector<uint8_t> bytes(values.size() * (half ? sizeof(uint16_t) : sizeof(float)));
    for (size_t i = 0; i < values.size(); ++i) {
        if (half) {
            const uint16_t value = ToHalf(values[i]);
            std::memcpy(bytes.data() + i * sizeof(value), &value, sizeof(value));
        } else {
            std::memcpy(bytes.data() + i * sizeof(float), &values[i], sizeof(float));
        }
    }
    return bytes;
}

struct Allocation {
    void* address = nullptr;
    explicit Allocation(size_t bytes) {
        if (bytes != 0U) Check(aclrtMalloc(&address, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }
    ~Allocation() { if (address != nullptr) aclrtFree(address); }
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
};

struct Tensor {
    static constexpr size_t kGuard = 64U;
    Allocation allocation;
    std::vector<uint8_t> original;
    size_t payloadBytes;
    aclTensor* tensor = nullptr;

    Tensor(const std::vector<float>& values, const std::vector<int64_t>& shape, bool half)
        : allocation(values.size() * (half ? 2U : 4U) + 2U * kGuard),
          original(values.size() * (half ? 2U : 4U) + 2U * kGuard, uint8_t{0xa5U}),
          payloadBytes(values.size() * (half ? 2U : 4U)) {
        const auto encoded = Encode(values, half);
        std::copy(encoded.begin(), encoded.end(), original.begin() + kGuard);
        Check(aclrtMemcpy(allocation.address, original.size(), original.data(), original.size(),
                          ACL_MEMCPY_HOST_TO_DEVICE), "upload guarded tensor");
        std::vector<int64_t> strides(shape.size(), 1);
        for (size_t i = shape.size(); i > 1U; --i) strides[i - 2U] = strides[i - 1U] * shape[i - 1U];
        tensor = aclCreateTensor(shape.data(), shape.size(), half ? ACL_FLOAT16 : ACL_FLOAT,
                                 strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                                 static_cast<uint8_t*>(allocation.address) + kGuard);
        if (tensor == nullptr) throw std::runtime_error("aclCreateTensor returned nullptr");
    }
    ~Tensor() { if (tensor != nullptr) aclDestroyTensor(tensor); }
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    std::vector<uint8_t> Read(bool unchanged) const {
        std::vector<uint8_t> bytes(original.size());
        Check(aclrtMemcpy(bytes.data(), bytes.size(), allocation.address, bytes.size(),
                          ACL_MEMCPY_DEVICE_TO_HOST), "download guarded tensor");
        if (unchanged && bytes != original) throw std::runtime_error("input data or guard was modified");
        if (!std::equal(bytes.begin(), bytes.begin() + kGuard, original.begin()) ||
            !std::equal(bytes.end() - kGuard, bytes.end(), original.end() - kGuard)) {
            throw std::runtime_error("output memory guard was modified");
        }
        return bytes;
    }
};

struct Runtime {
    int device;
    aclrtStream stream = nullptr;
    explicit Runtime(int id) : device(id) {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device), "aclrtSetDevice");
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    }
    ~Runtime() {
        if (stream != nullptr) aclrtDestroyStream(stream);
        aclrtResetDevice(device);
        aclFinalize();
    }
};

struct Profile { const char* name; bool inputHalf; bool thresholdHalf; bool outputHalf; };

void RunCase(Runtime& runtime, const Profile& profile, int batch, float ratio,
             bool hasThreshold, float threshold, unsigned seed, int repeats) {
    const float actualThreshold = Quantize(threshold, profile.thresholdHalf);
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> coordinate(0, 4096);
    std::vector<float> boxes(static_cast<size_t>(batch) * 4U);
    std::vector<float> anchors(static_cast<size_t>(batch) * 2U);
    std::vector<float> expected(batch);
    for (int i = 0; i < batch; ++i) {
        float y1 = Quantize(coordinate(generator) * 0.25f, profile.inputHalf);
        float y2 = Quantize(y1 + coordinate(generator) * 0.25f, profile.inputHalf);
        const unsigned mode = (static_cast<unsigned>(i) + seed) % 8U;
        if (mode <= 2U) y2 = y1;  // Exactly representable equality limits for half as well.
        if (mode == 3U) y1 = y2 = 0.0f;
        if (mode == 4U) { y1 = 0.0f; y2 = 2048.0f; }
        const float limit = ReferenceLine(y1, y2, ratio, hasThreshold ? actualThreshold : 0.0f);
        float cy = Quantize(coordinate(generator) * 0.5f, profile.inputHalf);
        if (mode == 0U || mode == 3U || mode == 4U) cy = Quantize(limit, profile.inputHalf);
        if (mode == 1U) cy = AdjacentPositive(Quantize(limit, profile.inputHalf), profile.inputHalf, true);
        if (mode == 2U) cy = AdjacentPositive(Quantize(limit, profile.inputHalf), profile.inputHalf, false);
        boxes[4U * i] = 0.0f;
        boxes[4U * i + 1U] = y1;
        boxes[4U * i + 2U] = 2048.0f;
        boxes[4U * i + 3U] = y2;
        anchors[2U * i] = 1024.0f;
        anchors[2U * i + 1U] = cy;
        expected[i] = cy > limit ? 1.0f : 0.0f;
    }
    Tensor person(boxes, {batch, 4}, profile.inputHalf);
    Tensor anchor(anchors, {batch, 2}, profile.inputHalf);
    // A threshold tensor is not constructed at all for the absent-input case.
    std::unique_ptr<Tensor> tolerance;
    if (hasThreshold) tolerance.reset(new Tensor({actualThreshold}, {1}, profile.thresholdHalf));
    Tensor output(std::vector<float>(batch, -123.0f), {batch, 1}, profile.outputHalf);
    for (int repeat = 0; repeat < repeats; ++repeat) {
        uint64_t workspaceBytes = 0;
        aclOpExecutor* executor = nullptr;
        Check(aclnnSafetyBeltViolationGeometryGetWorkspaceSize(
                  person.tensor, anchor.tensor, tolerance ? tolerance->tensor : nullptr,
                  ratio, output.tensor, &workspaceBytes, &executor), "GetWorkspaceSize");
        if (executor == nullptr) throw std::runtime_error("GetWorkspaceSize returned a null executor");
        Allocation workspace(static_cast<size_t>(workspaceBytes));
        Check(aclnnSafetyBeltViolationGeometry(workspace.address, workspaceBytes, executor, runtime.stream),
              "aclnnSafetyBeltViolationGeometry");
        Check(aclrtSynchronizeStream(runtime.stream), "aclrtSynchronizeStream");
    }
    const auto result = output.Read(false);
    for (int i = 0; i < batch; ++i) {
        float actual;
        if (profile.outputHalf) {
            uint16_t packed;
            std::memcpy(&packed, result.data() + Tensor::kGuard + 2U * i, sizeof(packed));
            actual = FromHalf(packed);
        } else {
            std::memcpy(&actual, result.data() + Tensor::kGuard + 4U * i, sizeof(actual));
        }
        if (actual != expected[i]) {
            throw std::runtime_error(std::string(profile.name) + " B=" + std::to_string(batch) +
                                     " ratio=" + std::to_string(ratio) + " threshold=" +
                                     (hasThreshold ? std::to_string(actualThreshold) : "absent") +
                                     " row=" + std::to_string(i) + " expected=" + std::to_string(expected[i]) +
                                     " actual=" + std::to_string(actual));
        }
    }
    person.Read(true);
    anchor.Read(true);
    if (tolerance) tolerance->Read(true);
}
}  // namespace

int main(int argc, char** argv) {
    static_assert(sizeof(float) == 4U && std::numeric_limits<float>::is_iec559, "IEEE binary32 required");
    try {
        int device = 0, selectedBatch = 0, repeats = 1;
        bool compatibility = false;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--compat") compatibility = true;
            else if ((option == "--device" || option == "--batch" || option == "--repeats") && i + 1 < argc) {
                const int value = std::stoi(argv[++i]);
                if (option == "--device") device = value;
                if (option == "--batch") selectedBatch = value;
                if (option == "--repeats") repeats = value;
            } else throw std::runtime_error("Usage: npu_smoke [--device N] [--batch 1..4096] [--repeats N] [--compat]");
        }
        if (selectedBatch < 0 || selectedBatch > 4096 || repeats < 1) throw std::runtime_error("invalid argument range");
        const std::vector<int> batches = selectedBatch ? std::vector<int>{selectedBatch} :
                                                       std::vector<int>{1, 32, 33, 256, 1024, 4096};
        std::vector<Profile> profiles{{"float/float/float", false, false, false},
                                      {"half/float/float", true, false, false}};
        if (compatibility) {
            profiles.push_back({"half/half/float", true, true, false});
            profiles.push_back({"half/float/half", true, false, true});
            profiles.push_back({"half/half/half", true, true, true});
        }
        Runtime runtime(device);
        unsigned cases = 0;
        for (const auto& profile : profiles) {
            for (int batch : batches) {
                for (float ratio : {0.8f, 0.9f, 1.0f, 0.37f}) {
                    for (int thresholdCase = 0; thresholdCase < 4; ++thresholdCase) {
                        const float threshold = thresholdCase <= 1 ? 0.0f : (thresholdCase == 2 ? 1.25f : 32.0f);
                        RunCase(runtime, profile, batch, ratio, thresholdCase != 0, threshold,
                                2135U + cases, repeats);
                        ++cases;
                    }
                }
            }
            std::cout << "PASS profile " << profile.name << '\n';
        }
        std::cout << "PASS " << cases << " real-NPU cases; " << repeats << " launches/case.\n"
                  << "Guards detect writes outside tensor payloads; they do not prove absence of invalid reads.\n"
                  << "No kernel timing measured. Use msprof/msopprof for device timing and a memory checker for reads.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
