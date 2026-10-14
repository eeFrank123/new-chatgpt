#pragma once
#include <cstdint>
namespace platform_ascendc {
struct PlatformAscendC {
    explicit PlatformAscendC(const void* p) : cores(*static_cast<const uint32_t*>(p)) {}
    uint32_t GetCoreNumAiv() const { return cores; }
    uint32_t cores;
};
}
