// Opbuild's metadata extraction uses C++11 even when the project uses C++17.
// This translation unit checks the real shared headers without CANN stubs.
#include "../code/op_kernel/safety_belt_violation_geometry_tiling.h"
#include "../code/op_host/tiling_policy.h"

// Explicit test values are separate from the production helper expressions.
#if SBVG_TARGET_ROWS_PER_CORE == 1024
constexpr uint32_t kExpectedRowsFor1024 = 1024U;
constexpr uint32_t kExpectedRowsFor4096 = 1024U;
#elif SBVG_TARGET_ROWS_PER_CORE == 2048
constexpr uint32_t kExpectedRowsFor1024 = 1024U;
constexpr uint32_t kExpectedRowsFor4096 = 2048U;
#elif SBVG_TARGET_ROWS_PER_CORE == 4096
constexpr uint32_t kExpectedRowsFor1024 = 1024U;
constexpr uint32_t kExpectedRowsFor4096 = 4096U;
#else
constexpr uint32_t kExpectedRowsFor1024 = 512U;
constexpr uint32_t kExpectedRowsFor4096 = 512U;
#endif

static_assert(sizeof(SafetyBeltViolationGeometryTilingData) == 16U, "Tiling ABI");
static_assert(safety_belt::kStaticShapeCount == 11U, "Static shape count");
#define CHECK_SHAPE(index, batch, rows) \
    static_assert(safety_belt::StaticBatchForShape(index) == batch, "Batch mapping"); \
    static_assert(safety_belt::StaticCoreRowsForShape(index) == rows, "Partition mapping")
CHECK_SHAPE(0U, 1U, 64U);
CHECK_SHAPE(1U, 32U, 64U);
CHECK_SHAPE(2U, 33U, 64U);
CHECK_SHAPE(3U, 256U, 256U);
CHECK_SHAPE(4U, 1024U, kExpectedRowsFor1024);
CHECK_SHAPE(5U, 4096U, kExpectedRowsFor4096);
CHECK_SHAPE(6U, 64U, 64U);
CHECK_SHAPE(7U, 128U, 128U);
CHECK_SHAPE(8U, 129U, 192U);
CHECK_SHAPE(9U, 257U, 320U);
CHECK_SHAPE(10U, 512U, 512U);
CHECK_SHAPE(11U, 0U, 0U);
CHECK_SHAPE(4294967295U, 0U, 0U);
#undef CHECK_SHAPE
