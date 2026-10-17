set(SBVG_SMALL_BACKEND 5 CACHE STRING "Small kernel backend: 0=gather, 1=strided DMA, 2=paired DMA, 3=staged gather, 4=banked pingpong, 5=NPU scalar")
if(NOT SBVG_SMALL_BACKEND MATCHES "^[012345]$")
    message(FATAL_ERROR "SBVG_SMALL_BACKEND must be 0, 1, 2, 3, 4 or 5")
endif()

option(SBVG_FORCE_INLINE_ENABLED "Force project device helpers to inline" ON)
set(SBVG_SCALAR_SMALL_LIMIT 64 CACHE STRING "Largest B handled by the NPU scalar backend (1..64)")
if(NOT SBVG_SCALAR_SMALL_LIMIT MATCHES "^([1-9]|[1-5][0-9]|6[0-4])$")
    message(FATAL_ERROR "SBVG_SCALAR_SMALL_LIMIT must be in [1,64]")
endif()
set(sbvg_force_inline 0)
if(SBVG_FORCE_INLINE_ENABLED)
    set(sbvg_force_inline 1)
endif()

# Preserve explicit synchronization and keep compiler auto-sync enabled by default.
option(SBVG_MANUAL_SYNC_ONLY "Disable compiler auto-sync; retain explicit kernel barriers" OFF)
option(SBVG_SCALAR_THRESHOLD "Read float threshold with coherent scalar GM loads; comparison only" OFF)
option(SBVG_BANKED_COMPACT "Use compact UB addresses with identical pingpong math; comparison only" OFF)
set(sbvg_banked_compact 0)
if(SBVG_BANKED_COMPACT)
    set(sbvg_banked_compact 1)
endif()
set(sbvg_scalar_threshold 0)
if(SBVG_SCALAR_THRESHOLD)
    set(sbvg_scalar_threshold 1)
endif()
set(sbvg_kernel_options -DASCENDC_DUMP=0 -ffp-contract=off -fno-fast-math
    -DSBVG_SMALL_BACKEND=${SBVG_SMALL_BACKEND}
    -DSBVG_TARGET_ROWS_PER_CORE=${SBVG_TARGET_ROWS_PER_CORE}
    -DSBVG_SCALAR_THRESHOLD=${sbvg_scalar_threshold}
    -DSBVG_BANKED_COMPACT=${sbvg_banked_compact}
    -DSBVG_FORCE_INLINE_ENABLED=${sbvg_force_inline}
    -DSBVG_SCALAR_SMALL_LIMIT=${SBVG_SCALAR_SMALL_LIMIT})
if(SBVG_MANUAL_SYNC_ONLY)
    list(APPEND sbvg_kernel_options --cce-auto-sync=off)
endif()
npu_op_kernel_options(ascendc_kernels ALL OPTIONS ${sbvg_kernel_options})

# kernel源码
npu_op_kernel_sources(ascendc_kernels
    KERNEL_DIR ./
)

# kernel动态库
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# 添加到package中
npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)
