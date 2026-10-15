# The manual-sync comparison archive changes only this option. Keep the default
# compiler synchronization pass enabled in the primary submission.
option(SBVG_MANUAL_SYNC_ONLY "Disable compiler auto-sync; retain explicit kernel barriers" OFF)
if(SBVG_MANUAL_SYNC_ONLY)
    npu_op_kernel_options(ascendc_kernels ALL OPTIONS
        -DASCENDC_DUMP=0 --cce-auto-sync=off
    )
else()
    npu_op_kernel_options(ascendc_kernels ALL OPTIONS -DASCENDC_DUMP=0)
endif()

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
