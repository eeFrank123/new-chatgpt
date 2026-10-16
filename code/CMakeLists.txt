cmake_minimum_required(VERSION 3.16.0)

project(custom_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SBVG_TARGET_ROWS_PER_CORE 512 CACHE STRING
    "Unbenchmarked target rows per Vector core; a multiple of 64 in [64, 4096]")
if(NOT SBVG_TARGET_ROWS_PER_CORE MATCHES "^[0-9]+$")
    message(FATAL_ERROR "SBVG_TARGET_ROWS_PER_CORE must be an integer")
endif()
math(EXPR sbvg_target_remainder "${SBVG_TARGET_ROWS_PER_CORE} % 64")
if(SBVG_TARGET_ROWS_PER_CORE LESS 64 OR
   SBVG_TARGET_ROWS_PER_CORE GREATER 4096 OR
   NOT sbvg_target_remainder EQUAL 0)
    message(FATAL_ERROR "SBVG_TARGET_ROWS_PER_CORE must be a multiple of 64 in [64, 4096]")
endif()

find_package(ASC REQUIRED)

if(NOT DEFINED ASCEND_COMPUTE_UNIT)
    set(ASCEND_COMPUTE_UNIT ascend910b)
endif()

set(package_name custom)

npu_op_package(${package_name}
    TYPE SHARED
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
