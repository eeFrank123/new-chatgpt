cmake_minimum_required(VERSION 3.16.0)

project(custom_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
