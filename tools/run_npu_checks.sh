#!/usr/bin/env bash
# Run on the Ascend Linux machine after installing this project's custom package.
# Source the CANN environment and the custom package's bin/set_env.bash first.
set -euo pipefail

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cann_root="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}"
if [[ -z "$cann_root" || ! -d "$cann_root/include" ]]; then
    printf '%s\n' 'Set ASCEND_HOME_PATH to the CANN toolkit root after sourcing its set_env.sh.' >&2
    exit 2
fi

# Point these at the installed custom package; no system directories are changed.
op_include="${SBVG_OP_API_INCLUDE:-}"
op_lib="${SBVG_OP_API_LIB:-}"
if [[ -z "$op_include" || -z "$op_lib" ]]; then
    printf '%s\n' 'Set SBVG_OP_API_INCLUDE (directory containing aclnn_safety_belt_violation_geometry.h)' \
                  'and SBVG_OP_API_LIB (directory containing libcust_opapi.so).' >&2
    exit 2
fi
if [[ ! -f "$op_include/aclnn_safety_belt_violation_geometry.h" || ! -f "$op_lib/libcust_opapi.so" ]]; then
    printf '%s\n' 'The generated custom ACLNN header/library was not found in the supplied directories.' >&2
    exit 2
fi

build_dir="${SBVG_TEST_BUILD_DIR:-$task_root/build/npu-tests}"
mkdir -p -- "$build_dir"
"${CXX:-g++}" -std=c++17 -O2 -ffp-contract=off -Wall -Wextra \
    -I"$cann_root/include" -I"$cann_root/include/aclnn" -I"$op_include" \
    "$task_root/tests/npu_smoke.cpp" \
    -L"$op_lib" -L"$cann_root/lib64" \
    -Wl,-rpath,"$op_lib" -Wl,-rpath,"$cann_root/lib64" \
    -lcust_opapi -lopapi -lnnopbase -lascendcl -o "$build_dir/npu_smoke"

export LD_LIBRARY_PATH="$op_lib:$cann_root/lib64:${LD_LIBRARY_PATH:-}"
"$build_dir/npu_smoke" "$@"
