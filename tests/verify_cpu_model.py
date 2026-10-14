#!/usr/bin/env python3
"""Run the real kernel source through a CPU API model and compare with NumPy.

This does not compile Ascend C for an NPU and is not a CANN simulator.
Requires Python 3, NumPy, and g++ with _Float16 support (GCC 12+).
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def reference(boxes, anchors, ratio, threshold):
    boxes32 = boxes.astype(np.float32)
    anchors32 = anchors.astype(np.float32)
    height = np.subtract(boxes32[:, 3], boxes32[:, 1], dtype=np.float32)
    scaled = np.multiply(height, np.float32(ratio), dtype=np.float32)
    line = np.add(boxes32[:, 1], scaled, dtype=np.float32)
    boundary = np.add(line, np.float32(0 if threshold is None else threshold), dtype=np.float32)
    return (anchors32[:, 1] > boundary).astype(np.float32), boundary


def make_case(rng, batch, dtype, ratio, threshold):
    xy = rng.uniform(0, 1024, (batch, 2)).astype(np.float32)
    wh = rng.uniform(0, 1024, (batch, 2)).astype(np.float32)
    boxes = np.concatenate((xy, xy + wh), axis=1).astype(dtype)
    anchors = rng.uniform(0, 2200, (batch, 2)).astype(dtype)
    boxes[::11, 3] = boxes[::11, 1]  # zero-height boxes
    boxes[::17] = 0
    if batch >= 3:
        boxes[1] = [0, 0, 2048, 2048]
        boxes[2] = [-20, -300, 40, 600]
    _, boundary = reference(boxes, anchors, ratio, threshold)
    rounded = boundary.astype(dtype)
    anchors[0::4, 1] = rounded[0::4]
    anchors[1::4, 1] = np.nextafter(rounded[1::4], dtype(-np.inf))
    anchors[2::4, 1] = np.nextafter(rounded[2::4], dtype(np.inf))
    return boxes, anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sanitize', action='store_true', help='enable ASan and UBSan')
    args = parser.parse_args()
    build = ROOT / 'build' / 'cpu_model'
    build.mkdir(parents=True, exist_ok=True)
    executable = build / ('kernel_driver_sanitized' if args.sanitize else 'kernel_driver')
    flags = ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer'] if args.sanitize else ['-O2']
    subprocess.run([
        'g++', '-std=c++17', *flags, '-Wall', '-Wextra', '-Wno-unused-parameter',
        '-ffp-contract=off', '-fno-fast-math', '-I' + str(ROOT / 'tests' / 'cpu_model'),
        str(ROOT / 'tests' / 'kernel_driver.cpp'), '-o', str(executable),
    ], check=True)
    host_executable = build / 'host_driver'
    subprocess.run([
        'g++', '-std=c++17', *flags, '-Wall', '-Wextra', '-DCHECK_STORAGE_FLAGS',
        '-I' + str(ROOT / 'tests' / 'host_model'),
        str(ROOT / 'tests' / 'host_driver.cpp'), '-o', str(host_executable),
    ], check=True)
    subprocess.run([str(host_executable)], check=True)

    rng = np.random.default_rng(2135)
    batches = [1, 2, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65,
               127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 4095, 4096]
    records, cases = [], []
    exact_equalities = 0
    for dtype in (np.float16, np.float32):
        for batch in batches:
            for ratio in (0.8, 0.9, 1.0, 0.33333334):
                for threshold in (None, 0.0, 0.125, 7.5, 0.9999):
                    profiles = [(32, 32)] if dtype == np.float32 else [(32, 32), (16, 32), (32, 16), (16, 16)]
                    for threshold_bits, output_bits in profiles:
                        effective_threshold = threshold
                        if threshold is not None and threshold_bits == 16:
                            effective_threshold = float(np.float16(threshold))
                        boxes, anchors = make_case(rng, batch, dtype, ratio, effective_threshold)
                        expected, boundary = reference(boxes, anchors, ratio, effective_threshold)
                        equalities = int(np.count_nonzero(anchors[:, 1].astype(np.float32) == boundary))
                        for cores in (1, 8, 48):
                            # A nonzero value in the absent-threshold field detects accidental use.
                            header = struct.pack('<IIIIIIff', batch, np.dtype(dtype).itemsize * 8,
                                                 cores, threshold is not None, threshold_bits, output_bits,
                                                 ratio, 12345.0 if threshold is None else threshold)
                            records.append(header + boxes.tobytes() + anchors.tobytes())
                            cases.append((dtype.__name__, batch, ratio, threshold, cores,
                                          threshold_bits, output_bits, expected))
                            exact_equalities += equalities

    # The three examples stated in the problem, in one batch.
    boxes = np.array([[100, 100, 200, 500]] * 3, dtype=np.float32)
    anchors = np.array([[150, 200], [150, 600], [150, 500]], dtype=np.float32)
    records.append(struct.pack('<IIIIIIff', 3, 32, 32, 0, 32, 32, 1.0, 12345.0)
                   + boxes.tobytes() + anchors.tobytes())
    cases.append(('float32', 3, 1.0, None, 32, 32, 32, np.array([0, 1, 0], dtype=np.float32)))
    exact_equalities += 1

    result = subprocess.run([str(executable)], input=b''.join(records), capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    expected_size = sum(case[1] * case[6] // 8 for case in cases)
    assert len(result.stdout) == expected_size, (len(result.stdout), expected_size)
    offset = 0
    rows = 0
    for dtype, batch, ratio, threshold, cores, threshold_bits, output_bits, expected in cases:
        actual = np.frombuffer(result.stdout, dtype='<f2' if output_bits == 16 else '<f4',
                               count=batch, offset=offset)
        np.testing.assert_array_equal(
            actual.astype(np.float32), expected,
            err_msg=f'{dtype=} {batch=} {ratio=} {threshold=} {cores=} {threshold_bits=} {output_bits=}',
        )
        offset += batch * output_bits // 8
        rows += batch
    report = {
        'validation': 'CPU API functional model; NOT NPU compilation/execution',
        'cases_passed': len(cases),
        'rows_compared_exactly': rows,
        'exact_equality_rows': exact_equalities,
        'partition_checks_passed': 4096 * 64,
        'sanitizers': args.sanitize,
        'bounds_and_uninitialized_UB_checks': True,
        'missing_threshold_tested_with_null_pointer': True,
        'host_registration_and_tiling_model': 'passed',
        'fp16_template_compatibility_profiles_tested': True,
    }
    report_file = build / ('results_sanitized.json' if args.sanitize else 'results.json')
    report_file.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
