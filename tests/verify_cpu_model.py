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


def numerical_regressions(dtype):
    """Inputs chosen to reject tempting, numerically different rewrites."""
    # Positive image-like coordinates already distinguish a fused multiply-add,
    # all-double evaluation, and the algebraic weighted-sum formula.
    points = [
        ('fma_and_double_080', 1894.8663330078125, 2330.902099609375, 0.8, 0.0),
        ('fma_and_double_090', 1085.9471435546875, 1870.511474609375, 0.9, 0.0),
        ('weighted_sum_080', 1669.1910400390625, 2956.93115234375, 0.8, 0.0),
        ('rounded_subtraction', 358.29754638671875, 1354.4876708984375, 0.33333334, 0.0),
        ('subtract_compare', 0.0, 1024.0, 1.0, 0.0001),
        ('nonstandard_ratio', 0.125, 2031.5, 0.00001, 0.125),
        ('near_one_ratio', 111.0, 2010.0, float(np.nextafter(np.float32(1), np.float32(0))), 0.0),
    ]
    if dtype == np.float32:
        # The problem allows finite real coordinates and does not clip them.
        # y1 + (y2-y1) can lose y2 under cancellation even at ratio == 1.
        points += [('ratio_one_cancellation', -1e20, 1.0, 1.0, 0.0),
                   ('large_finite_coordinates', -3e30, 3e30, 0.33333334, 1e25),
                   ('large_finite_threshold', 0.0, 1024.0, 0.8, 1e30),
                   ('small_ratio_large_coordinates', 0.0, 1e30, 1e-30, 0.25),
                   ('ratio_one_positive_subnormal', 0.0, 2.0**-149, 1.0, 0.0),
                   ('ratio_one_negative_subnormal', -2.0**-148, -2.0**-149, 1.0, 0.0),
                   ('ratio_one_signed_zero', -0.0, -0.0, 1.0, 0.0)]
    else:
        # Half subtraction would overflow, whereas converting first is finite.
        points += [('half_before_cast_overflow', -65504.0, 65504.0, 0.8, 0.0),
                   ('half_tiny_input', 0.0, float(np.nextafter(np.float16(0), np.float16(1))), 0.8, 0.0)]
    for label, y1, y2, ratio, threshold in points:
        # A nonaligned shape straddles the same numerical cases and the tail.
        boxes = np.array([[0, y1, 2048, y2]] * 33, dtype=dtype)
        anchors = np.zeros((33, 2), dtype=dtype)
        _, boundary = reference(boxes, anchors, ratio, threshold)
        center = boundary.astype(dtype)
        anchors[0::3, 1] = center[0::3]
        anchors[1::3, 1] = np.nextafter(center[1::3], dtype(-np.inf))
        anchors[2::3, 1] = np.nextafter(center[2::3], dtype(np.inf))
        if label == 'ratio_one_cancellation':
            anchors[2::3, 1] = dtype(0.5)  # Reject the y2 shortcut without relying on subnormal cy.
        yield label, boxes, anchors, ratio, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sanitize', action='store_true', help='enable ASan and UBSan')
    parser.add_argument('--rows-per-core', type=int, default=512,
                        help='tiling target: a multiple of 64 in [64, 4096] (default: 512)')
    parser.add_argument('--legacy-gather-api', action='store_true',
                        help='hide the 4-argument GatherMask API to check the production SFINAE fallback')
    parser.add_argument('--small-backend', choices=('banked', 'staged', 'gather', 'paired', 'strided'), default='banked',
                        help='B<=64 kernel: banked pingpong (default), staged Gather, or older comparisons')
    parser.add_argument('--compact-buffers', action='store_true',
                        help='use compact UB addresses with identical banked-backend pingpong math')
    parser.add_argument('--scalar-threshold', action='store_true',
                        help='compare coherent scalar GM float-threshold reads with default vector DMA')
    args = parser.parse_args()
    if args.scalar_threshold and args.small_backend not in ('banked', 'staged', 'gather'):
        parser.error('--scalar-threshold supports only banked, staged or gather backends')
    if args.compact_buffers and args.small_backend != 'banked':
        parser.error('--compact-buffers requires --small-backend banked')
    if not 64 <= args.rows_per_core <= 4096 or args.rows_per_core % 64 != 0:
        parser.error('--rows-per-core must be a multiple of 64 in [64, 4096]')
    build_suffix = '_legacy_gather' if args.legacy_gather_api else ''
    build_suffix += '_' + args.small_backend
    if args.compact_buffers:
        build_suffix += '_compact'
    if args.scalar_threshold:
        build_suffix += '_scalar_threshold'
    build = ROOT / 'build' / 'cpu_model' / f'rows_{args.rows_per_core}{build_suffix}'
    build.mkdir(parents=True, exist_ok=True)
    executable = build / ('kernel_driver_sanitized' if args.sanitize else 'kernel_driver')
    flags = ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer'] if args.sanitize else ['-O2']
    tiling_flag = f'-DSBVG_TARGET_ROWS_PER_CORE={args.rows_per_core}'
    # The actual platform's opbuild subprocess uses C++11 regardless of the
    # CMake CXX_STANDARD. Check both shared constant evaluation and full Host.
    subprocess.run([
        'g++', '-std=c++11', '-pedantic-errors', '-Wall', '-Wextra', tiling_flag,
        '-fsyntax-only', str(ROOT / 'tests' / 'tiling_cpp11_compile.cpp'),
    ], check=True)
    api_flags = ['-DSBVG_CPU_MODEL_LEGACY_GATHER_ONLY'] if args.legacy_gather_api else []
    api_flags.append('-DSBVG_SMALL_BACKEND=' + str({'gather': 0, 'strided': 1, 'paired': 2, 'staged': 3, 'banked': 4}[args.small_backend]))
    api_flags.append('-DSBVG_BANKED_COMPACT=' + str(int(args.compact_buffers)))
    api_flags.append('-DSBVG_SCALAR_THRESHOLD=' + str(int(args.scalar_threshold)))
    subprocess.run([
        'g++', '-std=c++17', *flags, tiling_flag, *api_flags, '-Wall', '-Wextra', '-Wno-unused-parameter',
        '-ffp-contract=off', '-fno-fast-math', '-I' + str(ROOT / 'tests' / 'cpu_model'),
        str(ROOT / 'tests' / 'kernel_driver.cpp'), '-o', str(executable),
    ], check=True)
    host_executable = build / 'host_driver'
    subprocess.run([
        'g++', '-std=c++11', '-pedantic-errors', *flags, tiling_flag, '-Wall', '-Wextra', '-DCHECK_STORAGE_FLAGS',
        '-I' + str(ROOT / 'tests' / 'host_model'),
        str(ROOT / 'tests' / 'host_driver.cpp'), '-o', str(host_executable),
    ], check=True)
    subprocess.run([str(host_executable)], check=True)
    model_executable = build / 'model_contract_driver'
    subprocess.run([
        'g++', '-std=c++17', '-O2', *api_flags, '-Wall', '-Wextra',
        '-I' + str(ROOT / 'tests' / 'cpu_model'),
        str(ROOT / 'tests' / 'model_contract_driver.cpp'), '-o', str(model_executable),
    ], check=True)
    subprocess.run([str(model_executable), 'positive'], check=True)
    subprocess.run([str(model_executable), 'positive_explicit_padding'], check=True)
    subprocess.run([str(model_executable), 'positive_raw_views'], check=True)
    subprocess.run([str(model_executable), 'positive_counter_gather'], check=True)
    subprocess.run([str(model_executable), 'positive_cmpmask_spr'], check=True)
    for mode in ('positive_gather_overload_detection', 'positive_direct_select_128',
                 'positive_direct_select_512', 'positive_direct_select_65', 'positive_aligned_copy',
                 'positive_counter_mask_reset', 'positive_vector_barriers', 'positive_all_barrier',
                 'positive_counter_arithmetic_65', 'positive_counter_arithmetic_4096',
                 'positive_multiblock_rows', 'positive_multiblock_gaps',
                 'positive_highdim_cast_rows', 'positive_highdim_cast_dense',
                 'positive_highdim_mul_broadcast', 'positive_highdim_mul_counter',
                 'positive_ratio_field_dma', 'positive_split_events', 'positive_reverse_events',
                 'positive_masked_column_cast', 'positive_scalar_cache_refresh'):
        subprocess.run([str(model_executable), mode], check=True)
    model_rejection_modes = ['gm_bounds', 'uninitialized', 'illegal_dma_position',
                             'missing_input_sync', 'missing_output_sync', 'wait_without_set',
                             'raw_uninitialized', 'raw_view_bounds', 'raw_ub_bounds', 'raw_missing_mask',
                             'missing_cmpmask_spr', 'missing_cmpmask_scalar', 'unaligned_copy_count',
                             'counter_reset_without_norm', 'missing_vector_raw',
                             'missing_vector_chain_barrier', 'missing_cmpmask_barrier',
                             'multiblock_gm_input_tail', 'multiblock_gm_output_tail',
                             'multiblock_ub_input_tail', 'multiblock_ub_output_tail',
                             'multiblock_wrong_gm_stride', 'multiblock_wrong_ub_stride',
                             'highdim_cast_short_input', 'highdim_cast_short_output',
                             'highdim_cast_wrong_source_stride', 'highdim_cast_wrong_destination_stride',
                             'highdim_mul_wrong_broadcast_block_stride', 'highdim_mul_wrong_broadcast_repeat_stride',
                             'highdim_mul_wrong_source_stride', 'highdim_mul_wrong_destination_stride',
                             'highdim_mul_missing_input_barrier', 'highdim_mul_missing_result_barrier',
                             'ratio_dma_multiple_blocks', 'ratio_dma_extra_field',
                             'split_event_wrong_id', 'split_event_duplicate_set',
                             'split_event_invalid_id', 'split_event_late_copy',
                             'masked_column_cast_full_mask', 'masked_column_cast_wrong_stride',
                             'scalar_cache_missing_refresh']
    if not args.legacy_gather_api:
        model_rejection_modes.append('counter_gather_without_mask')
    for mode in model_rejection_modes:
        rejected = subprocess.run([str(model_executable), mode], capture_output=True)
        assert rejected.returncode != 0, f'CPU model failed to reject {mode}'

    rng = np.random.default_rng(2135)
    batches = [1, 2, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65,
               127, 128, 129, 255, 256, 257, 511, 512, 513,
               1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096]
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

    regression_labels = []
    for dtype in (np.float16, np.float32):
        for label, boxes, anchors, ratio, threshold in numerical_regressions(dtype):
            regression_labels.append(f'{dtype.__name__}:{label}')
            expected, boundary = reference(boxes, anchors, ratio, threshold)
            for cores in (1, 8, 48):
                records.append(struct.pack('<IIIIIIff', len(boxes), np.dtype(dtype).itemsize * 8,
                                           cores, 1, 32, 32, ratio, threshold)
                               + boxes.tobytes() + anchors.tobytes())
                cases.append((dtype.__name__, len(boxes), ratio, threshold, cores, 32, 32, expected))
                exact_equalities += int(np.count_nonzero(anchors[:, 1].astype(np.float32) == boundary))

    # Attr values are runtime float32 values, not a closed list of common ratios.
    random_ratios = rng.uniform(0.00001, 1.0, 16).astype(np.float32)
    for dtype in (np.float16, np.float32):
        for batch in (33, 1025):
            for ratio in random_ratios:
                for threshold in (None, 0.00001, 33.125):
                    boxes, anchors = make_case(rng, batch, dtype, ratio, threshold)
                    expected, boundary = reference(boxes, anchors, ratio, threshold)
                    for cores in (1, 8, 48):
                        records.append(struct.pack('<IIIIIIff', batch, np.dtype(dtype).itemsize * 8,
                                                   cores, threshold is not None, 32, 32, ratio,
                                                   12345.0 if threshold is None else threshold)
                                       + boxes.tobytes() + anchors.tobytes())
                        cases.append((dtype.__name__, batch, ratio, threshold, cores, 32, 32, expected))
                        exact_equalities += int(np.count_nonzero(anchors[:, 1].astype(np.float32) == boundary))

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
        'target_rows_per_core': args.rows_per_core,
        'gather_api': 'legacy_7_argument_only' if args.legacy_gather_api else '4_and_7_argument_overloads',
        'small_batch_kernel': args.small_backend,
        'compact_banked_buffers': args.compact_buffers,
        'scalar_float_threshold': args.scalar_threshold,
        'half_small_conversion': 'one masked Cast:3 column repeats, active rows only; no half-coordinate prezero',
        'identity_operations': 'constant unit ratio skips multiply1; absent threshold skips add+0; Sub and Add(y1) retained',
        'cases_passed': len(cases),
        'rows_compared_exactly': rows,
        'exact_equality_rows': exact_equalities,
        'partition_checks_passed': 4096 * 64,
        'sanitizers': args.sanitize,
        'bounds_and_uninitialized_UB_checks': True,
        'cross_pipeline_read_visibility_model': 'MTE2/V/MTE3 plus explicit event pairing; not an asynchronous simulator',
        'same_vector_pipeline_read_visibility_model': 'UB and CMPMASK RAW dependencies require an explicit barrier or waited outgoing vector event',
        'synchronization_model_limitations': ['not an asynchronous simulator', 'does not model WAR/WAW hazards',
                                            'does not model SDK event ID allocation or compiler-inserted synchronization'],
        'event_id_model': '8 independent IDs per HardEvent; matching Set/Wait; partial visibility; monotonic waits',
        'model_hazard_rejection_selfchecks': model_rejection_modes,
        'dma_padding_model_selfchecks': ['implicit_first_element', 'explicit_zero_right_padding',
                                        'multiblock_1_32_33_float_and_half_per_row_first_element',
                                        'GM_byte_gaps_and_UB_32_byte_gaps',
                                        'multiblock_packed_and_gapped_GM_output'],
        'physical_GM_allocation_checks': True,
        'exactly_one_write_per_output_row': True,
        'missing_threshold_tested_with_null_pointer': True,
        'static_and_generic_kernel_dispatch_checks': 'constant-ratio cases receive null tiling; shape-only cases DMA precisely the ratio field once per core; fallback asserts exactly one full tiling load per block',
        'host_registration_and_tiling_model': 'passed',
        'shared_header_cpp11_constant_evaluation': 'passed without CANN stubs',
        'host_cpp11_compile': 'passed with Host API model; not a CANN build',
        'fp16_template_compatibility_profiles_tested': True,
        'random_runtime_ratio_count': len(random_ratios),
        'raw_UB_alias_and_bounds_model': 'shared 192 KiB UB; per-view bounds and logical DMA positions',
        'small_batch_model_selfchecks': ['counter_gather_without_tail_reads',
                                         'cmpmask_spr_select_with_broadcast_alias'],
        'direct_api_model_selfchecks': ['grouped_float_and_half_counter_gather_33',
                                        'highdim_mul_3_repeats_block_broadcast_distinct_lanes_and_destination_gaps',
                                        'highdim_mul_counter_137_tail',
                                        'ratio_DMA_exact_4B_field_and_32B_broadcast',
                                        'mode1_select_128_and_512_distinct_masks_across_repeats',
                                        'mode1_select_counter_65_tail',
                                        'mode1_select_32_byte_true_data_broadcast',
                                        'memory_compare_preserves_cmpmask_scalar',
                                        'aligned_count_dma',
                                        'counter_arithmetic_65_exact_view_tail',
                                        'counter_arithmetic_4096_broadcast_and_in_place',
                                        'highdim_half_to_float_mask8_repeats33_with_32_byte_row_pitches',
                                        'highdim_dense_cast_128_two_repeats_both_directions',
                                        'reset_mask_preserves_counter_mode_until_set_mask_norm',
                                        'normal_mask_mode_at_every_kernel_exit'],
        'numerical_regressions': regression_labels,
    }
    report_file = build / ('results_sanitized.json' if args.sanitize else 'results.json')
    report_file.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
