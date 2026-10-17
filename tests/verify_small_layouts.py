#!/usr/bin/env python3
"""Check both small UB layouts for all B=1..64 with the CPU API model.

This validates input/output semantics and bounds, not CANN compilation,
asynchronous hardware behavior, or bank performance.
"""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    build = ROOT / 'build' / 'v10_small_layouts'
    build.mkdir(parents=True, exist_ok=True)
    profiles = []
    for compact in (False, True):
        for legacy in (False, True):
            name = ('compact' if compact else 'banked') + ('_legacy' if legacy else '_modern')
            executable = build / (name + '.exe')
            flags = [f'-DSBVG_BANKED_COMPACT={int(compact)}', '-DSBVG_SCALAR_THRESHOLD=0']
            if legacy:
                flags.append('-DSBVG_CPU_MODEL_LEGACY_GATHER_ONLY')
            subprocess.run([
                'g++', '-std=c++17', '-O2', '-ffp-contract=off', '-fno-fast-math',
                '-Wall', '-Wextra', '-Wno-unused-parameter', *flags,
                '-I' + str(ROOT / 'tests' / 'cpu_model'),
                str(ROOT / 'tests' / 'banked_small_driver.cpp'), '-o', str(executable),
            ], check=True)
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
            assert result.stdout.strip() == '13824 cases, 449280 rows passed', result.stdout
            profiles.append({'layout': 'compact' if compact else 'banked',
                             'gather_api': 'legacy' if legacy else 'modern',
                             'cases_passed': 13824, 'rows_compared_exactly': 449280,
                             'batches': 'every integer 1..64', 'scalar_threshold': False})
    report = {'validation': 'CPU API functional model; NOT CANN/NPU validation',
              'profiles': profiles, 'total_cases': 55296, 'total_rows': 1797120,
              'checks': ['strict stepwise float32 reference', 'all legal IO modes',
                         'static and runtime ratio', 'unit ratio specialization',
                         'equality and adjacent-float boundaries', 'exact GM extents',
                         'output guard elements', 'one write per output row',
                         'UB bounds and initialized reads', 'mask and event cleanup']}
    (build / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
