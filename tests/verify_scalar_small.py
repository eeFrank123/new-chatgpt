#!/usr/bin/env python3
"""Exercise B=1..64 of the actual NPU scalar backend with a CPU API model.

This checks semantics, initialization, bounds and event contracts, not device
compilation, scalar instruction lowering, or performance on an Ascend NPU.
"""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    build = ROOT / 'build' / 'v11_scalar_small'
    build.mkdir(parents=True, exist_ok=True)
    profiles = []
    for force in (True, False):
        name = 'forced_inline' if force else 'optional_inline'
        executable = build / (name + '.exe')
        subprocess.run([
            'g++', '-std=c++17', '-O2', '-ffp-contract=off', '-fno-fast-math',
            '-Wall', '-Wextra', '-Wno-unused-parameter',
            '-DSBVG_REVIEW_SCALAR_SMALL', '-DSBVG_SCALAR_THRESHOLD=0',
            f'-DSBVG_FORCE_INLINE_ENABLED={int(force)}',
            '-I' + str(ROOT / 'tests' / 'cpu_model'),
            str(ROOT / 'tests' / 'banked_small_driver.cpp'), '-o', str(executable),
        ], check=True)
        result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
        assert result.stdout.strip() == '13824 cases, 449280 rows passed', result.stdout
        profiles.append({'force_inline': force, 'cases_passed': 13824,
                         'rows_compared_exactly': 449280, 'batches': 'every integer 1..64'})
    report = {'validation': 'CPU API functional model; NOT CANN/NPU validation',
              'profiles': profiles, 'total_cases': 27648, 'total_rows': 898560,
              'checks': ['strict stepwise float32 reference', 'all eight legal IO modes',
                         'static and runtime ratio', 'unit ratio specialization',
                         'equality and adjacent-float boundaries', 'exact GM extents',
                         'output guards and one write per row', 'UB initialized reads',
                         'MTE2_S, MTE2_V, V_S and S_MTE3 event visibility']}
    (build / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
