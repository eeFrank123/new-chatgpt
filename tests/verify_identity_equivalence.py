"""Independent FP32 equivalence checks for identity-only geometry rewrites."""
import json
from pathlib import Path

import numpy as np

np.seterr(all='ignore')
rng = np.random.default_rng(739281)
one, zero = np.float32(1.0), np.float32(0.0)

# Every positive/negative FP32 subnormal, both signed zeros, and representative
# exponent/fraction boundaries. Multiplication by +1 must preserve every bit.
identity_checks = 0
for start in range(0, 0x00800000, 0x100000):
    bits = np.arange(start, min(start + 0x100000, 0x00800000), dtype=np.uint32)
    for sign in (np.uint32(0), np.uint32(0x80000000)):
        values = np.bitwise_or(bits, sign).view(np.float32)
        scaled = np.multiply(values, one, dtype=np.float32)
        np.testing.assert_array_equal(scaled.view(np.uint32), values.view(np.uint32))
        identity_checks += int(len(values))
edges = np.array([0, 0x80000000, 0x00800000, 0x80800000, 0x3f800000,
                  0xbf800000, 0x7f7fffff, 0xff7fffff, 0x7f800000,
                  0xff800000], dtype=np.uint32).view(np.float32)
np.testing.assert_array_equal(np.multiply(edges, one, dtype=np.float32).view(np.uint32),
                              edges.view(np.uint32))
identity_checks += len(edges)

def finite_random(count):
    raw = rng.integers(0, 1 << 32, count, dtype=np.uint32)
    # Replace only the non-finite exponent by zero while retaining random
    # fractions and sign, rather than clipping the full exponent range.
    nonfinite = (raw & np.uint32(0x7f800000)) == np.uint32(0x7f800000)
    raw[nonfinite] &= np.uint32(0x807fffff)
    return raw.view(np.float32)

geometry_checks = 0
for _ in range(8):
    a, b, cy = [finite_random(131072) for _ in range(3)]
    y1, y2 = np.minimum(a, b), np.maximum(a, b)
    height = np.subtract(y2, y1, dtype=np.float32)
    scaled = np.multiply(height, one, dtype=np.float32)
    line = np.add(y1, scaled, dtype=np.float32)
    boundary = np.add(line, zero, dtype=np.float32)
    simplified_line = np.add(y1, height, dtype=np.float32)
    for probe in (cy, line, np.nextafter(line, np.float32(-np.inf)),
                  np.nextafter(line, np.float32(np.inf)), np.zeros_like(line),
                  -np.zeros_like(line)):
        np.testing.assert_array_equal(probe > boundary, probe > simplified_line)
        geometry_checks += len(probe)

report = {
    'multiplication_identity_bit_checks': identity_checks,
    'all_signed_fp32_subnormals_and_zeros_covered': True,
    'geometry_strict_comparison_checks': geometry_checks,
    'random_coordinate_exponents': 'entire finite FP32 range',
    'boundary_probes': ['equal', 'adjacent_lower', 'adjacent_higher', '+0', '-0'],
    'overflowing_intermediate_height': 'allowed and checked',
    'rewrites': ['omit multiply by exact positive 1', 'omit add of exact positive 0'],
    'kept': ['rounded y2-y1', 'rounded y1+height', 'strict cy>boundary'],
    'limitations': 'IEEE/NumPy reference proof; does not establish hardware FTZ behavior or NPU latency',
}
out = Path(__file__).resolve().parents[1] / 'build' / 'identity_equivalence.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report))
