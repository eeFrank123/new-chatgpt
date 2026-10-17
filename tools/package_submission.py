#!/usr/bin/env python3
"""Build reproducible CANNJudge source ZIPs without changing the source tree.

Examples:
    python tools/package_submission.py
    python tools/package_submission.py --rows 128 256 512 1024 --output-dir dist
    python tools/package_submission.py --prefix safetybelt_2135_v9 --rows 512
    python tools/package_submission.py --scalar-threshold
    python tools/package_submission.py --small-backend gather
    python tools/package_submission.py --manual-sync-only --rows 512

Rows per core is an unbenchmarked tuning parameter. Compile and measure the
profiles on the target NPU before treating any configuration as the fastest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "code"
EXCLUDED_DIRS = {"__pycache__", "build", "dist", "CMakeFiles", "node_modules"}
EXCLUDED_NAMES = {
    "CMakeCache.txt", "Makefile", "build.ninja", "cmake_install.cmake",
    "compile_commands.json", "install_manifest.txt",
}
EXCLUDED_SUFFIXES = {
    ".a", ".bin", ".dll", ".dylib", ".exe", ".lib", ".log", ".o", ".obj",
    ".pdb", ".pyc", ".pyo", ".so", ".tmp", ".zip",
}
PROFILE_DEFAULTS = {
    "code/op_host/tiling_policy.h": (
        rb"(?m)^(#define[ \t]+SBVG_TARGET_ROWS_PER_CORE[ \t]+)[0-9]+([ \t]*)$"
    ),
    "code/op_kernel/safety_belt_violation_geometry_tiling.h": (
        rb"(?m)^(#define[ \t]+SBVG_TARGET_ROWS_PER_CORE[ \t]+)[0-9]+([ \t]*)$"
    ),
    "code/CMakeLists.txt": (
        rb"(?m)^(set\(SBVG_TARGET_ROWS_PER_CORE[ \t]+)[0-9]+([ \t]+CACHE[ \t]+STRING[ \t]*)$"
    ),
}
REQUIRED_SOURCES = set(PROFILE_DEFAULTS) | {
    "code/op_host/CMakeLists.txt",
    "code/op_host/safety_belt_violation_geometry.cpp",
    "code/op_kernel/CMakeLists.txt",
    "code/op_kernel/safety_belt_violation_geometry.cpp",
    "code/op_kernel/safety_belt_violation_geometry_gather.h",
    "code/op_kernel/safety_belt_violation_geometry_small.h",
    "code/op_kernel/safety_belt_violation_geometry_strided.h",
    "code/op_kernel/safety_belt_violation_geometry_paired.h",
    "code/op_kernel/safety_belt_violation_geometry_staged.h",
    "code/op_kernel/safety_belt_violation_geometry_threshold.h",
    "code/op_kernel/safety_belt_violation_geometry_ratio.h",
    "code/op_kernel/safety_belt_violation_geometry_tiling.h",
    "code/op_kernel/tiling_key_safety_belt_violation_geometry.h",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_archive_name(name: str) -> None:
    path = PurePosixPath(name)
    if (path.is_absolute() or "\\" in name or ":" in name or
            path.parts[0:1] != ("code",) or
            any(part in {"", ".", ".."} for part in path.parts)):
        raise ValueError(f"Unsafe archive path: {name!r}")


def read_sources() -> dict[str, bytes]:
    if not SOURCE_ROOT.is_dir() or SOURCE_ROOT.is_symlink():
        raise ValueError(f"Source directory must be a regular directory: {SOURCE_ROOT}")
    files: dict[str, bytes] = {}
    for current, directories, names in os.walk(SOURCE_ROOT, followlinks=False):
        directory = Path(current)
        directories[:] = sorted(
            name for name in directories
            if not name.startswith(".") and name not in EXCLUDED_DIRS
        )
        for name in directories:
            if (directory / name).is_symlink():
                raise ValueError(f"Source symlink is not allowed: {directory / name}")
        for name in sorted(names):
            path = directory / name
            if (name.startswith(".") or name in EXCLUDED_NAMES or
                    path.suffix.lower() in EXCLUDED_SUFFIXES or ".so." in name):
                continue
            if path.is_symlink():
                raise ValueError(f"Source symlink is not allowed: {path}")
            if not path.is_file():
                continue
            path.resolve().relative_to(SOURCE_ROOT.resolve())
            archive_name = path.relative_to(ROOT).as_posix()
            validate_archive_name(archive_name)
            content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            # Source text is deliberately required; binary build output never belongs in a submission.
            content.decode("utf-8")
            if b"\0" in content:
                raise ValueError(f"Binary file is not allowed in source ZIP: {path}")
            files[archive_name] = content
    for required in sorted(REQUIRED_SOURCES):
        if required not in files:
            raise ValueError(f"Required source file is missing: {required}")
    return dict(sorted(files.items()))


def source_git_state() -> dict[str, str | bool | None]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *arguments], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
        )
        return result.stdout.strip()

    try:
        return {
            "head": git("rev-parse", "HEAD"),
            "code_has_uncommitted_changes": bool(git("status", "--porcelain", "--", "code")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "code_has_uncommitted_changes": None}


def profile_sources(sources: dict[str, bytes], rows: int, manual_sync_only: bool = False,
                    small_backend: str = 'staged', scalar_threshold: bool = False) -> dict[str, bytes]:
    result = sources.copy()
    for name, pattern in PROFILE_DEFAULTS.items():
        replacement = lambda match: match[1] + str(rows).encode("ascii") + match[2]
        content, count = re.subn(pattern, replacement, result[name])
        if count != 1:
            raise ValueError(f"Expected one rows-per-core default in {name}; found {count}")
        result[name] = content
    if manual_sync_only:
        name = "code/op_kernel/CMakeLists.txt"
        before = b'option(SBVG_MANUAL_SYNC_ONLY "Disable compiler auto-sync; retain explicit kernel barriers" OFF)'
        after = before[:-4] + b'ON)'
        if result[name].count(before) != 1:
            raise ValueError("Expected exactly one manual-sync option default")
        result[name] = result[name].replace(before, after)
    backend = str({'gather': 0, 'strided': 1, 'paired': 2, 'staged': 3}[small_backend]).encode('ascii')
    replacements = {
        "code/op_kernel/safety_belt_violation_geometry.cpp": (
            b"#define SBVG_SMALL_BACKEND 3\n", b"#define SBVG_SMALL_BACKEND " + backend + b"\n"),
        "code/op_kernel/CMakeLists.txt": (
            b"set(SBVG_SMALL_BACKEND 3 CACHE STRING", b"set(SBVG_SMALL_BACKEND " + backend + b" CACHE STRING"),
    }
    for name, (before, after) in replacements.items():
        if result[name].count(before) != 1:
            raise ValueError(f"Expected exactly one small-backend default in {name}")
        result[name] = result[name].replace(before, after)
    if scalar_threshold:
        name = "code/op_kernel/CMakeLists.txt"
        before = b'option(SBVG_SCALAR_THRESHOLD "Read float threshold with coherent scalar GM loads; comparison only" OFF)'
        if result[name].count(before) != 1:
            raise ValueError("Expected exactly one scalar-threshold option default")
        result[name] = result[name].replace(before, before[:-4] + b'ON)')
        name = "code/op_kernel/safety_belt_violation_geometry_threshold.h"
        before = b'#define SBVG_SCALAR_THRESHOLD 0\n'
        if result[name].count(before) != 1:
            raise ValueError("Expected exactly one scalar-threshold header default")
        result[name] = result[name].replace(before, b'#define SBVG_SCALAR_THRESHOLD 1\n')
    return result


def write_archive(path: Path, sources: dict[str, bytes]) -> str:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sources.items():
            validate_archive_name(name)
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != list(sources):
            raise ValueError(f"Archive file list differs from source snapshot: {path}")
        if archive.testzip() is not None:
            raise ValueError(f"Archive failed its CRC check: {path}")
        for name, expected in sources.items():
            validate_archive_name(name)
            actual = archive.read(name)
            if actual != expected or b"\r" in actual:
                raise ValueError(f"Archive content differs from LF-normalized source: {name}")
    return sha256(path.read_bytes())


def parse_rows(value: str) -> int:
    try:
        rows = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("rows must be an integer") from error
    if rows < 64 or rows > 4096 or rows % 64:
        raise argparse.ArgumentTypeError("rows must be a multiple of 64 in [64, 4096]")
    return rows


def parse_prefix(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value) is None:
        raise argparse.ArgumentTypeError(
            "prefix must contain 1-80 ASCII letters, digits, underscores, or hyphens "
            "and start with a letter or digit"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--rows", type=parse_rows, nargs="+", default=[512])
    parser.add_argument("--prefix", type=parse_prefix, default="safetybelt_2135_v9")
    parser.add_argument("--manual-sync-only", action="store_true",
                        help="make a separate comparison archive disabling compiler auto-sync")
    parser.add_argument("--scalar-threshold", action="store_true",
                        help="make a separate comparison archive reading float threshold through coherent scalar GM loads")
    parser.add_argument("--small-backend", choices=('staged', 'gather', 'paired', 'strided'), default='staged',
                        help="B<=64 backend: staged float Gather (default), original Gather, or older DMA comparisons")
    args = parser.parse_args()
    if args.scalar_threshold and args.small_backend not in ('staged', 'gather'):
        parser.error('--scalar-threshold supports only staged or gather backends')
    if len(set(args.rows)) != len(args.rows):
        parser.error("--rows values must be unique")
    output = args.output_dir.resolve()
    if output == SOURCE_ROOT.resolve() or SOURCE_ROOT.resolve() in output.parents:
        parser.error("--output-dir must be outside code/")
    try:
        sources = read_sources()
        # Resolve every configuration before writing any output.
        profiles = [(rows, profile_sources(sources, rows, args.manual_sync_only, args.small_backend, args.scalar_threshold))
                    for rows in args.rows]
        manifest = {
            "format_version": 1,
            "prefix": args.prefix,
            "compiler_auto_sync": not args.manual_sync_only,
            "small_backend": args.small_backend,
            "scalar_float_threshold": args.scalar_threshold,
            "note": (
                "Configurations are unbenchmarked. NPU compilation, correctness testing, and "
                "latency measurements on the target CANN/Ascend environment are required. "
                "Archives reflect the working-tree source snapshot, not only the recorded git HEAD."
            ),
            "source_git": source_git_state(),
            "source_files": {name: sha256(content) for name, content in sources.items()},
            "profiles": [],
        }
        output.mkdir(parents=True, exist_ok=True)
        for rows, files in profiles:
            suffix = ('_' + args.small_backend + ("_scalar_threshold" if args.scalar_threshold else "")
                      + ("_manualsync" if args.manual_sync_only else ""))
            name = f"{args.prefix}_rows{rows}{suffix}.zip"
            path = output / name
            digest = write_archive(path, files)
            manifest["profiles"].append({
                "rows_per_core": rows,
                "compiler_auto_sync": not args.manual_sync_only,
                "small_backend": args.small_backend,
                "scalar_float_threshold": args.scalar_threshold,
                "archive": name,
                "sha256": digest,
                "file_count": len(files),
                "files": {name: sha256(content) for name, content in files.items()},
            })
            print(f"{name}: {len(files)} source files; SHA256 {digest}")
        manifest_path = output / f"manifest{suffix}.json"
        manifest_path.write_bytes(
            (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )
        print(f"Manifest: {manifest_path}")
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(1, f"Packaging failed: {error}\n")


if __name__ == "__main__":
    main()
