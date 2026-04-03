#!/usr/bin/env python3
"""
test_models_aot.py — Compare JIT vs Native AOT: correctness and performance.

Builds both JIT (dotnet publish) and AOT (dotnet publish -p:PublishAot=true) binaries,
then runs each cached model through both, comparing:
  - Correctness: expected substring in generated output
  - Startup time: total subprocess wall-clock time (includes process init + model load + inference)
  - Inference speed: decode tok/s and prefill tok/s from --json timings

Usage:
    python scripts/test_models_aot.py                   # run all cached models
    python scripts/test_models_aot.py --filter smol      # only models matching "smol"
    python scripts/test_models_aot.py --list              # show available test cases
    python scripts/test_models_aot.py --skip-build        # reuse existing publish output
    python scripts/test_models_aot.py --runs 3            # average over N runs per binary
    python scripts/test_models_aot.py --save results.json # save results to JSON
    python scripts/test_models_aot.py --show results.json # display saved results
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Reuse test case definitions and model resolution
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_models import TEST_CASES, TestCase, _model_is_cached
from bench_compare import resolve_model, _get_gguf_layers
import re as _re

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_PROJECT = REPO_ROOT / "src" / "DotLLM.Cli"


def _extract_param_billions(name: str) -> float | None:
    """Extract model parameter count in billions from a test case name.

    Matches patterns like '135M', '0.5B', '1B', '3.8B', '7B', '11B'.
    Returns value in billions (e.g. 135M → 0.135, 3B → 3.0).
    """
    # Try NB or N.NB first (billions)
    m = _re.search(r'(\d+(?:\.\d+)?)[Bb]', name)
    if m:
        return float(m.group(1))
    # Try NM (millions → convert to billions)
    m = _re.search(r'(\d+(?:\.\d+)?)[Mm]', name)
    if m:
        return float(m.group(1)) / 1000.0
    return None


def _parse_size_filter(spec: str) -> callable:
    """Parse a size filter like '<3B', '>=1B', '<=0.5B' into a predicate on billions.

    Supported operators: <, <=, >, >=, =, ==
    Suffix: B or M (case-insensitive). B = billions, M = millions.
    Examples: '<3B', '>=1B', '<=500M', '=7B'
    """
    m = _re.match(r'^([<>=!]+)\s*(\d+(?:\.\d+)?)\s*([BbMm]?)$', spec.strip())
    if not m:
        raise ValueError(
            f"Invalid size filter '{spec}'. "
            f"Expected format: <operator><number><B|M> (e.g. '<3B', '>=500M')")

    op_str, val_str, unit = m.group(1), m.group(2), m.group(3).upper()
    val = float(val_str)
    if unit == 'M':
        val /= 1000.0  # convert to billions

    import operator
    ops = {
        '<': operator.lt, '<=': operator.le,
        '>': operator.gt, '>=': operator.ge,
        '=': operator.eq, '==': operator.eq,
    }
    if op_str not in ops:
        raise ValueError(f"Unknown operator '{op_str}' in size filter. Use <, <=, >, >=, =")

    op_fn = ops[op_str]
    return lambda billions: op_fn(billions, val)


def _jit_publish_dir() -> Path:
    """JIT (framework-dependent) publish output directory."""
    return CLI_PROJECT / "bin" / "Release" / "net10.0" / "publish"


def _aot_publish_dir() -> Path:
    """Native AOT publish output directory."""
    # AOT publish uses a RID-specific path
    if sys.platform == "win32":
        rid = "win-x64"
    elif sys.platform == "darwin":
        rid = "osx-arm64"
    else:
        rid = "linux-x64"
    return CLI_PROJECT / "bin" / "Release" / "net10.0" / rid / "publish"


def _exe_name(aot: bool) -> str:
    """Binary name for AOT vs JIT."""
    ext = ".exe" if sys.platform == "win32" else ""
    if aot:
        return f"dotllm{ext}"
    return f"DotLLM.Cli{ext}"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _build_jit() -> Path:
    """Build the JIT (framework-dependent) binary via dotnet publish."""
    print("[build] Publishing JIT binary...")
    cmd = [
        "dotnet", "publish", str(CLI_PROJECT),
        "-c", "Release",
        "--no-restore",
    ]
    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    elapsed = time.monotonic() - start

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s):")
        print(result.stderr or result.stdout)
        sys.exit(1)

    exe = _jit_publish_dir() / _exe_name(aot=False)
    if not exe.exists():
        # fallback: try non-publish bin dir
        exe = CLI_PROJECT / "bin" / "Release" / "net10.0" / _exe_name(aot=False)
    print(f"  OK ({elapsed:.1f}s) → {exe}")
    return exe


def _build_aot() -> Path:
    """Build the Native AOT binary via dotnet publish -p:PublishAot=true."""
    print("[build] Publishing Native AOT binary...")
    if sys.platform == "win32":
        rid = "win-x64"
    elif sys.platform == "darwin":
        rid = "osx-arm64"
    else:
        rid = "linux-x64"

    cmd = [
        "dotnet", "publish", str(CLI_PROJECT),
        "-c", "Release",
        "-p:PublishAot=true",
        f"-r", rid,
    ]
    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    elapsed = time.monotonic() - start

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s):")
        # Show last 20 lines of output for diagnostics
        lines = (result.stderr or result.stdout).strip().splitlines()
        for line in lines[-20:]:
            print(f"    {line}")
        sys.exit(1)

    exe = _aot_publish_dir() / _exe_name(aot=True)
    size_mb = exe.stat().st_size / (1024 * 1024) if exe.exists() else 0
    print(f"  OK ({elapsed:.1f}s, {size_mb:.1f}MB) → {exe}")
    return exe


# ---------------------------------------------------------------------------
# Run a single test
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of a single CLI invocation."""
    passed: bool
    text: str
    elapsed_s: float  # wall-clock subprocess time
    decode_tok_s: float = 0.0
    prefill_tok_s: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    load_ms: float = 0.0
    error: str | None = None


def _run_once(exe: Path, model_path: Path, tc: TestCase,
              device: str = "cpu", gpu_layers: int | None = None) -> RunResult:
    """Run the CLI binary once and capture results."""
    cmd = [str(exe)]

    cmd += [
        "run", str(model_path),
        "-p", tc.prompt,
        "-n", str(tc.max_tokens),
        "-t", "0",  # greedy
        "--json",
    ]
    if gpu_layers is not None:
        cmd += ["--gpu-layers", str(gpu_layers)]
    else:
        cmd += ["--device", device]

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return RunResult(False, "", time.monotonic() - start, error="TIMEOUT (300s)")
    except FileNotFoundError:
        return RunResult(False, "", time.monotonic() - start, error=f"binary not found: {exe}")

    elapsed = time.monotonic() - start

    if result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip()
        # Find the first Error: line
        for line in err.splitlines():
            if "Error:" in line:
                return RunResult(False, "", elapsed, error=line.strip())
        return RunResult(False, "", elapsed, error=f"exit {result.returncode}: {err[:200]}")

    # Parse JSON output
    raw = result.stdout.strip()
    json_start = raw.find("{")
    if json_start < 0:
        return RunResult(False, "", elapsed, error=f"no JSON: {raw[:200]}")
    try:
        data = json.loads(raw[json_start:])
    except json.JSONDecodeError:
        return RunResult(False, "", elapsed, error=f"bad JSON: {raw[:200]}")

    text = data.get("text", "")
    timings = data.get("timings", {})
    passed = tc.expected in text

    return RunResult(
        passed=passed,
        text=text.strip()[:80],
        elapsed_s=elapsed,
        decode_tok_s=timings.get("decode_tok_s", 0) or 0,
        prefill_tok_s=timings.get("prefill_tok_s", 0) or 0,
        prefill_ms=timings.get("prefill_ms", 0) or 0,
        decode_ms=timings.get("decode_ms", 0) or 0,
        load_ms=timings.get("load_ms", 0) or 0,
        error=None if passed else f"expected '{tc.expected}' not in '{text[:60]}'",
    )


# ---------------------------------------------------------------------------
# Aggregated comparison result
# ---------------------------------------------------------------------------

@dataclass
class CompareResult:
    """Comparison of JIT vs AOT for a single model."""
    name: str
    arch: str
    quant: str
    device: str
    runs: int
    # JIT
    jit_passed: bool
    jit_elapsed_s: float      # best wall-clock time
    jit_decode_tok_s: float   # best decode throughput
    jit_prefill_tok_s: float  # best prefill throughput
    jit_load_ms: float
    jit_text: str
    # AOT
    aot_passed: bool
    aot_elapsed_s: float
    aot_decode_tok_s: float
    aot_prefill_tok_s: float
    aot_load_ms: float
    aot_text: str
    # Derived
    startup_speedup: float   # jit_elapsed / aot_elapsed
    decode_ratio: float      # aot_decode_tok_s / jit_decode_tok_s (>1 = AOT faster)
    error: str | None = None


def _run_comparison(jit_exe: Path, aot_exe: Path, model_path: Path, tc: TestCase,
                    runs: int, device: str = "cpu",
                    gpu_layers: int | None = None) -> CompareResult:
    """Run JIT and AOT binaries multiple times and compare best results."""
    quant = tc.quant or "default"

    def best_of(exe: Path) -> RunResult:
        results = []
        for _ in range(runs):
            r = _run_once(exe, model_path, tc, device=device, gpu_layers=gpu_layers)
            results.append(r)
        # Pick the run with best decode throughput (or first if all failed)
        passed_runs = [r for r in results if r.passed]
        if passed_runs:
            return max(passed_runs, key=lambda r: r.decode_tok_s)
        return results[0]

    jit = best_of(jit_exe)
    aot = best_of(aot_exe)

    startup_speedup = jit.elapsed_s / aot.elapsed_s if aot.elapsed_s > 0 else 0
    decode_ratio = aot.decode_tok_s / jit.decode_tok_s if jit.decode_tok_s > 0 else 0

    error = None
    if jit.error and aot.error:
        error = f"both failed: JIT={jit.error}; AOT={aot.error}"
    elif jit.error:
        error = f"JIT: {jit.error}"
    elif aot.error:
        error = f"AOT: {aot.error}"

    device_label = f"gpu:{gpu_layers}" if gpu_layers is not None else device

    return CompareResult(
        name=tc.name, arch=tc.arch, quant=quant, device=device_label, runs=runs,
        jit_passed=jit.passed, jit_elapsed_s=jit.elapsed_s,
        jit_decode_tok_s=jit.decode_tok_s, jit_prefill_tok_s=jit.prefill_tok_s,
        jit_load_ms=jit.load_ms, jit_text=jit.text,
        aot_passed=aot.passed, aot_elapsed_s=aot.elapsed_s,
        aot_decode_tok_s=aot.decode_tok_s, aot_prefill_tok_s=aot.prefill_tok_s,
        aot_load_ms=aot.load_ms, aot_text=aot.text,
        startup_speedup=startup_speedup, decode_ratio=decode_ratio,
        error=error,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_results(results: list[CompareResult]) -> None:
    """Print the comparison table."""
    print()
    print(f"  {'Model':<26} {'Arch':<8} {'Quant':<7} {'Dev':<9} │ "
          f"{'JIT wall':>9} {'JIT dec':>9} │ "
          f"{'AOT wall':>9} {'AOT dec':>9} │ "
          f"{'Startup':>8} {'Decode':>8} │ {'OK'}")
    print(f"  {'':26} {'':8} {'':7} {'':9} │ "
          f"{'(sec)':>9} {'(tok/s)':>9} │ "
          f"{'(sec)':>9} {'(tok/s)':>9} │ "
          f"{'speedup':>8} {'ratio':>8} │")
    print("  " + "─" * 128)

    for r in results:
        if r.error:
            print(f"  {r.name:<26} {r.arch:<8} {r.quant:<7} {r.device:<9} │  ERROR: {r.error}")
            continue

        jit_ok = "+" if r.jit_passed else "X"
        aot_ok = "+" if r.aot_passed else "X"

        startup_str = f"{r.startup_speedup:.2f}x" if r.startup_speedup > 0 else "---"
        decode_str = f"{r.decode_ratio:.2f}x" if r.decode_ratio > 0 else "---"

        print(
            f"  {r.name:<26} {r.arch:<8} {r.quant:<7} {r.device:<9} │ "
            f"{r.jit_elapsed_s:>8.2f}s {r.jit_decode_tok_s:>8.1f}  │ "
            f"{r.aot_elapsed_s:>8.2f}s {r.aot_decode_tok_s:>8.1f}  │ "
            f"{startup_str:>8} {decode_str:>8} │ "
            f"JIT:{jit_ok} AOT:{aot_ok}"
        )

    print("  " + "─" * 128)
    print()
    print("  Startup speedup = JIT wall-clock / AOT wall-clock (>1 = AOT faster overall)")
    print("  Decode ratio    = AOT tok/s / JIT tok/s (>1 = AOT faster decode, <1 = JIT faster)")
    print()

    valid = [r for r in results if not r.error and r.startup_speedup > 0]
    if valid:
        avg_startup = sum(r.startup_speedup for r in valid) / len(valid)
        avg_decode = sum(r.decode_ratio for r in valid) / len(valid)
        jit_correct = sum(1 for r in valid if r.jit_passed)
        aot_correct = sum(1 for r in valid if r.aot_passed)
        print(f"  Summary ({len(valid)} models):")
        print(f"    Avg startup speedup: {avg_startup:.2f}x (AOT vs JIT wall-clock)")
        print(f"    Avg decode ratio:    {avg_decode:.2f}x (AOT/JIT tok/s)")
        print(f"    Correctness:         JIT {jit_correct}/{len(valid)}, AOT {aot_correct}/{len(valid)}")
        print()


def _save_results(path: str, results: list[CompareResult],
                  jit_exe: Path, aot_exe: Path, args: argparse.Namespace) -> None:
    """Export results to JSON."""
    aot_size = aot_exe.stat().st_size if aot_exe.exists() else 0
    export = {
        "label": "aot_comparison",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "system": {
            "cpu": platform.processor() or platform.machine(),
            "cores": os.cpu_count() or 0,
            "os": f"{platform.system()} {platform.release()}",
        },
        "config": {
            "runs": args.runs,
            "device": args.device,
            "aot_binary_size_bytes": aot_size,
        },
        "results": [
            {
                "name": r.name, "arch": r.arch, "quant": r.quant,
                "device": r.device, "runs": r.runs,
                "jit_passed": r.jit_passed,
                "jit_elapsed_s": round(r.jit_elapsed_s, 3),
                "jit_decode_tok_s": round(r.jit_decode_tok_s, 2),
                "jit_prefill_tok_s": round(r.jit_prefill_tok_s, 2),
                "jit_load_ms": round(r.jit_load_ms, 1),
                "aot_passed": r.aot_passed,
                "aot_elapsed_s": round(r.aot_elapsed_s, 3),
                "aot_decode_tok_s": round(r.aot_decode_tok_s, 2),
                "aot_prefill_tok_s": round(r.aot_prefill_tok_s, 2),
                "aot_load_ms": round(r.aot_load_ms, 1),
                "startup_speedup": round(r.startup_speedup, 3),
                "decode_ratio": round(r.decode_ratio, 3),
                "error": r.error,
            }
            for r in results
        ],
    }
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(export, f, indent=2)
    print(f"[save] Results written to {dest}")


def _show_results(path: str) -> int:
    """Load and display results from a JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {path}: {e}", file=sys.stderr)
        return 1

    results_raw = data.get("results", [])
    if not results_raw:
        print(f"No results in {path}", file=sys.stderr)
        return 1

    # Print metadata
    print(f"[show] {path}")
    ts = data.get("timestamp", "")
    if ts:
        print(f"  Time:   {ts}")
    system = data.get("system", {})
    if system:
        print(f"  System: {system.get('cpu', '?')} ({system.get('cores', '?')} cores)")
    config = data.get("config", {})
    aot_mb = config.get("aot_binary_size_bytes", 0) / (1024 * 1024)
    print(f"  Config: runs={config.get('runs', '?')}, aot_binary={aot_mb:.1f}MB")

    results = [
        CompareResult(
            name=r["name"], arch=r["arch"], quant=r["quant"],
            device=r.get("device", "cpu"), runs=r["runs"],
            jit_passed=r["jit_passed"], jit_elapsed_s=r["jit_elapsed_s"],
            jit_decode_tok_s=r["jit_decode_tok_s"], jit_prefill_tok_s=r["jit_prefill_tok_s"],
            jit_load_ms=r.get("jit_load_ms", 0), jit_text="",
            aot_passed=r["aot_passed"], aot_elapsed_s=r["aot_elapsed_s"],
            aot_decode_tok_s=r["aot_decode_tok_s"], aot_prefill_tok_s=r["aot_prefill_tok_s"],
            aot_load_ms=r.get("aot_load_ms", 0), aot_text="",
            startup_speedup=r["startup_speedup"], decode_ratio=r["decode_ratio"],
            error=r.get("error"))
        for r in results_raw
    ]
    _print_results(results)
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_devices(device_arg: str) -> list[str]:
    """Expand 'both' into ['cpu', 'gpu'], otherwise return single-element list."""
    if device_arg == "both":
        return ["cpu", "gpu"]
    return [device_arg]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare JIT vs Native AOT: correctness and performance."
    )
    parser.add_argument("--filter", type=str, default=None,
                        help="Comma-separated name/arch substrings to match")
    parser.add_argument("--size", type=str, default=None,
                        help="Filter by parameter count, e.g. '<3B', '>=1B', '<=500M'")
    parser.add_argument("--list", action="store_true",
                        help="List available test cases and exit")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "gpu", "both"],
                        help="Compute device: cpu, gpu, or both (default: cpu)")
    parser.add_argument("--gpu-layers", type=int, default=None,
                        help="GPU layers for hybrid mode (overrides --device)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Reuse existing publish output (skip dotnet publish)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per binary per model (best-of-N, default: 1)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--show", type=str, default=None,
                        help="Load and display results from JSON (no tests run)")
    args = parser.parse_args()

    # --show mode
    if args.show:
        return _show_results(args.show)

    # Filter test cases
    cases = TEST_CASES
    if args.filter:
        filters = [f.strip().lower() for f in args.filter.split(",")]
        cases = [
            tc for tc in cases
            if any(f in tc.name.lower() or f in tc.arch.lower() for f in filters)
        ]
        if not cases:
            print(f"No test cases match filter '{args.filter}'.")
            return 1

    if args.size:
        size_pred = _parse_size_filter(args.size)
        cases = [
            tc for tc in cases
            if (b := _extract_param_billions(tc.name)) is not None and size_pred(b)
        ]
        if not cases:
            print(f"No test cases match size filter '{args.size}'.")
            return 1

    # List mode
    if args.list:
        print(f"{'Name':<35} {'Arch':<10} {'Quant':<8} {'Size':<8} {'Cached':<8}")
        print("-" * 75)
        for tc in cases:
            cached = "yes" if _model_is_cached(tc) else "no"
            quant = tc.quant or "default"
            b = _extract_param_billions(tc.name)
            size = f"{b:.1f}B" if b is not None and b >= 1 else (f"{b*1000:.0f}M" if b else "?")
            print(f"{tc.name:<35} {tc.arch:<10} {quant:<8} {size:<8} {cached:<8}")
        return 0

    # Only run cached models
    cases = [tc for tc in cases if _model_is_cached(tc)]
    if not cases:
        print("No cached models found. Download models first via test_models.py --download.")
        return 1

    # Resolve device list
    if args.gpu_layers is not None:
        devices = ["hybrid"]
    else:
        devices = _resolve_devices(args.device)

    # Build phase
    if args.skip_build:
        jit_exe = _jit_publish_dir() / _exe_name(aot=False)
        aot_exe = _aot_publish_dir() / _exe_name(aot=True)
        if not jit_exe.exists():
            # Try non-publish path
            jit_exe = CLI_PROJECT / "bin" / "Release" / "net10.0" / _exe_name(aot=False)
        for label, exe in [("JIT", jit_exe), ("AOT", aot_exe)]:
            if not exe.exists():
                print(f"[skip-build] {label} binary not found: {exe}")
                print("  Run without --skip-build first, or build manually.")
                return 1
            size_mb = exe.stat().st_size / (1024 * 1024)
            print(f"[skip-build] {label}: {exe} ({size_mb:.1f}MB)")
    else:
        jit_exe = _build_jit()
        aot_exe = _build_aot()

    total_runs = len(cases) * len(devices)
    print()
    print(f"JIT vs AOT comparison: {len(cases)} models, {len(devices)} device(s) "
          f"({', '.join(devices)}), {args.runs} run(s) per binary")
    print()

    # Run comparisons
    results: list[CompareResult] = []
    idx = 0

    for tc in cases:
        try:
            model_path = resolve_model(tc.repo, tc.quant, quiet=True)
        except SystemExit:
            print(f"  SKIP {tc.name} — model resolution failed")
            continue

        for device in devices:
            idx += 1
            gpu_layers = args.gpu_layers if device == "hybrid" else None

            # For GPU mode, compute full offload layer count from GGUF
            if device == "gpu":
                num_layers = tc.layers or _get_gguf_layers(model_path)
                gpu_layers = num_layers if num_layers > 0 else None
                if gpu_layers is None:
                    # Can't determine layers, fall back to --device gpu
                    gpu_layers = None

            dev_label = f"gpu-layers={gpu_layers}" if gpu_layers else device
            print(f"[{idx}/{total_runs}] {tc.name} ({tc.arch}, {tc.quant or 'default'}) "
                  f"on {dev_label}...")

            r = _run_comparison(
                jit_exe, aot_exe, model_path, tc, runs=args.runs,
                device=device if gpu_layers is None else "gpu",
                gpu_layers=gpu_layers,
            )
            results.append(r)

            if r.error:
                print(f"  ERROR: {r.error}")
            else:
                print(f"  JIT: {r.jit_elapsed_s:.2f}s wall, {r.jit_decode_tok_s:.1f} tok/s"
                      f"  {'PASS' if r.jit_passed else 'FAIL'}")
                print(f"  AOT: {r.aot_elapsed_s:.2f}s wall, {r.aot_decode_tok_s:.1f} tok/s"
                      f"  {'PASS' if r.aot_passed else 'FAIL'}")
                print(f"  -> startup {r.startup_speedup:.2f}x, decode {r.decode_ratio:.2f}x")

    # Summary table
    _print_results(results)

    # Save
    if args.save:
        _save_results(args.save, results, jit_exe, aot_exe, args)

    # Exit code: 1 if any correctness failures
    failures = sum(1 for r in results if not r.error and (not r.jit_passed or not r.aot_passed))
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
