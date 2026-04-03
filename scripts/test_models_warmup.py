#!/usr/bin/env python3
"""
test_models_warmup.py — Measure warm-up effect on first-request latency.

For each cached model, starts the dotLLM server twice (with and without --no-warmup),
sends a single chat completion request, and compares first-request latency. Shows how
much JIT compilation and CUDA kernel loading overhead the warm-up eliminates.

Usage:
    python scripts/test_models_warmup.py                      # run all cached models
    python scripts/test_models_warmup.py --filter qwen        # only models matching "qwen"
    python scripts/test_models_warmup.py --device gpu          # GPU mode
    python scripts/test_models_warmup.py --list                # show available test cases
    python scripts/test_models_warmup.py --warmup-iterations 5 # override iteration count
    python scripts/test_models_warmup.py --save results.json   # save results to JSON
    python scripts/test_models_warmup.py --show results.json   # display saved results
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
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Reuse test case definitions and model resolution from test_models
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_models import TEST_CASES, TestCase, _model_is_cached
from bench_compare import resolve_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PORT = 18090  # avoid conflict with test_server.py (18080)
PROMPT = "The capital of France is"
MAX_TOKENS = 5


# ---------------------------------------------------------------------------
# Server management (adapted from test_server.py)
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _start_server(model_path: str, port: int, device: str = "cpu",
                  no_warmup: bool = False,
                  warmup_iterations: int = 3,
                  gpu_layers: int | None = None,
                  cache_type_k: str | None = None,
                  cache_type_v: str | None = None) -> subprocess.Popen:
    """Start dotLLM serve as a subprocess."""
    cmd = [
        "dotnet", "run",
        "--project", str(_repo_root() / "src" / "DotLLM.Cli"),
        "-c", "Release", "--",
        "serve", model_path,
        "--port", str(port),
        "--no-browser",
        "--no-prompt-cache",  # disable prompt cache to isolate warm-up effect
    ]
    if gpu_layers is not None:
        cmd += ["--gpu-layers", str(gpu_layers)]
    else:
        cmd += ["--device", device]
    if no_warmup:
        cmd += ["--no-warmup"]
    else:
        cmd += ["--warmup-iterations", str(warmup_iterations)]
    if cache_type_k:
        cmd += ["--cache-type-k", cache_type_k]
    if cache_type_v:
        cmd += ["--cache-type-v", cache_type_v]

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )


def _wait_for_ready(base_url: str, timeout: float = 300) -> float:
    """Wait for /ready to return 200. Returns time spent waiting (seconds)."""
    start = time.monotonic()
    deadline = start + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/ready")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return time.monotonic() - start
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Server not ready after {timeout}s")


def _send_request(base_url: str, prompt: str, max_tokens: int) -> tuple[float, str]:
    """Send a single chat completion request. Returns (elapsed_ms, generated_text)."""
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as resp:
        elapsed_ms = (time.monotonic() - start) * 1000
        data = json.loads(resp.read().decode("utf-8"))

    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return elapsed_ms, text


def _stop_server(proc: subprocess.Popen) -> str:
    """Stop server and return captured stdout."""
    proc.terminate()
    try:
        stdout, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate(timeout=5)
    return stdout or ""


def _extract_warmup_time(stdout: str) -> float | None:
    """Extract warm-up total time from server stdout (e.g. 'Warm-up complete in 1970ms')."""
    for line in stdout.splitlines():
        if "Warm-up complete in" in line:
            # "[dotllm] Warm-up complete in 1970ms"
            try:
                ms_str = line.split("in")[1].strip().rstrip("ms")
                return float(ms_str)
            except (IndexError, ValueError):
                pass
    return None


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

@dataclass
class WarmupResult:
    name: str
    arch: str
    quant: str
    device: str
    # With warm-up
    startup_ms: float       # time to /ready (includes model load + warm-up)
    warmup_ms: float        # warm-up portion (from server log)
    first_req_ms: float     # first request latency
    first_req_text: str
    # Without warm-up
    cold_startup_ms: float  # time to /ready (model load only)
    cold_first_req_ms: float  # first request latency (pays JIT penalty)
    cold_first_req_text: str
    # Derived
    speedup: float          # cold_first_req_ms / first_req_ms
    error: str | None = None


def _run_warmup_test(
    model_path: str,
    tc: TestCase,
    port: int,
    device: str,
    warmup_iterations: int,
    gpu_layers: int | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
) -> WarmupResult:
    """Run warm-up comparison for a single model: warm server, then cold server."""
    base_url = f"http://localhost:{port}"
    quant = tc.quant or "default"

    # --- Run WITH warm-up ---
    proc = _start_server(str(model_path), port, device,
                         no_warmup=False, warmup_iterations=warmup_iterations,
                         gpu_layers=gpu_layers,
                         cache_type_k=cache_type_k, cache_type_v=cache_type_v)
    try:
        startup_ms = _wait_for_ready(base_url) * 1000
        first_req_ms, first_req_text = _send_request(base_url, PROMPT, MAX_TOKENS)
    except Exception as e:
        stdout = _stop_server(proc)
        return WarmupResult(
            name=tc.name, arch=tc.arch, quant=quant, device=device,
            startup_ms=0, warmup_ms=0, first_req_ms=0, first_req_text="",
            cold_startup_ms=0, cold_first_req_ms=0, cold_first_req_text="",
            speedup=0, error=f"warm: {e}")
    stdout = _stop_server(proc)
    warmup_ms = _extract_warmup_time(stdout) or 0

    # Brief pause between server instances
    time.sleep(1)

    # --- Run WITHOUT warm-up ---
    proc = _start_server(str(model_path), port, device,
                         no_warmup=True,
                         gpu_layers=gpu_layers,
                         cache_type_k=cache_type_k, cache_type_v=cache_type_v)
    try:
        cold_startup_ms = _wait_for_ready(base_url) * 1000
        cold_first_req_ms, cold_first_req_text = _send_request(base_url, PROMPT, MAX_TOKENS)
    except Exception as e:
        _stop_server(proc)
        return WarmupResult(
            name=tc.name, arch=tc.arch, quant=quant, device=device,
            startup_ms=startup_ms, warmup_ms=warmup_ms,
            first_req_ms=first_req_ms, first_req_text=first_req_text,
            cold_startup_ms=0, cold_first_req_ms=0, cold_first_req_text="",
            speedup=0, error=f"cold: {e}")
    _stop_server(proc)

    speedup = cold_first_req_ms / first_req_ms if first_req_ms > 0 else 0

    return WarmupResult(
        name=tc.name, arch=tc.arch, quant=quant, device=device,
        startup_ms=startup_ms, warmup_ms=warmup_ms,
        first_req_ms=first_req_ms, first_req_text=first_req_text,
        cold_startup_ms=cold_startup_ms,
        cold_first_req_ms=cold_first_req_ms, cold_first_req_text=cold_first_req_text,
        speedup=speedup)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_results(results: list[WarmupResult]) -> None:
    """Print the comparison table."""
    print()
    print(f"{'Model':<28} {'Arch':<8} {'Quant':<7} "
          f"{'Startup':>9} {'Warmup':>9} {'1st Req':>9}  │  "
          f"{'Startup':>9} {'1st Req':>9}  │  {'Speedup':>7}")
    print(f"{'':28} {'':8} {'':7} "
          f"{'(warm)':>9} {'time':>9} {'(warm)':>9}  │  "
          f"{'(cold)':>9} {'(cold)':>9}  │  ")
    print("─" * 130)

    for r in results:
        if r.error:
            print(f"{r.name:<28} {r.arch:<8} {r.quant:<7}  ERROR: {r.error}")
            continue

        speedup_str = f"{r.speedup:.1f}x" if r.speedup > 0 else "—"
        print(
            f"{r.name:<28} {r.arch:<8} {r.quant:<7} "
            f"{r.startup_ms:>8.0f}ms {r.warmup_ms:>8.0f}ms {r.first_req_ms:>8.0f}ms  │  "
            f"{r.cold_startup_ms:>8.0f}ms {r.cold_first_req_ms:>8.0f}ms  │  {speedup_str:>7}"
        )

    print("─" * 130)
    print()
    print("  Startup = time to /ready    Warmup = warm-up portion (from log)")
    print("  1st Req = first chat completion latency    Speedup = cold / warm 1st req")
    print()

    # Summary
    valid = [r for r in results if not r.error and r.speedup > 0]
    if valid:
        avg_speedup = sum(r.speedup for r in valid) / len(valid)
        avg_warm = sum(r.first_req_ms for r in valid) / len(valid)
        avg_cold = sum(r.cold_first_req_ms for r in valid) / len(valid)
        print(f"  Average: warm 1st req = {avg_warm:.0f}ms, cold 1st req = {avg_cold:.0f}ms, "
              f"speedup = {avg_speedup:.1f}x")


def _save_results(path: str, results: list[WarmupResult], args: argparse.Namespace) -> None:
    """Export results to JSON."""
    export = {
        "label": "warmup_comparison",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "system": {
            "cpu": platform.processor() or platform.machine(),
            "cores": os.cpu_count() or 0,
            "os": f"{platform.system()} {platform.release()}",
        },
        "config": {
            "device": args.device,
            "warmup_iterations": args.warmup_iterations,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
        },
        "results": [
            {
                "name": r.name, "arch": r.arch, "quant": r.quant, "device": r.device,
                "startup_ms": round(r.startup_ms, 1),
                "warmup_ms": round(r.warmup_ms, 1),
                "first_req_ms": round(r.first_req_ms, 1),
                "cold_startup_ms": round(r.cold_startup_ms, 1),
                "cold_first_req_ms": round(r.cold_first_req_ms, 1),
                "speedup": round(r.speedup, 2),
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
    print(f"  Config: device={config.get('device', '?')}, "
          f"warmup_iterations={config.get('warmup_iterations', '?')}")

    results = [
        WarmupResult(
            name=r["name"], arch=r["arch"], quant=r["quant"], device=r["device"],
            startup_ms=r["startup_ms"], warmup_ms=r["warmup_ms"],
            first_req_ms=r["first_req_ms"], first_req_text="",
            cold_startup_ms=r["cold_startup_ms"],
            cold_first_req_ms=r["cold_first_req_ms"], cold_first_req_text="",
            speedup=r["speedup"], error=r.get("error"))
        for r in results_raw
    ]
    _print_results(results)
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure warm-up effect on first-request latency across models."
    )
    parser.add_argument("--filter", type=str, default=None,
                        help="Comma-separated name/arch substrings to match")
    parser.add_argument("--list", action="store_true",
                        help="List available test cases and exit")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "gpu"],
                        help="Compute device (default: cpu)")
    parser.add_argument("--warmup-iterations", type=int, default=3,
                        help="Number of warm-up iterations (default: 3)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Server port (default: {DEFAULT_PORT})")
    parser.add_argument("--cache-type-k", type=str, default=None,
                        help="KV-cache key quantization type")
    parser.add_argument("--cache-type-v", type=str, default=None,
                        help="KV-cache value quantization type")
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

    # List mode
    if args.list:
        print(f"{'Name':<35} {'Arch':<10} {'Quant':<8} {'Cached':<8} Notes")
        print("-" * 105)
        for tc in cases:
            cached = "yes" if _model_is_cached(tc) else "no"
            quant = tc.quant or "default"
            print(f"{tc.name:<35} {tc.arch:<10} {quant:<8} {cached:<8} {tc.notes}")
        return 0

    # Only run cached models
    cases = [tc for tc in cases if _model_is_cached(tc)]
    if not cases:
        print("No cached models found. Download models first via test_models.py --download.")
        return 1

    print(f"Warm-up comparison: {len(cases)} models, device={args.device}, "
          f"warmup_iterations={args.warmup_iterations}")
    print(f"Each model tested twice: with warm-up, then without (cold start)")
    print()

    results: list[WarmupResult] = []

    for i, tc in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {tc.name} ({tc.arch}, {tc.quant or 'default'})...")

        try:
            model_path = resolve_model(tc.repo, tc.quant, quiet=True)
        except SystemExit:
            print(f"  SKIP — model resolution failed")
            continue

        r = _run_warmup_test(
            str(model_path), tc, args.port, args.device,
            warmup_iterations=args.warmup_iterations,
            cache_type_k=args.cache_type_k,
            cache_type_v=args.cache_type_v,
        )
        results.append(r)

        if r.error:
            print(f"  ERROR: {r.error}")
        else:
            print(f"  warm: startup={r.startup_ms:.0f}ms (warmup={r.warmup_ms:.0f}ms), "
                  f"1st req={r.first_req_ms:.0f}ms")
            print(f"  cold: startup={r.cold_startup_ms:.0f}ms, "
                  f"1st req={r.cold_first_req_ms:.0f}ms → {r.speedup:.1f}x speedup")

    # Summary table
    _print_results(results)

    # Save
    if args.save:
        _save_results(args.save, results, args)

    errors = sum(1 for r in results if r.error)
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
