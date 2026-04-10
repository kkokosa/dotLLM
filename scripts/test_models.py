#!/usr/bin/env python3
"""
test_models.py — Quick correctness smoke test across model architectures.

Runs dotLLM CLI with greedy decoding on predefined prompts and checks that the
expected substring appears in the generated output. Designed for local testing
after architecture changes — NOT for CI (models are too large).

Usage:
    python scripts/test_models.py                          # run all cached models
    python scripts/test_models.py --filter phi             # only models matching "phi"
    python scripts/test_models.py --filter qwen,mistral    # multiple filters
    python scripts/test_models.py --list                   # show available test cases
    python scripts/test_models.py --download               # download missing models
    python scripts/test_models.py --device gpu             # GPU-only
    python scripts/test_models.py --device both            # run each test on CPU then GPU
    python scripts/test_models.py --cache-type-k q8_0 --cache-type-v q4_0  # KV-cache quantization
    python scripts/test_models.py --cache-type-k q8_0,q4_0 --cache-type-v q8_0,q4_0  # cartesian product
    python scripts/test_models.py --save results.json      # save results to JSON
    python scripts/test_models.py --show results.json      # display saved results
    python scripts/test_models.py --compare before.json after.json  # diff two saved runs
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
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

# Reuse model resolution and hybrid utilities from bench_compare
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_compare import resolve_model, parse_hybrid_modes, compute_gpu_layers, _get_gguf_layers


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single model correctness test."""
    name: str
    repo: str
    quant: str | None
    arch: str
    prompt: str
    expected: str  # substring that must appear in generated output
    layers: int = 0  # transformer block count (0 = unknown, read from GGUF at runtime)
    max_tokens: int = 2
    notes: str = ""


# Ordered by model size (smallest first) for faster feedback
TEST_CASES: list[TestCase] = [
    # --- Llama architecture ---
    TestCase(
        name="SmolLM-135M",
        repo="QuantFactory/SmolLM-135M-GGUF",
        quant="Q8_0",
        arch="Llama",
        prompt="The capital of France is",
        expected="Paris",
        layers=30,
        notes="baseline Llama arch, SentencePiece",
    ),
    TestCase(
        name="SmolLM2-135M-Instruct",
        repo="bartowski/SmolLM2-135M-Instruct-GGUF",
        quant="Q8_0",
        arch="Llama",
        prompt="The capital of France is",
        expected="Paris",
        layers=30,
        notes="SmolLM2, SentencePiece",
    ),
    TestCase(
        name="Llama-3.2-1B-Instruct-Q4",
        repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
        quant="Q4_K_M",
        arch="Llama",
        prompt="The capital of France is",
        expected="Paris",
        layers=16,
        notes="Llama 3.2, tiktoken, Q4_K_M",
    ),
    TestCase(
        name="Llama-3.2-1B-Instruct-Q8",
        repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
        quant="Q8_0",
        arch="Llama",
        prompt="The capital of France is",
        expected="Paris",
        layers=16,
        notes="Llama 3.2, tiktoken, Q8_0",
    ),
    TestCase(
        name="Llama-3.2-3B-Instruct-Q4",
        repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        quant="Q4_K_M",
        arch="Llama",
        prompt="The capital of France is",
        expected="Paris",
        layers=28,
        notes="Llama 3.2 3B, tiktoken, Q4_K_M",
    ),
    TestCase(
        name="Llama-3.2-3B-Instruct-Q8",
        repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        quant="Q8_0",
        arch="Llama",
        prompt="The capital of France is",
        expected="Paris",
        layers=28,
        notes="Llama 3.2 3B, tiktoken, Q8_0",
    ),
    TestCase(
        name="Bielik-1.5B-Instruct",
        repo="speakleash/Bielik-1.5B-v3.0-Instruct-GGUF",
        quant="Q8_0",
        arch="Llama",
        prompt="Stolicą Polski jest",
        expected="Warszawa",
        layers=28,
        max_tokens=3,
        notes="Polish 1.5B, Llama arch",
    ),
    TestCase(
        name="Bielik-11B-Instruct",
        repo="speakleash/Bielik-11B-v3.0-Instruct-GGUF",
        quant="Q4_K_M",
        arch="Llama",
        prompt="Stolicą Polski jest",
        expected="Warszawa",
        layers=48,
        max_tokens=3,
        notes="Polish 11B, Llama arch, Q4_K_M",
    ),

    # --- Qwen architecture ---
    TestCase(
        name="Qwen2.5-0.5B-Instruct",
        repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        quant="Q8_0",
        arch="Qwen",
        prompt="The capital of France is",
        expected="Paris",
        layers=24,
        notes="tiktoken, tied embeddings, Q/K biases",
    ),
    TestCase(
        name="Qwen3-0.6B",
        repo="Qwen/Qwen3-0.6B-GGUF",
        quant="Q8_0",
        arch="Qwen",
        prompt="The capital of France is",
        expected="Paris",
        layers=28,
        notes="QK-norms, explicit head_dim",
    ),

    # --- Phi architecture ---
    TestCase(
        name="Phi-3-mini-4k-instruct",
        repo="microsoft/Phi-3-mini-4k-instruct-gguf",
        quant=None,
        arch="Phi",
        prompt="The capital of France is",
        expected="Paris",
        layers=32,
        notes="fused QKV + fused gate_up FFN, phi3 arch",
    ),
    TestCase(
        name="Phi-4-mini-instruct",
        repo="unsloth/Phi-4-mini-instruct-GGUF",
        quant="Q8_0",
        arch="Phi",
        prompt="The capital of France is",
        expected="Paris",
        layers=32,
        notes="Phi-4 mini",
    ),

    # --- Mistral architecture ---
    TestCase(
        name="Ministral-3-3B-Instruct",
        repo="mistralai/Ministral-3-3B-Instruct-2512-GGUF",
        quant=None,
        arch="Mistral",
        prompt="The capital of France is",
        expected="Paris",
        layers=24,
        notes="mistral3 arch string",
    ),
    TestCase(
        name="Mistral-7B-Instruct-v0.3-Q4",
        repo="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        quant="Q4_K_M",
        arch="Mistral",
        prompt="The capital of France is",
        expected="Paris",
        layers=32,
        notes="sliding window, Q4_K_M",
    ),
    TestCase(
        name="Mistral-7B-Instruct-v0.3-Q8",
        repo="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        quant="Q8_0",
        arch="Mistral",
        prompt="The capital of France is",
        expected="Paris",
        layers=32,
        notes="sliding window, Q8_0",
    ),
]


def _find_cli() -> Path:
    """Find the dotLLM CLI executable (works on Windows, Linux, and macOS)."""
    repo_root = Path(__file__).resolve().parent.parent
    bin_dir = repo_root / "src" / "DotLLM.Cli" / "bin"
    for config in ("Release", "Debug"):
        for ext in (".exe", ""):
            p = bin_dir / config / "net10.0" / f"DotLLM.Cli{ext}"
            if p.exists():
                return p
    return Path("dotnet")  # fallback to dotnet run


@dataclass
class TestResult:
    """Result of a single test run."""
    passed: bool
    detail: str
    elapsed: float
    decode_tok_s: float = 0.0
    prefill_tok_s: float = 0.0


def _run_test(cli: Path, model_path: Path, tc: TestCase, device: str = "cpu",
              gpu_layers: int | None = None,
              cache_type_k: str | None = None,
              cache_type_v: str | None = None,
              cache_window: int = 0) -> TestResult:
    """
    Run a single test case with --json output.

    Args:
        gpu_layers: If set, pass --gpu-layers N to the CLI (hybrid mode).
        cache_type_k: KV-cache key quantization type (e.g. "q8_0").
        cache_type_v: KV-cache value quantization type (e.g. "q4_0").
        cache_window: Mixed-precision window size.
    """
    if cli.name == "dotnet":
        cmd = [
            str(cli), "run",
            "--project", str(Path(__file__).resolve().parent.parent / "src" / "DotLLM.Cli"),
            "-c", "Release", "--",
        ]
    else:
        cmd = [str(cli)]

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
    if cache_type_k:
        cmd += ["--cache-type-k", cache_type_k]
    if cache_type_v:
        cmd += ["--cache-type-v", cache_type_v]
    if cache_window > 0:
        cmd += ["--cache-window", str(cache_window)]

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=600,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return TestResult(False, "TIMEOUT (600s)", time.monotonic() - start)
    except FileNotFoundError:
        return TestResult(False, f"CLI not found: {cli}", time.monotonic() - start)

    elapsed = time.monotonic() - start

    if result.returncode != 0:
        error_text = result.stderr.strip() or result.stdout.strip()
        for line in error_text.splitlines():
            if "Error:" in line:
                return TestResult(False, line.strip(), elapsed)
        return TestResult(False, f"exit code {result.returncode}: {error_text[:200]}", elapsed)

    # Extract JSON from stdout
    raw = result.stdout.strip()
    json_start = raw.find("{")
    if json_start < 0:
        return TestResult(False, f"no JSON in output: {raw[:200]}", elapsed)
    try:
        data = json.loads(raw[json_start:])
    except json.JSONDecodeError:
        return TestResult(False, f"invalid JSON: {raw[:200]}", elapsed)

    generated_text = data.get("text", "")
    timings = data.get("timings", {})
    decode_tok_s = timings.get("decode_tok_s", 0) or 0
    prefill_tok_s = timings.get("prefill_tok_s", 0) or 0
    tok_s = decode_tok_s or prefill_tok_s

    if tc.expected in generated_text:
        detail = f"{generated_text.strip()[:50]}  ({tok_s:.1f} tok/s)"
        return TestResult(True, detail, elapsed, decode_tok_s, prefill_tok_s)
    else:
        detail = f"expected '{tc.expected}' not in: '{generated_text[:100]}'"
        return TestResult(False, detail, elapsed, decode_tok_s, prefill_tok_s)


def _model_is_cached(tc: TestCase) -> bool:
    """Check if the model is already downloaded."""
    from bench_compare import _default_models_dir, _apply_quant_filter

    if tc.repo.endswith(".gguf"):
        return Path(tc.repo).exists()

    models_dir = _default_models_dir()
    repo_dir = models_dir / tc.repo.replace("/", os.sep)
    if not repo_dir.exists():
        return False
    cached = [f.name for f in repo_dir.iterdir() if f.suffix == ".gguf"]
    cached = _apply_quant_filter(cached, tc.quant)
    return len(cached) >= 1


def _parse_cache_types(value: str | None) -> list[str]:
    """Parse comma-separated cache type values, returning list (empty if None/f32-only)."""
    if not value:
        return []
    types = [t.strip().lower() for t in value.split(",") if t.strip()]
    return types


def _cache_combos(k_types: list[str], v_types: list[str]) -> list[tuple[str, str]]:
    """Generate cartesian product of K/V cache type combinations.

    Returns list of (k_type, v_type) tuples. Empty list means no quantized cache runs.
    """
    if not k_types and not v_types:
        return []
    # Default to f32 if only one side is specified
    if not k_types:
        k_types = ["f32"]
    if not v_types:
        v_types = ["f32"]
    combos = list(product(k_types, v_types))
    # Filter out (f32, f32) — that's the baseline, not a cache quant run
    combos = [(k, v) for k, v in combos if k != "f32" or v != "f32"]
    return combos


def _cache_label(k: str, v: str) -> str:
    """Short label for a cache type combo."""
    if k == v:
        return f"KV:{k}"
    return f"KV:{k}/{v}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quick correctness smoke test across model architectures."
    )
    parser.add_argument("--filter", type=str, default=None,
                        help="Comma-separated name/arch substrings to match (e.g. 'phi,qwen')")
    parser.add_argument("--list", action="store_true",
                        help="List available test cases and exit")
    parser.add_argument("--download", action="store_true",
                        help="Download missing models before testing")
    parser.add_argument("--cached-only", action="store_true",
                        help="Only run tests for models already downloaded")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "gpu", "both"],
                        help="Compute device: cpu (default), gpu, or both")
    parser.add_argument("--hybrid-modes", type=str, default=None,
                        help="Comma-separated fractions (0-1) of layers to offload to GPU. "
                             "E.g. '0.25,0.5' runs hybrid mode tests alongside regular device tests.")
    parser.add_argument("--cache-type-k", type=str, default=None,
                        help="KV-cache key quantization: q8_0, q4_0, or comma-separated for cartesian product "
                             "(e.g. 'q8_0,q4_0'). Adds runs alongside baseline.")
    parser.add_argument("--cache-type-v", type=str, default=None,
                        help="KV-cache value quantization: q8_0, q4_0, or comma-separated for cartesian product.")
    parser.add_argument("--cache-window", type=int, default=0,
                        help="Mixed-precision window size for KV-cache quantization (default: 0 = all quantized).")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to a JSON file (e.g. --save results.json). "
                             "File includes both flat per-run results and a pivoted summary.")
    parser.add_argument("--show", type=str, default=None,
                        help="Load and display results from a JSON file (no tests run).")
    parser.add_argument("--compare", type=str, nargs=2, default=None,
                        metavar=("BEFORE", "AFTER"),
                        help="Compare two saved JSON files: show throughput deltas per model per mode.")
    args = parser.parse_args()

    # --show mode: load and display, then exit
    if args.show:
        return _show_results(args.show)

    # --compare mode: diff two JSON files, then exit
    if args.compare:
        return _compare_results(args.compare[0], args.compare[1])

    # Default to --cached-only when no explicit mode is given
    if not args.download and not args.list:
        args.cached_only = True

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
            print(f"Available: {', '.join(tc.name for tc in TEST_CASES)}")
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

    # Filter to cached-only if requested
    if args.cached_only:
        cases = [tc for tc in cases if _model_is_cached(tc)]
        if not cases:
            print("No cached models found. Run with --download to fetch them.")
            return 1

    # Find CLI
    cli = _find_cli()
    if cli.name == "dotnet":
        print("[cli] No prebuilt CLI found, will use 'dotnet run' (slower startup)")
    else:
        print(f"[cli] {cli}")

    # Determine which devices to test
    devices = ["cpu", "gpu"] if args.device == "both" else [args.device]

    # Parse hybrid modes
    hybrid_modes = parse_hybrid_modes(args.hybrid_modes)

    # Parse cache type combos
    k_types = _parse_cache_types(args.cache_type_k)
    v_types = _parse_cache_types(args.cache_type_v)
    cache_combos = _cache_combos(k_types, v_types)
    cache_window = args.cache_window

    # Build run configs
    has_hybrid = len(hybrid_modes) > 0
    has_cache = len(cache_combos) > 0
    multi_device = len(devices) > 1
    show_extra = multi_device or has_hybrid or has_cache

    # Run tests
    print()
    if show_extra:
        print(f"{'Test':<30} {'Device':<6} {'Mode':<16} {'Result':<8} {'Time':>8}  Details")
    else:
        print(f"{'Test':<30} {'Result':<8} {'Time':>8}  Details")
    sep_width = 115 if show_extra else 70
    print("=" * sep_width)

    passed = 0
    failed = 0
    skipped = 0

    # Collect throughput data: { tc.name: { col_key: (decode_tok_s, passed) } }
    throughput_data: dict[str, dict[str, tuple[float, bool]]] = {}
    # Ordered column keys for summary table
    mode_keys: list[str] = []
    for device in devices:
        mode_keys.append(device)
    for frac in hybrid_modes:
        mode_keys.append(f"h{int(frac * 100)}%")
    for device in devices:
        for k, v in cache_combos:
            key = f"{device}+{_cache_label(k, v)}" if multi_device else _cache_label(k, v)
            mode_keys.append(key)

    # Collect all results for --save
    all_results: list[dict] = []

    def _clean_detail(detail: str, prompt: str) -> str:
        if prompt in detail:
            detail = detail[detail.index(prompt) + len(prompt):]
        for marker in ["Generation Complete", "Performance", "Prefill"]:
            if marker in detail:
                detail = detail[:detail.index(marker)]
        return detail.strip()[:55]

    def _record_result(tc: TestCase, col_key: str, device: str, r: TestResult,
                       cache_k: str = "f32", cache_v: str = "f32") -> None:
        all_results.append({
            "name": tc.name, "arch": tc.arch, "quant": tc.quant or "default",
            "mode": col_key, "device": device,
            "cache_type_k": cache_k, "cache_type_v": cache_v,
            "passed": r.passed, "detail": r.detail,
            "elapsed": round(r.elapsed, 3),
            "decode_tok_s": round(r.decode_tok_s, 2),
            "prefill_tok_s": round(r.prefill_tok_s, 2),
        })

    def _print_row(tc_name: str, device: str, mode: str, r: TestResult, prompt: str = "") -> None:
        nonlocal passed, failed
        time_str = f"{r.elapsed:.1f}s"
        status = "PASS" if r.passed else "FAIL"
        detail = _clean_detail(r.detail, prompt) if r.passed else r.detail[:55]
        if r.passed:
            passed += 1
        else:
            failed += 1
        if show_extra:
            print(f"{tc_name:<30} {device:<6} {mode:<16} {status:<8} {time_str:>8}  {detail}")
        else:
            print(f"{tc_name:<30} {status:<8} {time_str:>8}  {detail}")

    for tc in cases:
        # Check if model is available
        if not _model_is_cached(tc) and not args.download:
            skipped += 1
            if show_extra:
                print(f"{tc.name:<30} {'':6} {'':16} {'SKIP':<8} {'':>8}  not cached (use --download)")
            else:
                print(f"{tc.name:<30} {'SKIP':<8} {'':>8}  not cached (use --download)")
            continue

        # Resolve model (downloads if --download and not cached)
        try:
            model_path = resolve_model(tc.repo, tc.quant, quiet=True)
        except SystemExit:
            failed += 1
            if show_extra:
                print(f"{tc.name:<30} {'':6} {'':16} {'FAIL':<8} {'':>8}  model resolution failed")
            else:
                print(f"{tc.name:<30} {'FAIL':<8} {'':>8}  model resolution failed")
            continue

        # Determine layer count for hybrid modes
        num_layers = tc.layers
        if hybrid_modes and num_layers == 0:
            num_layers = _get_gguf_layers(str(model_path))
            if num_layers == 0:
                print(f"  [hybrid] WARNING: Could not determine layer count for {tc.name}, skipping hybrid")

        row: dict[str, tuple[float, bool]] = {}

        # Baseline device runs (no cache quantization)
        for device in devices:
            r = _run_test(cli, model_path, tc, device=device)
            _print_row(tc.name, device, "", r, tc.prompt)
            row[device] = (r.decode_tok_s, r.passed)
            _record_result(tc, device, device, r)

        # Hybrid mode runs
        if hybrid_modes and num_layers > 0:
            for frac in hybrid_modes:
                gl = compute_gpu_layers(num_layers, frac)
                pct = int(frac * 100)
                col_key = f"h{pct}%"
                mode_str = f"h{pct}%({gl}L)"

                r = _run_test(cli, model_path, tc, gpu_layers=gl)
                _print_row(tc.name, "hybrid", mode_str, r, tc.prompt)
                row[col_key] = (r.decode_tok_s, r.passed)
                _record_result(tc, col_key, "hybrid", r)

        # Cache type combo runs — on ALL devices
        for device in devices:
            for cache_k, cache_v in cache_combos:
                label = _cache_label(cache_k, cache_v)
                col_key = f"{device}+{label}" if multi_device else label

                r = _run_test(cli, model_path, tc, device=device,
                              cache_type_k=cache_k, cache_type_v=cache_v,
                              cache_window=cache_window)
                _print_row(tc.name, device, label, r, tc.prompt)
                row[col_key] = (r.decode_tok_s, r.passed)
                _record_result(tc, col_key, device, r, cache_k, cache_v)

        if row:
            throughput_data[tc.name] = row

    # Summary
    print("=" * sep_width)
    total = passed + failed + skipped
    print(f"\n{passed}/{total} passed, {failed} failed, {skipped} skipped")

    # Throughput summary table (only when multiple modes were tested)
    if len(mode_keys) > 1 and throughput_data:
        _print_throughput_summary(throughput_data, mode_keys, cases, hybrid_modes)

    # Save results to JSON
    if args.save:
        _save_results(args.save, all_results, args)

    return 1 if failed > 0 else 0


def _build_summary(results: list[dict]) -> dict:
    """Pivot flat per-run results into a summary keyed by model and mode.

    Returns:
        {
            "modes": ["gpu", "KV:q8_0", ...],           # in first-seen order
            "models": {
                "SmolLM-135M": {
                    "quant": "Q8_0",
                    "arch": "Llama",
                    "throughput": {
                        "gpu": {"decode_tok_s": 43.9, "prefill_tok_s": ..., "passed": true},
                        ...
                    }
                },
                ...
            }
        }
    """
    modes: list[str] = []
    models: dict[str, dict] = {}
    for r in results:
        mode = r.get("mode", "?")
        if mode not in modes:
            modes.append(mode)
        name = r.get("name", "?")
        if name not in models:
            models[name] = {
                "quant": r.get("quant", "?"),
                "arch": r.get("arch", ""),
                "throughput": {},
            }
        models[name]["throughput"][mode] = {
            "decode_tok_s": r.get("decode_tok_s", 0),
            "prefill_tok_s": r.get("prefill_tok_s", 0),
            "passed": r.get("passed", False),
        }
    return {"modes": modes, "models": models}


def _save_results(path: str, results: list[dict], args: argparse.Namespace) -> None:
    """Export test results to a structured JSON file.

    Writes both the flat per-run results and a pivoted summary (model × mode → tok/s).
    The summary is what `--compare` consumes.
    """
    export = {
        "label": "test_models",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "system": {
            "cpu": platform.processor() or platform.machine(),
            "cores": os.cpu_count() or 0,
            "os": f"{platform.system()} {platform.release()}",
        },
        "config": {
            "device": args.device,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "cache_window": args.cache_window,
            "filter": args.filter,
        },
        "results": results,
        "summary": _build_summary(results),
    }
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\n[save] Results written to {dest}")


def _show_results(path: str) -> int:
    """Load and display test results from a JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {path}: {e}", file=sys.stderr)
        return 1

    results = data.get("results", [])
    if not results:
        print(f"No results found in {path}", file=sys.stderr)
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
    if config.get("cache_type_k") or config.get("cache_type_v"):
        print(f"  Cache:  K={config.get('cache_type_k', 'f32')} V={config.get('cache_type_v', 'f32')} "
              f"window={config.get('cache_window', 0)}")

    # Collect unique modes and build throughput data
    mode_keys: list[str] = []
    throughput_data: dict[str, dict[str, tuple[float, bool]]] = {}

    for r in results:
        mode = r.get("mode", "?")
        if mode not in mode_keys:
            mode_keys.append(mode)

        name = r.get("name", "?")
        if name not in throughput_data:
            throughput_data[name] = {}
        throughput_data[name][mode] = (r.get("decode_tok_s", 0), r.get("passed", False))

    # Print results table
    print()
    show_extra = len(mode_keys) > 1
    if show_extra:
        print(f"{'Test':<30} {'Device':<6} {'Mode':<16} {'Result':<8} {'Time':>8}  Details")
    else:
        print(f"{'Test':<30} {'Result':<8} {'Time':>8}  Details")
    sep_width = 115 if show_extra else 70
    print("=" * sep_width)

    passed = failed = 0
    for r in results:
        name = r.get("name", "?")
        device = r.get("device", "")
        mode = r.get("mode", "")
        ok = r.get("passed", False)
        elapsed = r.get("elapsed", 0)
        detail = r.get("detail", "")[:55]
        time_str = f"{elapsed:.1f}s"
        status = "PASS" if ok else "FAIL"

        if ok:
            passed += 1
        else:
            failed += 1

        # Mode column: show cache/hybrid info, not bare device name
        mode_display = mode if mode not in ("cpu", "gpu") else ""

        if show_extra:
            print(f"{name:<30} {device:<6} {mode_display:<16} {status:<8} {time_str:>8}  {detail}")
        else:
            print(f"{name:<30} {status:<8} {time_str:>8}  {detail}")

    total = passed + failed
    print("=" * sep_width)
    print(f"\n{passed}/{total} passed, {failed} failed")

    # Throughput summary
    if len(mode_keys) > 1 and throughput_data:
        # Build minimal TestCase list for the summary printer
        seen: dict[str, TestCase] = {}
        for r in results:
            name = r.get("name", "?")
            if name not in seen:
                seen[name] = TestCase(name=name, repo="", quant=r.get("quant"), arch=r.get("arch", ""),
                                      prompt="", expected="")
        _print_throughput_summary(throughput_data, mode_keys, list(seen.values()), [])

    return 1 if failed > 0 else 0


def _load_summary(path: str) -> dict:
    """Load a saved JSON file and return its summary pivot.

    Handles both new-format files (with a top-level 'summary' field) and
    legacy files (with only 'results') by rebuilding the pivot on the fly.
    """
    with open(path) as f:
        data = json.load(f)
    summary = data.get("summary")
    if summary is None:
        # Legacy file: rebuild from flat results
        summary = _build_summary(data.get("results", []))
    return {
        "summary": summary,
        "timestamp": data.get("timestamp", ""),
        "system": data.get("system", {}),
    }


def _compare_results(path_before: str, path_after: str) -> int:
    """Display a side-by-side throughput diff between two saved JSON files."""
    try:
        before = _load_summary(path_before)
        after = _load_summary(path_after)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading comparison files: {e}", file=sys.stderr)
        return 1

    sb = before["summary"]
    sa = after["summary"]

    print(f"[compare] BEFORE: {path_before}")
    if before.get("timestamp"):
        print(f"           time:   {before['timestamp']}")
    sys_b = before.get("system") or {}
    if sys_b:
        print(f"           system: {sys_b.get('cpu', '?')} ({sys_b.get('cores', '?')} cores)")
    print(f"[compare] AFTER:  {path_after}")
    if after.get("timestamp"):
        print(f"           time:   {after['timestamp']}")
    sys_a = after.get("system") or {}
    if sys_a:
        print(f"           system: {sys_a.get('cpu', '?')} ({sys_a.get('cores', '?')} cores)")
    print()

    # Union of modes (preserve order: before first, then new from after)
    modes: list[str] = list(sb.get("modes", []))
    for m in sa.get("modes", []):
        if m not in modes:
            modes.append(m)

    # Union of model names (same order preservation)
    models: list[str] = []
    seen: set[str] = set()
    for name in sb.get("models", {}).keys():
        if name not in seen:
            models.append(name)
            seen.add(name)
    for name in sa.get("models", {}).keys():
        if name not in seen:
            models.append(name)
            seen.add(name)

    if not models or not modes:
        print("No overlapping data to compare.")
        return 1

    # Column widths
    name_w = 28
    quant_w = 7
    col_w = 18  # "43.9 → 43.2 -1.6%" fits in ~18 chars

    # Header
    header = f"{'Model':<{name_w}} {'Quant':<{quant_w}}"
    for m in modes:
        header += f"  {m:>{col_w}}"
    print(header)
    print("-" * len(header))

    def _get(summary: dict, name: str, mode: str) -> tuple[float, bool] | None:
        m = summary.get("models", {}).get(name)
        if not m:
            return None
        t = m.get("throughput", {}).get(mode)
        if not t:
            return None
        return (t.get("decode_tok_s", 0.0), t.get("passed", False))

    # Collect regressions/improvements for summary
    regressions: list[tuple[str, str, float, float, float]] = []
    improvements: list[tuple[str, str, float, float, float]] = []

    for name in models:
        m_info = sa.get("models", {}).get(name) or sb.get("models", {}).get(name) or {}
        quant = m_info.get("quant", "?")
        display_name = name[:name_w].rstrip()
        line = f"{display_name:<{name_w}} {quant:<{quant_w}}"

        for mode in modes:
            bt = _get(sb, name, mode)
            at = _get(sa, name, mode)
            cell = _format_diff_cell(bt, at, col_w)
            line += f"  {cell:>{col_w}}"

            # Track changes (>= 2% delta, both runs valid)
            if bt and at and bt[0] > 0 and at[0] > 0:
                delta_pct = (at[0] - bt[0]) / bt[0] * 100.0
                if delta_pct <= -2.0:
                    regressions.append((name, mode, bt[0], at[0], delta_pct))
                elif delta_pct >= 2.0:
                    improvements.append((name, mode, bt[0], at[0], delta_pct))
        print(line)

    print("-" * len(header))
    print(f"  legend: 'before → after ±%'    dropped: removed from this mode    added: new in this mode")

    # Top regressions/improvements lists
    def _sort_by_abs(items: list) -> list:
        return sorted(items, key=lambda x: abs(x[4]), reverse=True)

    if regressions:
        print()
        print(f"Regressions (>=2% slower):")
        for name, mode, b, a, d in _sort_by_abs(regressions)[:10]:
            print(f"  {name:<28} {mode:<16}  {b:>7.1f} → {a:>7.1f}  ({d:+.1f}%)")
    if improvements:
        print()
        print(f"Improvements (>=2% faster):")
        for name, mode, b, a, d in _sort_by_abs(improvements)[:10]:
            print(f"  {name:<28} {mode:<16}  {b:>7.1f} → {a:>7.1f}  ({d:+.1f}%)")

    return 0


def _format_diff_cell(before: tuple[float, bool] | None,
                      after: tuple[float, bool] | None,
                      width: int) -> str:
    """Format a single before→after cell as 'B → A ±%' or 'dropped'/'added'/etc."""
    if before is None and after is None:
        return "--"
    if before is None:
        if after[0] > 0:
            fail = "x" if not after[1] else ""
            return f"added {after[0]:.1f}{fail}"
        return "added --"
    if after is None:
        if before[0] > 0:
            return f"dropped ({before[0]:.1f})"
        return "dropped"

    b_val, b_ok = before
    a_val, a_ok = after
    b_fail = "x" if not b_ok else ""
    a_fail = "x" if not a_ok else ""

    if b_val <= 0 and a_val <= 0:
        return "-- → --"
    if b_val <= 0:
        return f"-- → {a_val:.1f}{a_fail}"
    if a_val <= 0:
        return f"{b_val:.1f}{b_fail} → --"

    delta_pct = (a_val - b_val) / b_val * 100.0
    sign = "+" if delta_pct >= 0 else ""
    return f"{b_val:.1f}{b_fail}→{a_val:.1f}{a_fail} {sign}{delta_pct:.0f}%"


def _print_throughput_summary(
    data: dict[str, dict[str, tuple[float, bool]]],
    mode_keys: list[str],
    cases: list[TestCase],
    hybrid_modes: list[float],
) -> None:
    """Print a pivoted throughput summary table after the test results."""
    # Build quant lookup from test cases
    quant_map = {tc.name: tc.quant or "f16" for tc in cases}
    layers_map = {tc.name: tc.layers for tc in cases}

    # Column widths: fixed width per mode column for alignment
    name_w = 28
    quant_w = 7
    col_w = 9  # width for each tok/s column
    best_w = 6

    # Build column headers with layer counts where applicable
    col_headers: list[str] = []
    for key in mode_keys:
        col_headers.append(key)

    # Print header
    print()
    print("Throughput Summary (decode tok/s)")
    print()

    # Header row
    header = f"{'Model':<{name_w}} {'Quant':<{quant_w}}"
    for ch in col_headers:
        header += f"  {ch:>{col_w}}"
    header += f"  {'Best':>{best_w}}"
    print(header)

    sep = "-" * len(header)
    print(sep)

    for tc_name, row in data.items():
        quant = quant_map.get(tc_name, "?")
        # Truncate long model names
        display_name = tc_name[:name_w].rstrip()

        line = f"{display_name:<{name_w}} {quant:<{quant_w}}"

        # Find best tok/s (only from modes that have data)
        best_tok_s = -1.0
        best_key = ""
        for key in mode_keys:
            if key in row:
                tok_s, _ = row[key]
                if tok_s > best_tok_s:
                    best_tok_s = tok_s
                    best_key = key

        for key in mode_keys:
            if key not in row:
                line += f"  {'--':>{col_w}}"
                continue

            tok_s, ok = row[key]
            is_best = (key == best_key and best_tok_s > 0)

            if tok_s > 0:
                cell = f"{tok_s:.1f}"
                if not ok:
                    cell += "x"
                if is_best:
                    cell += "*"
                line += f"  {cell:>{col_w}}"
            else:
                cell = "--x" if not ok else "--"
                line += f"  {cell:>{col_w}}"

        line += f"  {best_key:>{best_w}}"
        print(line)

    print(sep)
    print(f"  * = fastest    x = correctness failed    -- = no data")


if __name__ == "__main__":
    # Ensure UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
