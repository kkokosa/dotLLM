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
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
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
              gpu_layers: int | None = None) -> TestResult:
    """
    Run a single test case with --json output.

    Args:
        gpu_layers: If set, pass --gpu-layers N to the CLI (hybrid mode).
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
    args = parser.parse_args()

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

    # Build run configs: list of (label, device, gpu_layers)
    # Regular device runs + hybrid mode runs
    has_hybrid = len(hybrid_modes) > 0
    show_mode_col = len(devices) > 1 or has_hybrid

    # Run tests
    print()
    mode_col = "  Mode    " if show_mode_col else ""
    print(f"{'Test':<35} {'Arch':<10}{mode_col} {'Result':<8} {'Time':>8}  Details")
    print("=" * (115 if show_mode_col else 105))

    passed = 0
    failed = 0
    skipped = 0

    # Collect throughput data for summary table: { tc.name: { mode_key: (decode_tok_s, passed) } }
    throughput_data: dict[str, dict[str, tuple[float, bool]]] = {}
    # Ordered mode keys for column headers
    mode_keys: list[str] = list(devices)
    for frac in hybrid_modes:
        mode_keys.append(f"h{int(frac * 100)}%")

    def _clean_detail(detail: str, prompt: str) -> str:
        if prompt in detail:
            detail = detail[detail.index(prompt) + len(prompt):]
        for marker in ["Generation Complete", "Performance", "Prefill"]:
            if marker in detail:
                detail = detail[:detail.index(marker)]
        return detail.strip()[:60]

    for tc in cases:
        # Check if model is available
        if not _model_is_cached(tc) and not args.download:
            skipped += 1
            mode_label = "" if not show_mode_col else "          "
            print(f"{tc.name:<35} {tc.arch:<10}{mode_label} {'SKIP':<8} {'':>8}  not cached (use --download)")
            continue

        # Resolve model (downloads if --download and not cached)
        try:
            model_path = resolve_model(tc.repo, tc.quant, quiet=True)
        except SystemExit:
            num_runs = len(devices) + len(hybrid_modes)
            failed += num_runs
            mode_label = "" if not show_mode_col else "          "
            print(f"{tc.name:<35} {tc.arch:<10}{mode_label} {'FAIL':<8} {'':>8}  model resolution failed")
            continue

        # Determine layer count for hybrid modes
        num_layers = tc.layers
        if hybrid_modes and num_layers == 0:
            num_layers = _get_gguf_layers(str(model_path))
            if num_layers == 0:
                print(f"  [hybrid] WARNING: Could not determine layer count for {tc.name}, skipping hybrid")

        row: dict[str, tuple[float, bool]] = {}

        # Regular device runs
        for device in devices:
            r = _run_test(cli, model_path, tc, device=device)
            time_str = f"{r.elapsed:.1f}s"
            mode_label = f"  {device:<8}" if show_mode_col else ""

            if r.passed:
                passed += 1
                detail = _clean_detail(r.detail, tc.prompt)
                print(f"{tc.name:<35} {tc.arch:<10}{mode_label} {'PASS':<8} {time_str:>8}  {detail}")
            else:
                failed += 1
                print(f"{tc.name:<35} {tc.arch:<10}{mode_label} {'FAIL':<8} {time_str:>8}  {r.detail}")

            row[device] = (r.decode_tok_s, r.passed)

        # Hybrid mode runs
        if hybrid_modes and num_layers > 0:
            for frac in hybrid_modes:
                gl = compute_gpu_layers(num_layers, frac)
                pct = int(frac * 100)
                hybrid_label = f"  h{pct}%({gl}L)" if show_mode_col else ""

                r = _run_test(cli, model_path, tc, gpu_layers=gl)
                time_str = f"{r.elapsed:.1f}s"

                if r.passed:
                    passed += 1
                    detail = _clean_detail(r.detail, tc.prompt)
                    print(f"{tc.name:<35} {tc.arch:<10}{hybrid_label} {'PASS':<8} {time_str:>8}  {detail}")
                else:
                    failed += 1
                    print(f"{tc.name:<35} {tc.arch:<10}{hybrid_label} {'FAIL':<8} {time_str:>8}  {r.detail}")

                row[f"h{pct}%"] = (r.decode_tok_s, r.passed)

        if row:
            throughput_data[tc.name] = row

    # Summary
    sep_width = 115 if show_mode_col else 105
    print("=" * sep_width)
    total = passed + failed + skipped
    print(f"\n{passed}/{total} passed, {failed} failed, {skipped} skipped")

    # Throughput summary table (only when multiple modes were tested)
    if len(mode_keys) > 1 and throughput_data:
        _print_throughput_summary(throughput_data, mode_keys, cases, hybrid_modes)

    return 1 if failed > 0 else 0


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
