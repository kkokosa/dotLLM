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

# Reuse model resolution from bench_compare
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_compare import resolve_model


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
        notes="baseline Llama arch, SentencePiece",
    ),
    TestCase(
        name="SmolLM2-135M-Instruct",
        repo="bartowski/SmolLM2-135M-Instruct-GGUF",
        quant="Q8_0",
        arch="Llama",
        prompt="The capital of France is",
        expected="Paris",
        notes="SmolLM2, SentencePiece",
    ),
    TestCase(
        name="Llama-3.2-1B-Instruct-Q4",
        repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
        quant="Q4_K_M",
        arch="Llama",
        prompt="The capital of France is",
        expected="Paris",
        notes="Llama 3.2, tiktoken, Q4_K_M",
    ),
    TestCase(
        name="Llama-3.2-1B-Instruct-Q8",
        repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
        quant="Q8_0",
        arch="Llama",
        prompt="The capital of France is",
        expected="Paris",
        notes="Llama 3.2, tiktoken, Q8_0",
    ),
    TestCase(
        name="Llama-3.2-3B-Instruct-Q4",
        repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        quant="Q4_K_M",
        arch="Llama",
        prompt="The capital of France is",
        expected="Paris",
        notes="Llama 3.2 3B, tiktoken, Q4_K_M",
    ),
    TestCase(
        name="Llama-3.2-3B-Instruct-Q8",
        repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        quant="Q8_0",
        arch="Llama",
        prompt="The capital of France is",
        expected="Paris",
        notes="Llama 3.2 3B, tiktoken, Q8_0",
    ),
    TestCase(
        name="Bielik-1.5B-Instruct",
        repo="speakleash/Bielik-1.5B-v3.0-Instruct-GGUF",
        quant="Q8_0",
        arch="Llama",
        prompt="Stolicą Polski jest",
        expected="Warszawa",
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
        notes="tiktoken, tied embeddings, Q/K biases",
    ),
    TestCase(
        name="Qwen3-0.6B",
        repo="Qwen/Qwen3-0.6B-GGUF",
        quant="Q8_0",
        arch="Qwen",
        prompt="The capital of France is",
        expected="Paris",
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
        notes="fused QKV + fused gate_up FFN, phi3 arch",
    ),
    TestCase(
        name="Phi-4-mini-instruct",
        repo="unsloth/Phi-4-mini-instruct-GGUF",
        quant="Q8_0",
        arch="Phi",
        prompt="The capital of France is",
        expected="Paris",
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
        notes="mistral3 arch string",
    ),
    TestCase(
        name="Mistral-7B-Instruct-v0.3-Q4",
        repo="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        quant="Q4_K_M",
        arch="Mistral",
        prompt="The capital of France is",
        expected="Paris",
        notes="sliding window, Q4_K_M",
    ),
    TestCase(
        name="Mistral-7B-Instruct-v0.3-Q8",
        repo="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        quant="Q8_0",
        arch="Mistral",
        prompt="The capital of France is",
        expected="Paris",
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


def _run_test(cli: Path, model_path: Path, tc: TestCase) -> tuple[bool, str, float]:
    """
    Run a single test case with --json output. Returns (passed, detail_text, elapsed_seconds).
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
        return False, "TIMEOUT (600s)", time.monotonic() - start
    except FileNotFoundError:
        return False, f"CLI not found: {cli}", time.monotonic() - start

    elapsed = time.monotonic() - start

    if result.returncode != 0:
        error_text = result.stderr.strip() or result.stdout.strip()
        # Find "Error: ..." line
        for line in error_text.splitlines():
            if "Error:" in line:
                return False, line.strip(), elapsed
        return False, f"exit code {result.returncode}: {error_text[:200]}", elapsed

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False, f"invalid JSON: {result.stdout[:200]}", elapsed

    generated_text = data.get("text", "")
    timings = data.get("timings", {})
    tok_s = timings.get("decode_tok_s", 0) or timings.get("prefill_tok_s", 0)

    if tc.expected in generated_text:
        detail = f"{generated_text.strip()[:50]}  ({tok_s:.1f} tok/s)"
        return True, detail, elapsed
    else:
        return False, f"expected '{tc.expected}' not in: '{generated_text[:100]}'", elapsed


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

    # Run tests
    print()
    print(f"{'Test':<35} {'Arch':<10} {'Result':<8} {'Time':>8}  Details")
    print("=" * 105)

    passed = 0
    failed = 0
    skipped = 0

    for tc in cases:
        # Check if model is available
        if not _model_is_cached(tc) and not args.download:
            skipped += 1
            print(f"{tc.name:<35} {tc.arch:<10} {'SKIP':<8} {'':>8}  not cached (use --download)")
            continue

        # Resolve model (downloads if --download and not cached)
        try:
            model_path = resolve_model(tc.repo, tc.quant, quiet=True)
        except SystemExit:
            failed += 1
            print(f"{tc.name:<35} {tc.arch:<10} {'FAIL':<8} {'':>8}  model resolution failed")
            continue

        # Run the test
        ok, detail, elapsed = _run_test(cli, model_path, tc)
        time_str = f"{elapsed:.1f}s"

        if ok:
            passed += 1
            # Show just the generated text after the prompt
            if tc.prompt in detail:
                detail = detail[detail.index(tc.prompt) + len(tc.prompt):]
            # Truncate at "Generation Complete" or similar perf box text
            for marker in ["Generation Complete", "Performance", "Prefill"]:
                if marker in detail:
                    detail = detail[:detail.index(marker)]
            detail = detail.strip()[:60]
            print(f"{tc.name:<35} {tc.arch:<10} {'PASS':<8} {time_str:>8}  {detail}")
        else:
            failed += 1
            print(f"{tc.name:<35} {tc.arch:<10} {'FAIL':<8} {time_str:>8}  {detail}")

    # Summary
    print("=" * 105)
    total = passed + failed + skipped
    print(f"\n{passed}/{total} passed, {failed} failed, {skipped} skipped")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    # Ensure UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
