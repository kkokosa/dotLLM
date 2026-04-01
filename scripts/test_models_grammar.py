#!/usr/bin/env python3
"""
test_models_grammar.py — Smoke test for GBNF grammar constrained decoding.

Runs dotLLM CLI with --response-format grammar --grammar @<file> and validates
that the generated output conforms to the grammar structure. Tests across
cached model architectures.

Usage:
    python scripts/test_models_grammar.py                        # run all cached models
    python scripts/test_models_grammar.py --filter llama         # only models matching "llama"
    python scripts/test_models_grammar.py --list                 # show available test cases
    python scripts/test_models_grammar.py --device gpu           # GPU-only
    python scripts/test_models_grammar.py --device both          # CPU then GPU
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_compare import resolve_model
from test_models import TEST_CASES as MODEL_TEST_CASES, _model_is_cached as _base_model_is_cached


# ---------------------------------------------------------------------------
# Test grammars
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent

# Each grammar test has: a GBNF file, a prompt, a Python regex that validates
# the grammar's output structure, and max tokens.
GRAMMAR_TESTS = [
    {
        "name": "yes-no-maybe",
        "grammar_inline": 'root ::= "yes" | "no" | "maybe"',
        "prompt": "Is the sky blue? Answer with exactly one word: yes, no, or maybe:",
        "max_tokens": 8,
        "validation_regex": r"^(yes|no|maybe)$",
    },
    {
        "name": "person",
        "grammar_file": str(SCRIPTS_DIR / "test-grammar.gbnf"),
        "prompt": "Describe a fictional person in the format 'Name (age) - occupation':",
        "max_tokens": 48,
        # Name Surname (age) - occupation (1-3 words)
        "validation_regex": r"^[A-Z][a-z]+ [A-Z][a-z]+ \([1-9][0-9]?\) - [a-z]+( [a-z]+( [a-z]+)?)?$",
    },
]


def _find_cli() -> Path:
    """Find the dotLLM CLI executable."""
    repo_root = Path(__file__).resolve().parent.parent
    bin_dir = repo_root / "src" / "DotLLM.Cli" / "bin"
    for config in ("Release", "Debug"):
        for ext in (".exe", ""):
            p = bin_dir / config / "net10.0" / f"DotLLM.Cli{ext}"
            if p.exists():
                return p
    return Path("dotnet")


@dataclass
class TestResult:
    passed: bool
    detail: str
    elapsed: float
    generated_text: str = ""


def _run_test(cli: Path, model_path: Path, grammar_cfg: dict,
              device: str = "cpu") -> TestResult:
    """Run a single grammar constraint test."""
    if cli.name == "dotnet":
        cmd = [
            str(cli), "run",
            "--project", str(Path(__file__).resolve().parent.parent / "src" / "DotLLM.Cli"),
            "-c", "Release", "--",
        ]
    else:
        cmd = [str(cli)]

    # Grammar source: inline string or @file
    if "grammar_file" in grammar_cfg:
        grammar_arg = f"@{grammar_cfg['grammar_file']}"
    else:
        grammar_arg = grammar_cfg["grammar_inline"]

    cmd += [
        "run", str(model_path),
        "-p", grammar_cfg["prompt"],
        "-n", str(grammar_cfg.get("max_tokens", 32)),
        "-t", "0",
        "--response-format", "grammar",
        "--grammar", grammar_arg,
        "--json",
        "--device", device,
    ]

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=600,
            encoding="utf-8", errors="replace",
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

    # Parse the CLI --json envelope
    raw = result.stdout.strip()
    json_start = raw.find("{")
    if json_start < 0:
        return TestResult(False, f"no JSON in CLI output: {raw[:200]}", elapsed)
    try:
        envelope = json.loads(raw[json_start:])
    except json.JSONDecodeError as e:
        return TestResult(False, f"invalid CLI JSON: {e}", elapsed)

    generated_text = envelope.get("text", "").strip()

    if not generated_text:
        return TestResult(False, "empty output", elapsed, generated_text)

    # Validate: generated text must match the structural validation regex
    validation = grammar_cfg.get("validation_regex")
    if validation and not re.fullmatch(validation, generated_text):
        preview = generated_text[:80].replace("\n", "\\n")
        return TestResult(
            False,
            f"structure mismatch | {preview}",
            elapsed, generated_text,
        )

    return TestResult(True, f"{generated_text[:60]}", elapsed, generated_text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for GBNF grammar constrained decoding."
    )
    parser.add_argument("--filter", type=str, default=None,
                        help="Comma-separated name/arch substrings to match")
    parser.add_argument("--list", action="store_true",
                        help="List available test cases and exit")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "gpu", "both"],
                        help="Compute device (default: cpu)")
    parser.add_argument("--grammar-filter", type=str, default=None,
                        help="Only run grammars matching this substring (e.g., 'person')")
    args = parser.parse_args()

    grammars = list(GRAMMAR_TESTS)
    if args.grammar_filter:
        gf = args.grammar_filter.lower()
        grammars = [g for g in grammars if gf in g["name"].lower()]
        if not grammars:
            print(f"No grammars match filter '{args.grammar_filter}'.")
            return 1

    cases = list(MODEL_TEST_CASES)
    if args.filter:
        filters = [f.strip().lower() for f in args.filter.split(",")]
        cases = [
            tc for tc in cases
            if any(f in tc.name.lower() or f in tc.arch.lower() for f in filters)
        ]
        if not cases:
            print(f"No test cases match filter '{args.filter}'.")
            return 1

    if args.list:
        print(f"{'Name':<35} {'Arch':<10} {'Quant':<8} {'Cached':<8} Notes")
        print("-" * 100)
        for tc in cases:
            cached = "yes" if _base_model_is_cached(tc) else "no"
            quant = tc.quant or "default"
            print(f"{tc.name:<35} {tc.arch:<10} {quant:<8} {cached:<8} {tc.notes}")
        print()
        print("Grammars:")
        for g in grammars:
            src = g.get("grammar_file", "inline")
            print(f"  {g['name']:<20} {src}")
        return 0

    cases = [tc for tc in cases if _base_model_is_cached(tc)]
    if not cases:
        print("No cached models found. Download models first via test_models.py --download.")
        return 1

    cli = _find_cli()
    if cli.name == "dotnet":
        print("[cli] No prebuilt CLI found, will use 'dotnet run' (slower startup)")
    else:
        print(f"[cli] {cli}")

    devices = ["cpu", "gpu"] if args.device == "both" else [args.device]
    multi_device = len(devices) > 1

    print(f"[grammar] {len(grammars)} grammar(s):")
    for g in grammars:
        src = g.get("grammar_file", "inline")
        print(f"  {g['name']:<20} {src}")
    print()
    if multi_device:
        print(f"{'Test':<30} {'Grammar':<16} {'Device':<6} {'Result':<8} {'Time':>8}  Details")
    else:
        print(f"{'Test':<30} {'Grammar':<16} {'Result':<8} {'Time':>8}  Details")
    sep_width = 115 if multi_device else 105
    print("=" * sep_width)

    passed = 0
    failed = 0

    for tc in cases:
        try:
            model_path = resolve_model(tc.repo, tc.quant, quiet=True)
        except SystemExit:
            failed += len(grammars) * len(devices)
            print(f"{tc.name:<30} {'':16} {'FAIL':<8} {'':>8}  model resolution failed")
            continue

        for gram in grammars:
            for device in devices:
                r = _run_test(cli, model_path, gram, device=device)
                time_str = f"{r.elapsed:.1f}s"
                status = "PASS" if r.passed else "FAIL"
                detail = r.detail[:48]

                if r.passed:
                    passed += 1
                else:
                    failed += 1

                if multi_device:
                    print(f"{tc.name:<30} {gram['name']:<16} {device:<6} {status:<8} {time_str:>8}  {detail}")
                else:
                    print(f"{tc.name:<30} {gram['name']:<16} {status:<8} {time_str:>8}  {detail}")

    print("=" * sep_width)
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
