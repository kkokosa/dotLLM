#!/usr/bin/env python3
"""
test_models_regex.py — Smoke test for regex constrained decoding.

Runs dotLLM CLI with --response-format regex --pattern <pattern> and validates
that the generated output fully matches the pattern. Tests multiple patterns
across cached model architectures.

Usage:
    python scripts/test_models_regex.py                        # run all cached models
    python scripts/test_models_regex.py --filter llama         # only models matching "llama"
    python scripts/test_models_regex.py --list                 # show available test cases
    python scripts/test_models_regex.py --device gpu           # GPU-only
    python scripts/test_models_regex.py --device both          # CPU then GPU
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
# Test patterns
# ---------------------------------------------------------------------------

PATTERNS_PATH = Path(__file__).resolve().parent / "test-regex-patterns.json"


def _load_patterns() -> list[dict]:
    """Load regex test patterns from JSON file."""
    return json.loads(PATTERNS_PATH.read_text(encoding="utf-8"))


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


def _run_test(cli: Path, model_path: Path, pattern_cfg: dict,
              device: str = "cpu") -> TestResult:
    """Run a single regex constraint test."""
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
        "-p", pattern_cfg["prompt"],
        "-n", str(pattern_cfg.get("max_tokens", 16)),
        "-t", "0",
        "--response-format", "regex",
        "--pattern", pattern_cfg["pattern"],
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

    # Validate: generated text must fully match the regex pattern
    pattern = pattern_cfg["pattern"]
    if not re.fullmatch(pattern, generated_text):
        preview = generated_text[:80].replace("\n", "\\n")
        return TestResult(
            False,
            f"no match for /{pattern}/ | {preview}",
            elapsed, generated_text,
        )

    return TestResult(True, f"{generated_text[:60]}", elapsed, generated_text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for regex constrained decoding."
    )
    parser.add_argument("--filter", type=str, default=None,
                        help="Comma-separated name/arch substrings to match")
    parser.add_argument("--list", action="store_true",
                        help="List available test cases and exit")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "gpu", "both"],
                        help="Compute device (default: cpu)")
    parser.add_argument("--pattern-filter", type=str, default=None,
                        help="Only run patterns matching this substring (e.g., 'date')")
    args = parser.parse_args()

    patterns = _load_patterns()
    if args.pattern_filter:
        pf = args.pattern_filter.lower()
        patterns = [p for p in patterns if pf in p["name"].lower()]
        if not patterns:
            print(f"No patterns match filter '{args.pattern_filter}'.")
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
        print("Patterns:")
        for p in patterns:
            print(f"  {p['name']:<15} /{p['pattern']}/")
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

    print(f"[regex] {len(patterns)} pattern(s):")
    for p in patterns:
        print(f"  {p['name']:<15} /{p['pattern']}/")
    print()
    if multi_device:
        print(f"{'Test':<30} {'Pattern':<12} {'Device':<6} {'Result':<8} {'Time':>8}  Details")
    else:
        print(f"{'Test':<30} {'Pattern':<12} {'Result':<8} {'Time':>8}  Details")
    sep_width = 110 if multi_device else 100
    print("=" * sep_width)

    passed = 0
    failed = 0

    for tc in cases:
        try:
            model_path = resolve_model(tc.repo, tc.quant, quiet=True)
        except SystemExit:
            failed += len(patterns) * len(devices)
            print(f"{tc.name:<30} {'':12} {'FAIL':<8} {'':>8}  model resolution failed")
            continue

        for pat in patterns:
            for device in devices:
                r = _run_test(cli, model_path, pat, device=device)
                time_str = f"{r.elapsed:.1f}s"
                status = "PASS" if r.passed else "FAIL"
                detail = r.detail[:50]

                if r.passed:
                    passed += 1
                else:
                    failed += 1

                if multi_device:
                    print(f"{tc.name:<30} {pat['name']:<12} {device:<6} {status:<8} {time_str:>8}  {detail}")
                else:
                    print(f"{tc.name:<30} {pat['name']:<12} {status:<8} {time_str:>8}  {detail}")

    print("=" * sep_width)
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
