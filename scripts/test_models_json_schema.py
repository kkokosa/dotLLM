#!/usr/bin/env python3
"""
test_models_json_schema.py — Smoke test for JSON Schema constrained decoding.

Runs dotLLM CLI with --response-format json_schema and validates that the
generated output conforms to the schema (correct types, required properties,
no additional properties). Tests across cached model architectures.

Usage:
    python scripts/test_models_json_schema.py                    # run all cached models
    python scripts/test_models_json_schema.py --filter llama     # only models matching "llama"
    python scripts/test_models_json_schema.py --list             # show available test cases
    python scripts/test_models_json_schema.py --device gpu       # GPU-only
    python scripts/test_models_json_schema.py --device both      # CPU then GPU
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_compare import resolve_model
from test_models import TEST_CASES as MODEL_TEST_CASES, _model_is_cached as _base_model_is_cached


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

SCHEMA_PATH = Path(__file__).resolve().parent / "test-schema.json"
SCHEMA_PROMPT = "Output a JSON object with a person's name and age."
SCHEMA_MAX_TOKENS = 60


def _load_schema() -> dict:
    """Load and parse the test schema."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


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


def _validate_against_schema(parsed: dict, schema: dict) -> str | None:
    """Validate parsed JSON against the test schema. Returns error string or None."""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    additional_forbidden = schema.get("additionalProperties") is False

    # Check required properties
    for key in required:
        if key not in parsed:
            return f"missing required property '{key}'"

    # Check no additional properties
    if additional_forbidden:
        extra = set(parsed.keys()) - set(props.keys())
        if extra:
            return f"unexpected properties: {sorted(extra)}"

    # Check types
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    for key, prop_schema in props.items():
        if key not in parsed:
            continue
        expected_type = prop_schema.get("type")
        if expected_type and expected_type in type_map:
            py_type = type_map[expected_type]
            value = parsed[key]
            if not isinstance(value, py_type):
                # Special case: integer type should reject floats with fractional parts
                if expected_type == "integer" and isinstance(value, float):
                    if value != int(value):
                        return f"'{key}' is float {value}, expected integer"
                    # whole-number float is acceptable (JSON has no int type)
                    continue
                # bool is subclass of int in Python — reject bools for integer/number
                if expected_type in ("integer", "number") and isinstance(value, bool):
                    return f"'{key}' is bool, expected {expected_type}"
                return f"'{key}' has type {type(value).__name__}, expected {expected_type}"

    return None


def _run_test(cli: Path, model_path: Path, schema: dict, tc, device: str = "cpu") -> TestResult:
    """Run a single JSON Schema test."""
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
        "-p", SCHEMA_PROMPT,
        "-n", str(SCHEMA_MAX_TOKENS),
        "-t", "0",
        "--response-format", "json_schema",
        "--schema", f"@{SCHEMA_PATH}",
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

    # Parse generated text as JSON
    try:
        parsed = json.loads(generated_text)
    except json.JSONDecodeError as e:
        preview = generated_text[:80].replace("\n", "\\n")
        return TestResult(False, f"invalid JSON: {e} | {preview}", elapsed, generated_text)

    if not isinstance(parsed, dict):
        return TestResult(False, f"not object: {type(parsed).__name__}", elapsed, generated_text)

    # Validate against schema
    error = _validate_against_schema(parsed, schema)
    if error:
        preview = generated_text[:60].replace("\n", " ")
        return TestResult(False, f"{error} | {preview}", elapsed, generated_text)

    preview = generated_text[:60].replace("\n", " ")
    return TestResult(True, preview, elapsed, generated_text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for JSON Schema constrained decoding."
    )
    parser.add_argument("--filter", type=str, default=None,
                        help="Comma-separated name/arch substrings to match")
    parser.add_argument("--list", action="store_true",
                        help="List available test cases and exit")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "gpu", "both"],
                        help="Compute device (default: cpu)")
    args = parser.parse_args()

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
        return 0

    cases = [tc for tc in cases if _base_model_is_cached(tc)]
    if not cases:
        print("No cached models found. Download models first via test_models.py --download.")
        return 1

    schema = _load_schema()

    cli = _find_cli()
    if cli.name == "dotnet":
        print("[cli] No prebuilt CLI found, will use 'dotnet run' (slower startup)")
    else:
        print(f"[cli] {cli}")

    devices = ["cpu", "gpu"] if args.device == "both" else [args.device]
    multi_device = len(devices) > 1

    print(f"[schema] {SCHEMA_PATH}")
    print(f"[schema] prompt: {SCHEMA_PROMPT}")
    print(f"[schema] required: {schema.get('required', [])}, "
          f"additionalProperties: {schema.get('additionalProperties', True)}")
    print(f"[schema] max_tokens: {SCHEMA_MAX_TOKENS}")
    print()
    if multi_device:
        print(f"{'Test':<30} {'Device':<6} {'Result':<8} {'Time':>8}  Details")
    else:
        print(f"{'Test':<30} {'Result':<8} {'Time':>8}  Details")
    sep_width = 100 if multi_device else 90
    print("=" * sep_width)

    passed = 0
    failed = 0

    for tc in cases:
        try:
            model_path = resolve_model(tc.repo, tc.quant, quiet=True)
        except SystemExit:
            failed += 1
            print(f"{tc.name:<30} {'FAIL':<8} {'':>8}  model resolution failed")
            continue

        for device in devices:
            r = _run_test(cli, model_path, schema, tc, device=device)
            time_str = f"{r.elapsed:.1f}s"
            status = "PASS" if r.passed else "FAIL"
            detail = r.detail[:55]

            if r.passed:
                passed += 1
            else:
                failed += 1

            if multi_device:
                print(f"{tc.name:<30} {device:<6} {status:<8} {time_str:>8}  {detail}")
            else:
                print(f"{tc.name:<30} {status:<8} {time_str:>8}  {detail}")

    print("=" * sep_width)
    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
