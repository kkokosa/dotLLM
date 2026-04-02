#!/usr/bin/env python3
"""
test_models_tools.py — Smoke test for tool calling across model architectures.

Runs dotLLM CLI chat command in single-turn mode and validates that the model
produces a detectable tool call with the expected function name.

Usage:
    python scripts/test_models_tools.py                # run all cached models
    python scripts/test_models_tools.py --filter llama  # only models matching "llama"
    python scripts/test_models_tools.py --list          # show available test cases
    python scripts/test_models_tools.py --download      # download missing models
    python scripts/test_models_tools.py --device gpu    # GPU
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_compare import resolve_model


# ---------------------------------------------------------------------------
# Tool calling test definitions
# ---------------------------------------------------------------------------

TOOLS_JSON = json.dumps([
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_time",
        "description": "Get current time in a timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "IANA timezone (e.g., Europe/London)"},
            },
            "required": ["timezone"],
        },
    },
])


@dataclass
class ToolTestCase:
    """A single tool calling test."""
    name: str
    repo: str
    quant: str | None
    arch: str
    user_prompt: str
    expected_tool: str  # function name that should be called
    notes: str = ""


TEST_CASES: list[ToolTestCase] = [
    ToolTestCase(
        name="Llama-3.2-1B-Instruct",
        repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
        quant="Q8_0",
        arch="Llama",
        user_prompt="What's the weather in Paris?",
        expected_tool="get_weather",
        notes="Llama format, <|python_tag|> or bare JSON",
    ),
    ToolTestCase(
        name="Qwen2.5-1.5B-Instruct",
        repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        quant="Q8_0",
        arch="Qwen",
        user_prompt="What's the weather in Paris?",
        expected_tool="get_weather",
        notes="Hermes format, <tool_call> tags",
    ),
    ToolTestCase(
        name="Ministral-3-3B-Instruct",
        repo="mistralai/Ministral-3-3B-Instruct-2512-GGUF",
        quant=None,
        arch="Mistral",
        user_prompt="What's the weather in Paris?",
        expected_tool="get_weather",
        notes="Mistral v3 format, [TOOL_CALLS]func[ARGS]",
    ),
]


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

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


def _default_models_dir() -> Path:
    home = Path.home()
    return home / ".dotllm" / "models"


def _apply_quant_filter(files: list[str], quant: str | None) -> list[str]:
    if not quant:
        return files
    q = quant.lower()
    return [f for f in files if q in f.lower()]


def _model_is_cached(tc: ToolTestCase) -> bool:
    if tc.repo.endswith(".gguf"):
        return Path(tc.repo).exists()
    models_dir = _default_models_dir()
    repo_dir = models_dir / tc.repo.replace("/", os.sep)
    if not repo_dir.exists():
        return False
    cached = [f.name for f in repo_dir.iterdir() if f.suffix == ".gguf"]
    cached = _apply_quant_filter(cached, tc.quant)
    return len(cached) >= 1


@dataclass
class TestResult:
    passed: bool
    detail: str
    elapsed: float
    generated_text: str = ""


def _run_test(cli: Path, model_path: Path, tc: ToolTestCase,
              device: str = "cpu") -> TestResult:
    """Run a single tool calling test using the run command with --tools and --json."""
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
        "-p", tc.user_prompt,
        "-n", "60",
        "-t", "0",  # greedy
        "--json",
        "--device", device,
        "--tools", TOOLS_JSON,
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

    # Parse CLI --json envelope
    raw = result.stdout.strip()
    json_start = raw.find("{")
    if json_start < 0:
        return TestResult(False, f"no JSON in CLI output: {raw[:200]}", elapsed)
    try:
        envelope = json.loads(raw[json_start:])
    except json.JSONDecodeError as e:
        return TestResult(False, f"invalid CLI JSON: {e}", elapsed)

    # Check finish_reason
    finish_reason = envelope.get("finish_reason", "")
    tool_calls = envelope.get("tool_calls")

    if finish_reason == "toolcalls" and tool_calls and len(tool_calls) > 0:
        # Tool calls detected by the engine
        tc_obj = tool_calls[0]
        name = tc_obj.get("function_name", "")
        args = tc_obj.get("arguments", "{}")
        if name == tc.expected_tool:
            preview = f"{name}({args})"
            if len(preview) > 55:
                preview = preview[:52] + "..."
            return TestResult(True, preview, elapsed, envelope.get("text", ""))
        else:
            return TestResult(False, f"wrong tool: {name} (expected {tc.expected_tool})", elapsed)

    # Fallback: check generated text for the expected tool name
    generated_text = envelope.get("text", "").strip()
    if not generated_text:
        return TestResult(False, f"empty output, finish_reason={finish_reason}", elapsed)

    if tc.expected_tool not in generated_text:
        preview = generated_text[:80].replace("\n", "\\n")
        return TestResult(False, f"'{tc.expected_tool}' not in output: {preview}", elapsed, generated_text)

    preview = generated_text[:55].replace("\n", "\\n")
    return TestResult(True, f"(text match) {preview}", elapsed, generated_text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for tool calling across model architectures."
    )
    parser.add_argument("--filter", type=str, default=None,
                        help="Comma-separated name/arch substrings to match")
    parser.add_argument("--list", action="store_true",
                        help="List available test cases and exit")
    parser.add_argument("--download", action="store_true",
                        help="Download missing models before testing")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "gpu", "both"],
                        help="Compute device (default: cpu)")
    args = parser.parse_args()

    cases = list(TEST_CASES)
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
        print(f"{'Name':<30} {'Arch':<10} {'Quant':<8} {'Cached':<8} {'Expected Tool':<15} Notes")
        print("-" * 105)
        for tc in cases:
            cached = "yes" if _model_is_cached(tc) else "no"
            quant = tc.quant or "default"
            print(f"{tc.name:<30} {tc.arch:<10} {quant:<8} {cached:<8} {tc.expected_tool:<15} {tc.notes}")
        return 0

    # Filter to cached models (unless --download)
    if not args.download:
        cached_cases = [tc for tc in cases if _model_is_cached(tc)]
        skipped = len(cases) - len(cached_cases)
        cases = cached_cases
        if not cases:
            print("No cached models found. Run with --download to fetch them.")
            return 1
        if skipped:
            print(f"[info] Skipping {skipped} uncached model(s). Use --download to fetch.")

    cli = _find_cli()
    if cli.name == "dotnet":
        print("[cli] No prebuilt CLI found, will use 'dotnet run' (slower startup)")
    else:
        print(f"[cli] {cli}")

    devices = ["cpu", "gpu"] if args.device == "both" else [args.device]
    multi_device = len(devices) > 1

    print(f"[tools] 2 tools defined (get_weather, get_time)")
    print(f"[tools] prompt: 'What's the weather in Paris?'")
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
            if multi_device:
                print(f"{tc.name:<30} {'':6} {'FAIL':<8} {'':>8}  model resolution failed")
            else:
                print(f"{tc.name:<30} {'FAIL':<8} {'':>8}  model resolution failed")
            continue

        for device in devices:
            r = _run_test(cli, model_path, tc, device=device)
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
