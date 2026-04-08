#!/usr/bin/env python3
"""
test_models_speculative.py — Measure speculative decoding speedup and correctness.

For each predefined model pair (target + draft), runs dotLLM CLI with and without
--speculative-model, comparing:
  - Correctness: outputs must match (greedy, temp=0)
  - Decode throughput: tok/s with vs without speculative decoding
  - Acceptance rate: proportion of draft tokens accepted by target

Usage:
    python scripts/test_models_speculative.py                    # run all pairs
    python scripts/test_models_speculative.py --filter llama     # only matching pairs
    python scripts/test_models_speculative.py --list             # show available pairs
    python scripts/test_models_speculative.py --runs 3           # best of N runs
    python scripts/test_models_speculative.py --max-tokens 64    # override generation length
    python scripts/test_models_speculative.py --speculative-k 3  # override draft candidates
    python scripts/test_models_speculative.py --save results.json
    python scripts/test_models_speculative.py --show results.json
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
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_compare import resolve_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_PROJECT = REPO_ROOT / "src" / "DotLLM.Cli"


@dataclass
class SpecPair:
    """A target + draft model pair for speculative decoding testing."""
    name: str
    target_repo: str
    target_quant: str | None
    draft_repo: str
    draft_quant: str | None
    arch: str
    prompt: str
    expected: str         # substring that must appear in output
    max_tokens: int = 32
    speculative_k: int = 5
    notes: str = ""


SPEC_PAIRS: list[SpecPair] = [
    SpecPair(
        name="Llama-3.2 3B+1B",
        target_repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        target_quant="Q8_0",
        draft_repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
        draft_quant="Q4_K_M",
        arch="Llama",
        prompt="Explain what a CPU cache is in one sentence.",
        expected="cache",
        max_tokens=48,
        notes="3B target Q8_0, 1B draft Q4_K_M, same vocab 128256",
    ),
    SpecPair(
        name="Qwen2.5 3B+0.5B",
        target_repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
        target_quant="Q8_0",
        draft_repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        draft_quant="Q8_0",
        arch="Qwen",
        prompt="Explain what a CPU cache is in one sentence.",
        expected="cache",
        max_tokens=48,
        notes="3B target Q8_0, 0.5B draft Q8_0, same vocab 151936, 6x size ratio",
    ),
    SpecPair(
        name="Qwen2.5 7B+1.5B",
        target_repo="bartowski/Qwen2.5-7B-Instruct-GGUF",
        target_quant="Q8_0",
        draft_repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        draft_quant="Q8_0",
        arch="Qwen",
        prompt="Explain what a CPU cache is in one sentence.",
        expected="cache",
        max_tokens=48,
        notes="7B target (152064) + 1.5B draft (151936), 128-token vocab diff, ~5x size ratio",
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    passed: bool
    text: str
    elapsed_s: float
    decode_tok_s: float = 0.0
    prefill_tok_s: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    generated_tokens: int = 0
    spec_drafted: int = 0
    spec_accepted: int = 0
    spec_rate: float = 0.0
    error: str | None = None


def _run_once(
    target_path: Path,
    draft_path: Path | None,
    pair: SpecPair,
    max_tokens: int,
    speculative_k: int,
) -> RunResult:
    """Run dotLLM CLI once and capture results."""
    cmd = [
        "dotnet", "run",
        "--project", str(CLI_PROJECT),
        "-c", "Release", "--",
        "run", str(target_path),
        "-p", pair.prompt,
        "-n", str(max_tokens),
        "-t", "0",
        "--json",
    ]
    if draft_path is not None:
        cmd += ["--speculative-model", str(draft_path)]
        cmd += ["--speculative-k", str(speculative_k)]

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return RunResult(False, "", time.monotonic() - start, error="TIMEOUT (300s)")

    elapsed = time.monotonic() - start

    if result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip()
        for line in err.splitlines():
            if "Error" in line or "error" in line:
                return RunResult(False, "", elapsed, error=line.strip()[:200])
        return RunResult(False, "", elapsed, error=f"exit {result.returncode}: {err[:200]}")

    # Parse JSON
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
    passed = pair.expected.lower() in text.lower()

    return RunResult(
        passed=passed,
        text=text.strip()[:120],
        elapsed_s=elapsed,
        decode_tok_s=timings.get("decode_tok_s", 0) or 0,
        prefill_tok_s=timings.get("prefill_tok_s", 0) or 0,
        prefill_ms=timings.get("prefill_ms", 0) or 0,
        decode_ms=timings.get("decode_ms", 0) or 0,
        generated_tokens=timings.get("generated_tokens", 0) or 0,
        spec_drafted=timings.get("speculative_draft_tokens", 0) or 0,
        spec_accepted=timings.get("speculative_accepted_tokens", 0) or 0,
        spec_rate=timings.get("speculative_acceptance_rate", 0) or 0,
    )


def _best_of(
    target_path: Path,
    draft_path: Path | None,
    pair: SpecPair,
    runs: int,
    max_tokens: int,
    speculative_k: int,
) -> RunResult:
    """Run N times, return the one with best decode tok/s."""
    results = []
    for _ in range(runs):
        r = _run_once(target_path, draft_path, pair, max_tokens, speculative_k)
        results.append(r)
    passed = [r for r in results if r.passed]
    if passed:
        return max(passed, key=lambda r: r.decode_tok_s)
    return results[0]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class CompareResult:
    name: str
    arch: str
    notes: str
    runs: int
    max_tokens: int
    speculative_k: int
    # Baseline (no draft)
    base_passed: bool
    base_text: str
    base_elapsed_s: float
    base_decode_tok_s: float
    base_prefill_tok_s: float
    base_decode_ms: float
    base_generated: int
    # Speculative
    spec_passed: bool
    spec_text: str
    spec_elapsed_s: float
    spec_decode_tok_s: float
    spec_prefill_tok_s: float
    spec_decode_ms: float
    spec_generated: int
    spec_drafted: int
    spec_accepted: int
    spec_rate: float
    # Derived
    decode_speedup: float   # spec_decode_tok_s / base_decode_tok_s
    wall_speedup: float     # base_elapsed_s / spec_elapsed_s
    text_match: bool        # output texts match
    error: str | None = None


def _run_comparison(
    target_path: Path,
    draft_path: Path,
    pair: SpecPair,
    runs: int,
    max_tokens: int,
    speculative_k: int,
) -> CompareResult:
    base = _best_of(target_path, None, pair, runs, max_tokens, speculative_k)
    spec = _best_of(target_path, draft_path, pair, runs, max_tokens, speculative_k)

    decode_speedup = spec.decode_tok_s / base.decode_tok_s if base.decode_tok_s > 0 else 0
    wall_speedup = base.elapsed_s / spec.elapsed_s if spec.elapsed_s > 0 else 0

    # Normalize for comparison: strip trailing whitespace/punctuation
    base_norm = base.text.strip().rstrip(".")
    spec_norm = spec.text.strip().rstrip(".")
    text_match = base_norm == spec_norm

    error = None
    if base.error:
        error = f"baseline: {base.error}"
    elif spec.error:
        error = f"speculative: {spec.error}"

    return CompareResult(
        name=pair.name, arch=pair.arch, notes=pair.notes,
        runs=runs, max_tokens=max_tokens, speculative_k=speculative_k,
        base_passed=base.passed, base_text=base.text,
        base_elapsed_s=base.elapsed_s, base_decode_tok_s=base.decode_tok_s,
        base_prefill_tok_s=base.prefill_tok_s, base_decode_ms=base.decode_ms,
        base_generated=base.generated_tokens,
        spec_passed=spec.passed, spec_text=spec.text,
        spec_elapsed_s=spec.elapsed_s, spec_decode_tok_s=spec.decode_tok_s,
        spec_prefill_tok_s=spec.prefill_tok_s, spec_decode_ms=spec.decode_ms,
        spec_generated=spec.generated_tokens,
        spec_drafted=spec.spec_drafted, spec_accepted=spec.spec_accepted,
        spec_rate=spec.spec_rate,
        decode_speedup=decode_speedup, wall_speedup=wall_speedup,
        text_match=text_match, error=error,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_results(results: list[CompareResult]) -> None:
    print()
    print(f"  {'Pair':<22} {'Arch':<6} │ "
          f"{'Base dec':>9} {'Spec dec':>9} {'Speedup':>8} │ "
          f"{'Accept':>7} {'Drafted':>8} {'Accepted':>9} │ "
          f"{'Match':>6} {'OK'}")
    print(f"  {'':22} {'':6} │ "
          f"{'(tok/s)':>9} {'(tok/s)':>9} {'':>8} │ "
          f"{'rate':>7} {'':>8} {'':>9} │ "
          f"{'':>6}")
    print("  " + "─" * 105)

    for r in results:
        if r.error:
            print(f"  {r.name:<22} {r.arch:<6} │  ERROR: {r.error}")
            continue

        speedup_str = f"{r.decode_speedup:.2f}x"
        rate_str = f"{r.spec_rate * 100:.0f}%" if r.spec_rate > 0 else "—"
        match_str = "yes" if r.text_match else "DIFF"
        base_ok = "+" if r.base_passed else "X"
        spec_ok = "+" if r.spec_passed else "X"

        print(
            f"  {r.name:<22} {r.arch:<6} │ "
            f"{r.base_decode_tok_s:>8.1f}  {r.spec_decode_tok_s:>8.1f}  {speedup_str:>8} │ "
            f"{rate_str:>7} {r.spec_drafted:>8} {r.spec_accepted:>9} │ "
            f"{match_str:>6} base:{base_ok} spec:{spec_ok}"
        )

    print("  " + "─" * 105)
    print()
    print("  Speedup   = spec decode tok/s / baseline decode tok/s (>1 = speculative is faster)")
    print("  Accept    = draft tokens accepted by target model (higher = more benefit)")
    print("  Match     = greedy output identical with and without speculative decoding")
    print()

    valid = [r for r in results if not r.error]
    if valid:
        avg_speedup = sum(r.decode_speedup for r in valid) / len(valid)
        avg_rate = sum(r.spec_rate for r in valid) / len(valid)
        all_match = all(r.text_match for r in valid)
        all_correct = all(r.base_passed and r.spec_passed for r in valid)
        print(f"  Summary ({len(valid)} pairs):")
        print(f"    Avg decode speedup:    {avg_speedup:.2f}x")
        print(f"    Avg acceptance rate:   {avg_rate * 100:.0f}%")
        print(f"    Output match:          {'all identical' if all_match else 'DIFFERENCES DETECTED'}")
        print(f"    Correctness:           {'all passed' if all_correct else 'FAILURES'}")
        print()


def _save_results(path: str, results: list[CompareResult], args: argparse.Namespace) -> None:
    export = {
        "label": "speculative_comparison",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "system": {
            "cpu": platform.processor() or platform.machine(),
            "cores": os.cpu_count() or 0,
            "os": f"{platform.system()} {platform.release()}",
        },
        "config": {
            "runs": args.runs,
            "max_tokens": args.max_tokens,
            "speculative_k": args.speculative_k,
        },
        "results": [
            {
                "name": r.name, "arch": r.arch, "notes": r.notes,
                "runs": r.runs, "max_tokens": r.max_tokens,
                "speculative_k": r.speculative_k,
                "base_passed": r.base_passed, "base_text": r.base_text,
                "base_elapsed_s": round(r.base_elapsed_s, 3),
                "base_decode_tok_s": round(r.base_decode_tok_s, 2),
                "base_decode_ms": round(r.base_decode_ms, 1),
                "base_generated": r.base_generated,
                "spec_passed": r.spec_passed, "spec_text": r.spec_text,
                "spec_elapsed_s": round(r.spec_elapsed_s, 3),
                "spec_decode_tok_s": round(r.spec_decode_tok_s, 2),
                "spec_decode_ms": round(r.spec_decode_ms, 1),
                "spec_generated": r.spec_generated,
                "spec_drafted": r.spec_drafted,
                "spec_accepted": r.spec_accepted,
                "spec_rate": round(r.spec_rate, 4),
                "decode_speedup": round(r.decode_speedup, 3),
                "wall_speedup": round(r.wall_speedup, 3),
                "text_match": r.text_match,
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

    print(f"[show] {path}")
    ts = data.get("timestamp", "")
    if ts:
        print(f"  Time:   {ts}")
    system = data.get("system", {})
    if system:
        print(f"  System: {system.get('cpu', '?')} ({system.get('cores', '?')} cores)")
    config = data.get("config", {})
    print(f"  Config: runs={config.get('runs', '?')}, "
          f"max_tokens={config.get('max_tokens', '?')}, "
          f"K={config.get('speculative_k', '?')}")

    results = [
        CompareResult(
            name=r["name"], arch=r["arch"], notes=r.get("notes", ""),
            runs=r["runs"], max_tokens=r["max_tokens"],
            speculative_k=r["speculative_k"],
            base_passed=r["base_passed"], base_text=r["base_text"],
            base_elapsed_s=r["base_elapsed_s"],
            base_decode_tok_s=r["base_decode_tok_s"],
            base_prefill_tok_s=0, base_decode_ms=r["base_decode_ms"],
            base_generated=r["base_generated"],
            spec_passed=r["spec_passed"], spec_text=r["spec_text"],
            spec_elapsed_s=r["spec_elapsed_s"],
            spec_decode_tok_s=r["spec_decode_tok_s"],
            spec_prefill_tok_s=0, spec_decode_ms=r["spec_decode_ms"],
            spec_generated=r["spec_generated"],
            spec_drafted=r["spec_drafted"],
            spec_accepted=r["spec_accepted"],
            spec_rate=r["spec_rate"],
            decode_speedup=r["decode_speedup"],
            wall_speedup=r["wall_speedup"],
            text_match=r["text_match"],
            error=r.get("error"),
        )
        for r in results_raw
    ]
    _print_results(results)
    return 0


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def _is_cached(repo: str, quant: str | None) -> bool:
    """Check if a model is in the local cache."""
    models_dir = Path.home() / ".dotllm" / "models"
    repo_dir = models_dir / repo.replace("/", os.sep)
    if not repo_dir.exists():
        return False
    files = [f.name for f in repo_dir.iterdir() if f.suffix == ".gguf"]
    if quant:
        files = [f for f in files if quant.lower() in f.lower()]
    return len(files) > 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure speculative decoding speedup and correctness."
    )
    parser.add_argument("--filter", type=str, default=None,
                        help="Comma-separated name/arch substrings to match")
    parser.add_argument("--list", action="store_true",
                        help="List available test pairs and exit")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per config (best-of-N, default: 1)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max tokens for generation")
    parser.add_argument("--speculative-k", type=int, default=None,
                        help="Override number of draft candidates (K)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--show", type=str, default=None,
                        help="Load and display results from JSON (no tests run)")
    args = parser.parse_args()

    if args.show:
        return _show_results(args.show)

    pairs = SPEC_PAIRS

    if args.filter:
        filters = [f.strip().lower() for f in args.filter.split(",")]
        pairs = [
            p for p in pairs
            if any(f in p.name.lower() or f in p.arch.lower() for f in filters)
        ]
        if not pairs:
            print(f"No pairs match filter '{args.filter}'.")
            return 1

    # Check cached
    cached_pairs = [
        p for p in pairs
        if _is_cached(p.target_repo, p.target_quant) and _is_cached(p.draft_repo, p.draft_quant)
    ]

    if args.list:
        print(f"  {'Pair':<22} {'Arch':<6} {'Target':<40} {'Draft':<35} {'Cached'}")
        print("  " + "─" * 120)
        for p in pairs:
            target_label = f"{p.target_repo} ({p.target_quant or 'default'})"
            draft_label = f"{p.draft_repo} ({p.draft_quant or 'default'})"
            cached = "yes" if p in cached_pairs else "no"
            print(f"  {p.name:<22} {p.arch:<6} {target_label:<40} {draft_label:<35} {cached}")
        return 0

    if not cached_pairs:
        print("No model pairs fully cached. Download models first:")
        for p in pairs:
            if not _is_cached(p.target_repo, p.target_quant):
                print(f"  dotllm model pull {p.target_repo}")
            if not _is_cached(p.draft_repo, p.draft_quant):
                print(f"  dotllm model pull {p.draft_repo}")
        return 1

    print(f"Speculative decoding comparison: {len(cached_pairs)} pair(s), "
          f"{args.runs} run(s)")
    print()

    results: list[CompareResult] = []

    for i, pair in enumerate(cached_pairs, 1):
        max_tokens = args.max_tokens or pair.max_tokens
        spec_k = args.speculative_k or pair.speculative_k

        print(f"[{i}/{len(cached_pairs)}] {pair.name} (K={spec_k}, max_tokens={max_tokens})")

        try:
            target_path = resolve_model(pair.target_repo, pair.target_quant, quiet=True)
            draft_path = resolve_model(pair.draft_repo, pair.draft_quant, quiet=True)
        except SystemExit:
            print(f"  SKIP — model resolution failed")
            continue

        print(f"  target: {target_path.name}")
        print(f"  draft:  {draft_path.name}")

        # Baseline run
        print(f"  running baseline (no draft)...", end="", flush=True)
        base = _best_of(target_path, None, pair, args.runs, max_tokens, spec_k)
        if base.error:
            print(f" ERROR: {base.error}")
        else:
            print(f" {base.decode_tok_s:.1f} tok/s, {base.generated_tokens} tokens")

        # Speculative run
        print(f"  running speculative (K={spec_k})...", end="", flush=True)
        spec = _best_of(target_path, draft_path, pair, args.runs, max_tokens, spec_k)
        if spec.error:
            print(f" ERROR: {spec.error}")
        else:
            rate_str = f"{spec.spec_rate * 100:.0f}%" if spec.spec_rate > 0 else "—"
            print(f" {spec.decode_tok_s:.1f} tok/s, {spec.generated_tokens} tokens, "
                  f"acceptance {rate_str}")

        # Build comparison
        decode_speedup = spec.decode_tok_s / base.decode_tok_s if base.decode_tok_s > 0 else 0
        wall_speedup = base.elapsed_s / spec.elapsed_s if spec.elapsed_s > 0 else 0
        base_norm = base.text.strip().rstrip(".")
        spec_norm = spec.text.strip().rstrip(".")
        text_match = base_norm == spec_norm

        error = None
        if base.error:
            error = f"baseline: {base.error}"
        elif spec.error:
            error = f"speculative: {spec.error}"

        r = CompareResult(
            name=pair.name, arch=pair.arch, notes=pair.notes,
            runs=args.runs, max_tokens=max_tokens, speculative_k=spec_k,
            base_passed=base.passed, base_text=base.text,
            base_elapsed_s=base.elapsed_s, base_decode_tok_s=base.decode_tok_s,
            base_prefill_tok_s=base.prefill_tok_s, base_decode_ms=base.decode_ms,
            base_generated=base.generated_tokens,
            spec_passed=spec.passed, spec_text=spec.text,
            spec_elapsed_s=spec.elapsed_s, spec_decode_tok_s=spec.decode_tok_s,
            spec_prefill_tok_s=spec.prefill_tok_s, spec_decode_ms=spec.decode_ms,
            spec_generated=spec.generated_tokens,
            spec_drafted=spec.spec_drafted, spec_accepted=spec.spec_accepted,
            spec_rate=spec.spec_rate,
            decode_speedup=decode_speedup, wall_speedup=wall_speedup,
            text_match=text_match, error=error,
        )
        results.append(r)

        if not error:
            match_str = "MATCH" if text_match else "DIFF"
            print(f"  -> decode {decode_speedup:.2f}x, acceptance {spec.spec_rate*100:.0f}%, "
                  f"output {match_str}")
        print()

    _print_results(results)

    if args.save:
        _save_results(args.save, results, args)

    failures = sum(1 for r in results if not r.error and (not r.base_passed or not r.spec_passed))
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
