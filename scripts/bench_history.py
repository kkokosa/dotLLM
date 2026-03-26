#!/usr/bin/env python3
"""
bench_history.py — Benchmark across git commits.

Creates git worktrees for each commit, runs bench_compare.py in each, and
displays a trend table at the end. The user's working tree stays undisturbed.

Usage:
    python scripts/bench_history.py cpu --last 5
    python scripts/bench_history.py gpu --last 5 --device gpu
    python scripts/bench_history.py cpu --from f3d3bf8
    python scripts/bench_history.py cpu --last 3 --model QuantFactory/SmolLM-135M-GGUF --quant Q8_0
"""

from __future__ import annotations

import argparse
import atexit
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import stdev

# ---------------------------------------------------------------------------
# Import resolve_model from bench_compare (same directory)
# ---------------------------------------------------------------------------
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from bench_compare import resolve_model  # noqa: E402


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CommitInfo:
    full_hash: str
    short_hash: str
    subject: str
    source: str = ""  # e.g. "main" or branch name — for display only


# ---------------------------------------------------------------------------
# Worktree management
# ---------------------------------------------------------------------------

_active_worktrees: list[str] = []


def _cleanup_worktrees() -> None:
    """atexit handler — remove any leftover worktrees."""
    for path in list(_active_worktrees):
        _remove_worktree(path)
    # Final prune in case force-remove left stale refs
    if _active_worktrees:
        subprocess.run(["git", "worktree", "prune"], capture_output=True)


atexit.register(_cleanup_worktrees)


def _remove_readonly(func, path, _exc_info):
    """Error handler for shutil.rmtree on Windows — clear read-only and retry."""
    os.chmod(path, 0o777)
    func(path)


def _remove_worktree(path: str) -> None:
    """Remove a git worktree. Falls back to shutil.rmtree + prune on failure."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", path, "--force"],
            capture_output=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass

    # If the directory still exists, force-delete it
    if os.path.isdir(path):
        shutil.rmtree(path, onerror=_remove_readonly)
        subprocess.run(["git", "worktree", "prune"], capture_output=True)

    if path in _active_worktrees:
        _active_worktrees.remove(path)


def _create_worktree(commit_hash: str) -> str:
    """Create a detached worktree for the given commit. Returns the worktree path."""
    tmpdir = tempfile.mkdtemp(prefix="dotllm-bench-")
    result = subprocess.run(
        ["git", "worktree", "add", "--detach", tmpdir, commit_hash],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f"git worktree add failed: {result.stderr.strip()}")
    _active_worktrees.append(tmpdir)
    return tmpdir


# ---------------------------------------------------------------------------
# Commit enumeration
# ---------------------------------------------------------------------------

def _current_branch() -> str:
    """Return the current branch name, or 'HEAD' if detached."""
    return subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        text=True,
    ).strip()


def _main_branch() -> str:
    """Return the main branch name (main or master)."""
    for name in ("main", "master"):
        result = subprocess.run(
            ["git", "rev-parse", "--verify", name],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return name
    return "main"  # fallback


def _parse_log_lines(lines: list[str]) -> list[CommitInfo]:
    """Parse `git log --format='%H %h %s'` output into CommitInfo list (oldest-first)."""
    commits = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(" ", 2)
        commits.append(CommitInfo(
            full_hash=parts[0],
            short_hash=parts[1],
            subject=parts[2] if len(parts) > 2 else "",
        ))
    # git log outputs newest-first; reverse to oldest-first
    commits.reverse()
    return commits


def get_commits(last: int | None, from_hash: str | None, branch: str | None) -> list[CommitInfo]:
    """Return commits oldest-first.

    Default (no --from, no --branch): last N from main, plus current HEAD
    if on a different branch.
    """
    if from_hash:
        # Check if from_hash is the root commit (has no parent)
        parent_check = subprocess.run(
            ["git", "rev-parse", "--verify", f"{from_hash}^"],
            capture_output=True, text=True, encoding="utf-8",
        )
        if parent_check.returncode != 0:
            # Root commit — include it directly
            range_spec = f"{from_hash}..HEAD"
            root_line = subprocess.check_output(
                ["git", "log", "--format=%H %h %s", "-1", from_hash],
                text=True, encoding="utf-8",
            ).strip()
            lines = [root_line]
            extra = subprocess.check_output(
                ["git", "log", "--format=%H %h %s", range_spec],
                text=True, encoding="utf-8",
            ).strip()
            if extra:
                lines.extend(extra.splitlines())
        else:
            lines = subprocess.check_output(
                ["git", "log", "--format=%H %h %s", f"{from_hash}^..HEAD"],
                text=True, encoding="utf-8",
            ).strip().splitlines()
        return _parse_log_lines(lines)

    # Determine which branch to pull history from
    source_branch = branch or _main_branch()
    n = last or 5

    lines = subprocess.check_output(
        ["git", "log", "--format=%H %h %s", f"-{n}", source_branch],
        text=True, encoding="utf-8",
    ).strip().splitlines()
    commits = _parse_log_lines(lines)
    for c in commits:
        c.source = source_branch
    main_hashes = {c.full_hash for c in commits}

    # If we're on a different branch and no explicit --branch was given,
    # append branch-only commits (everything on HEAD not on main)
    current = _current_branch()
    if not branch and current not in ("HEAD", source_branch):
        branch_lines = subprocess.check_output(
            ["git", "log", "--format=%H %h %s", f"{source_branch}..HEAD"],
            text=True, encoding="utf-8",
        ).strip().splitlines()
        branch_commits = _parse_log_lines(branch_lines)  # oldest-first
        for bc in branch_commits:
            if bc.full_hash not in main_hashes:
                bc.source = current
                commits.append(bc)

    return commits


# ---------------------------------------------------------------------------
# Existing results cleanup
# ---------------------------------------------------------------------------

def _find_existing_results(name: str, output_dir: str) -> list[str]:
    """Find existing result files matching the name pattern."""
    pattern = os.path.join(output_dir, f"{name}_*.json")
    return sorted(glob.glob(pattern))


def _prompt_delete_existing(name: str, output_dir: str) -> None:
    """If result files for this name already exist, ask the user to delete them."""
    existing = _find_existing_results(name, output_dir)
    if not existing:
        return

    print(f"\nFound {len(existing)} existing result(s) for \"{name}\":")
    for f in existing:
        print(f"  {os.path.basename(f)}")

    try:
        answer = input("Delete them and start fresh? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        answer = "n"

    if answer in ("", "y", "yes"):
        for f in existing:
            os.remove(f)
        print(f"  Deleted {len(existing)} file(s).")
    else:
        print("  Keeping existing files (will skip commits that already have results).")


# ---------------------------------------------------------------------------
# Git metadata patching
# ---------------------------------------------------------------------------

def _patch_git_metadata(output_file: str, commit: CommitInfo) -> None:
    """Patch the exported JSON so git.commit/branch/dirty reflect the actual commit."""
    try:
        with open(output_file) as f:
            data = json.load(f)
        data.setdefault("git", {})
        data["git"]["commit"] = commit.full_hash
        data["git"]["branch"] = "detached"
        data["git"]["dirty"] = False
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
    except (json.JSONDecodeError, OSError):
        pass  # best-effort — don't fail the benchmark over metadata


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

def _build_bench_compare_cmd(
    model_path: str,
    worktree_path: str,
    output_path: str,
    label: str,
    prompt_size: str,
    tokens: int,
    runs: int,
    iterations: int | None,
    dotllm: bool,
    llamacpp: bool,
    llamacpp_bin: str | None,
    device: str = "cpu",
) -> list[str]:
    """Build the subprocess command for bench_compare.py."""
    # Run bench_compare.py from the current tree (not the worktree) —
    # only the BDN project comes from the worktree.
    bench_script = str(Path(__file__).resolve().parent / "bench_compare.py")
    worktree_csproj = str(
        Path(worktree_path) / "benchmarks" / "DotLLM.Benchmarks" / "DotLLM.Benchmarks.csproj"
    )

    cmd = [
        sys.executable, bench_script,
        "--model", model_path,
        "--bdn-project", worktree_csproj,
        "--export-json", output_path,
        "--label", label,
        "--prompt-size", prompt_size,
        "--tokens", str(tokens),
        "--runs", str(runs),
    ]
    if iterations is not None:
        cmd.extend(["--iterations", str(iterations)])

    if dotllm:
        cmd.append("--dotllm")
    if llamacpp:
        cmd.append("--llamacpp")
    if llamacpp_bin:
        cmd.extend(["--llamacpp-bin", llamacpp_bin])
    if device != "cpu":
        cmd.extend(["--device", device])

    return cmd


def _run_bench_for_commit(
    index: int,
    total: int,
    commit: CommitInfo,
    name: str,
    model_path: str,
    output_dir: str,
    prompt_size: str,
    tokens: int,
    runs: int,
    iterations: int | None,
    dotllm: bool,
    llamacpp: bool,
    llamacpp_bin: str | None,
    device: str = "cpu",
) -> bool:
    """Run benchmark for a single commit. Returns True on success."""
    output_file = os.path.join(output_dir, f"{name}_{index}.json")
    label = f"{name}_{index} ({commit.short_hash})"
    subject_preview = commit.subject[:60] + "..." if len(commit.subject) > 60 else commit.subject

    print(f"\n[{index + 1}/{total}] {commit.short_hash} — {subject_preview}")

    # Skip if output already exists (resume support)
    if os.path.isfile(output_file):
        print(f"  [skip] {os.path.basename(output_file)} already exists")
        return True

    # Create worktree
    print("  Creating worktree...")
    worktree_path = None
    try:
        worktree_path = _create_worktree(commit.full_hash)
    except RuntimeError as e:
        print(f"  [error] {e}", file=sys.stderr)
        return False

    start = time.monotonic()
    try:
        # Check that the benchmark project exists in this worktree
        worktree_csproj = (
            Path(worktree_path) / "benchmarks" / "DotLLM.Benchmarks" / "DotLLM.Benchmarks.csproj"
        )
        if not worktree_csproj.exists():
            print(f"  [skip] No benchmark project in this commit", file=sys.stderr)
            return False

        cmd = _build_bench_compare_cmd(
            model_path=model_path,
            worktree_path=worktree_path,
            output_path=output_file,
            label=label,
            prompt_size=prompt_size,
            tokens=tokens,
            runs=runs,
            iterations=iterations,
            dotllm=dotllm,
            llamacpp=llamacpp,
            llamacpp_bin=llamacpp_bin,
            device=device,
        )

        print(f"  Running bench_compare...")
        result = subprocess.run(cmd, timeout=1800)  # 30 min timeout per commit

        elapsed = time.monotonic() - start
        mins, secs = divmod(int(elapsed), 60)

        if result.returncode != 0:
            print(f"  [error] bench_compare exited with code {result.returncode} ({mins}m {secs:02d}s)")
            return False

        if os.path.isfile(output_file):
            _patch_git_metadata(output_file, commit)
            print(f"  Done ({mins}m {secs:02d}s). Exported {os.path.basename(output_file)}")
            return True
        else:
            print(f"  [error] Expected output file not created ({mins}m {secs:02d}s)")
            return False

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        mins, secs = divmod(int(elapsed), 60)
        print(f"  [error] Timed out after {mins}m {secs:02d}s", file=sys.stderr)
        return False

    finally:
        if worktree_path:
            _remove_worktree(worktree_path)


# ---------------------------------------------------------------------------
# Trend display — tables with CV and commit-to-commit deltas
# ---------------------------------------------------------------------------

_has_rich = False
try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    _has_rich = True
except ImportError:
    pass


def _find_result_files(name: str, output_dir: str) -> list[str]:
    """Find result files for a run name, sorted by index."""
    pattern = os.path.join(output_dir, f"{name}_*.json")
    files = sorted(glob.glob(pattern))
    return files


@dataclass
class _TrendRow:
    label: str
    date: str
    prefill: float
    decode: float
    cv: float  # 0 = no data


def _extract_cv(result: dict) -> float:
    """Extract or compute decode CV from a result dict."""
    cv = result.get("decode_cv", 0)
    if cv:
        return cv
    vals = result.get("all_decode_tok_per_sec")
    if vals and len(vals) >= 2:
        mean = sum(vals) / len(vals)
        if mean > 0:
            return stdev(vals) / mean
    return 0


def _delta_text(old: float, new: float, cv: float = 0) -> tuple[str, str]:
    """Format (text, rich_style) for a throughput delta. Higher-is-better."""
    if old <= 0:
        return "", "dim"
    pct = (new - old) / old * 100
    sign = "+" if pct >= 0 else ""
    text = f"{sign}{pct:.1f}%"
    threshold = max(cv * 100, 1.0)
    if abs(pct) < threshold:
        return f"~{text}", "dim"
    return text, "green" if pct > 0 else "red"


def _load_trend_rows(
    files: list[str], engine_filter: str,
) -> dict[str, list[_TrendRow]]:
    """Load result files, return {model: [_TrendRow in file order]}."""
    eng_low = engine_filter.lower()
    model_rows: dict[str, list[_TrendRow]] = {}

    for fpath in files:
        try:
            with open(fpath) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        label = raw.get("label", Path(fpath).stem)
        date = raw.get("timestamp", "")[:10]

        for r in raw.get("results", []):
            eng = r.get("engine", "dotLLM")
            if eng_low != "all" and eng_low not in eng.lower():
                continue
            model = r.get("model", "?")
            model_rows.setdefault(model, []).append(_TrendRow(
                label=label,
                date=date,
                prefill=r.get("prefill_tok_per_sec", 0),
                decode=r.get("decode_tok_per_sec", 0),
                cv=_extract_cv(r),
            ))

    return model_rows


def _render_rich(model: str, rows: list[_TrendRow]) -> None:
    console = Console()
    tbl = Table(title=f"Benchmark History — {model}", show_lines=False, pad_edge=True)
    tbl.add_column("Label", style="bold cyan", no_wrap=True)
    tbl.add_column("Date", style="dim")
    tbl.add_column("Prefill tok/s", justify="right")
    tbl.add_column("%chg pf", justify="right")
    tbl.add_column("Decode tok/s", justify="right")
    tbl.add_column("%chg dc", justify="right")
    tbl.add_column("CV", justify="right", style="dim")

    prev: _TrendRow | None = None
    for r in rows:
        cv_text = f"{r.cv:.1%}" if r.cv > 0 else "-"
        if prev:
            noise = max(r.cv, prev.cv)
            pf_dt, pf_ds = _delta_text(prev.prefill, r.prefill, noise)
            dc_dt, dc_ds = _delta_text(prev.decode, r.decode, noise)
        else:
            pf_dt = dc_dt = ""
            pf_ds = dc_ds = "dim"

        tbl.add_row(
            r.label, r.date,
            Text(f"{r.prefill:.1f}"),
            Text(pf_dt, style=pf_ds),
            Text(f"{r.decode:.1f}"),
            Text(dc_dt, style=dc_ds),
            cv_text,
        )
        prev = r

    console.print()
    console.print(tbl)
    console.print()


def _render_plain(model: str, rows: list[_TrendRow]) -> None:
    print(f"\nBenchmark History — {model}")
    hdr = f"{'Label':<25} {'Date':<12} {'Prefill':>10} {'%pf':>8} {'Decode':>10} {'%dc':>8} {'CV':>6}"
    print(hdr)
    print("-" * len(hdr))

    prev: _TrendRow | None = None
    for r in rows:
        cv_text = f"{r.cv:.1%}" if r.cv > 0 else "-"
        if prev:
            noise = max(r.cv, prev.cv)
            pf_dt, _ = _delta_text(prev.prefill, r.prefill, noise)
            dc_dt, _ = _delta_text(prev.decode, r.decode, noise)
        else:
            pf_dt = dc_dt = ""
        print(f"{r.label:<25} {r.date:<12} {r.prefill:>10.1f} {pf_dt:>8} {r.decode:>10.1f} {dc_dt:>8} {cv_text:>6}")
    print()


def _render_md(model: str, rows: list[_TrendRow]) -> None:
    print(f"\n## Benchmark History — {model}\n")
    print("| Label | Date | Prefill tok/s | %chg pf | Decode tok/s | %chg dc | CV |")
    print("|-------|------|--------------|-----------|-------------|----------|-----|")

    prev: _TrendRow | None = None
    for r in rows:
        cv_text = f"{r.cv:.1%}" if r.cv > 0 else "-"
        if prev:
            noise = max(r.cv, prev.cv)
            pf_dt, _ = _delta_text(prev.prefill, r.prefill, noise)
            dc_dt, _ = _delta_text(prev.decode, r.decode, noise)
        else:
            pf_dt = dc_dt = ""
        print(f"| {r.label} | {r.date} | {r.prefill:.1f} | {pf_dt} | {r.decode:.1f} | {dc_dt} | {cv_text} |")
    print()


_CV_LEGEND = (
    "CV (coefficient of variation) measures run-to-run noise as stddev/mean of decode throughput.\n"
    "Values below ~5% are stable; above ~15% means high variance -- treat deltas with caution.\n"
    "'-' means only a single iteration was recorded, so CV cannot be computed."
)


def _show_trend(
    name: str,
    output_dir: str,
    engine_filter: str,
    use_md: bool,
) -> None:
    """Build per-model trend tables with CV and commit-to-commit deltas."""
    files = _find_result_files(name, output_dir)

    if len(files) < 2:
        print("  Not enough results for a trend table.")
        return

    model_rows = _load_trend_rows(files, engine_filter)

    for model, rows in model_rows.items():
        if use_md:
            _render_md(model, rows)
        elif _has_rich:
            _render_rich(model, rows)
        else:
            _render_plain(model, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark dotLLM across git commits using worktrees.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/bench_history.py cpu --last 5\n"
            "  python scripts/bench_history.py cpu --from f3d3bf8\n"
            "  python scripts/bench_history.py cpu --last 3 --model QuantFactory/SmolLM-135M-GGUF --quant Q8_0\n"
        ),
    )
    parser.add_argument("name", type=str,
                        help="Prefix for output files (e.g. 'cpu' -> cpu_0.json, cpu_1.json, ...)")

    # Commit range — mutually exclusive
    range_group = parser.add_mutually_exclusive_group()
    range_group.add_argument("--last", type=int, default=None,
                             help="Last N commits on current branch (default: 5)")
    range_group.add_argument("--from", dest="from_hash", type=str, default=None,
                             help="From this commit to HEAD (inclusive)")
    parser.add_argument("--branch", type=str, default=None,
                        help="Branch to take commits from (default: main + current HEAD if on a feature branch)")

    # Model resolution
    parser.add_argument("--model", type=str, default=None,
                        help="HF repo ID(s) or .gguf path(s), comma-separated")
    parser.add_argument("--quant", type=str, default=None,
                        help="Quantization filter(s), comma-separated")

    # Benchmark parameters (passed through)
    parser.add_argument("--prompt-size", type=str, default="short",
                        choices=["short", "medium", "large"],
                        help="Prompt size (default: short)")
    parser.add_argument("--tokens", type=int, default=20,
                        help="Max tokens to generate (default: 20)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per commit (default: 5)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override BDN iteration count (default: per-benchmark, typically 5)")

    # Engine selection
    parser.add_argument("--dotllm", action="store_true",
                        help="Run dotLLM only (default when neither flag given)")
    parser.add_argument("--llamacpp", action="store_true",
                        help="Run llama.cpp only")
    parser.add_argument("--llamacpp-bin", type=str, default=None,
                        help="Path to llama-completion binary")

    # Output
    parser.add_argument("--output-dir", type=str, default="benchmarks/results",
                        help="Directory for output JSON files (default: benchmarks/results)")
    parser.add_argument("--show", action="store_true",
                        help="Show trend from existing results (don't run benchmarks)")
    parser.add_argument("--select", action="store_true",
                        help="Interactively select which commits to benchmark")
    parser.add_argument("--md", action="store_true",
                        help="Markdown output for trend table")
    parser.add_argument("--no-trend", action="store_true",
                        help="Skip final trend display")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "gpu", "both"],
                        help="Compute device: cpu (default), gpu, or both")

    args = parser.parse_args()

    # --show: just display existing results and exit
    if args.show:
        files = _find_result_files(args.name, args.output_dir)
        if not files:
            print(f"No results found for \"{args.name}\" in {args.output_dir}/", file=sys.stderr)
            return 1
        engine_filter = "dotLLM"
        if args.llamacpp and not args.dotllm:
            engine_filter = "llama.cpp"
        elif args.llamacpp and args.dotllm:
            engine_filter = "all"
        _show_trend(args.name, args.output_dir, engine_filter, args.md)
        print(_CV_LEGEND)
        return 0

    # Default to --dotllm when neither engine flag is specified
    dotllm = args.dotllm
    llamacpp = args.llamacpp
    if not dotllm and not llamacpp:
        dotllm = True

    # Default --last to 5 when --from is not specified
    last = args.last
    if last is None and args.from_hash is None:
        last = 5

    # 1. Enumerate commits
    commits = get_commits(last, args.from_hash, args.branch)
    if not commits:
        print("No commits found.", file=sys.stderr)
        return 1

    # 2. Resolve model(s) once (before entering the loop)
    #    Comma-separated --model and --quant, same as bench_compare.py
    model_paths: list[str] = []
    if args.model:
        model_specs = [m.strip() for m in args.model.split(",")]
        quant_specs = [q.strip() for q in args.quant.split(",")] if args.quant else [None]
        for model_spec in model_specs:
            for quant_spec in quant_specs:
                resolved = resolve_model(model_spec, quant_spec)
                model_paths.append(str(resolved.resolve()))
        print(f"[model] Resolved {len(model_paths)} model(s)")
    # Comma-joined absolute paths — bench_compare sees .gguf paths and skips resolution
    model_arg = ",".join(model_paths) if model_paths else None

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Check for existing results and offer to delete
    _prompt_delete_existing(args.name, args.output_dir)

    # Banner
    print(f'\n=== bench_history: "{args.name}" ===')
    if model_paths:
        for mp in model_paths:
            print(f"  Model: {mp}")
    print(f"  Output: {args.output_dir}/{args.name}_*.json")

    # Commit selection / confirmation
    # Compute a short label for the source branch (e.g. "main", "issue/66-..." -> "#66")
    _source_labels: dict[str, str] = {}
    for c in commits:
        if c.source and c.source not in _source_labels:
            label = c.source
            # Shorten "issue/NNN-..." to "#NNN"
            if label.startswith("issue/"):
                parts = label[6:].split("-", 1)
                label = f"#{parts[0]}"
            _source_labels[c.source] = label

    def _commit_display(c: CommitInfo) -> str:
        tag = f"[{_source_labels.get(c.source, c.source)}]" if c.source else ""
        subject = c.subject[:55] + "..." if len(c.subject) > 55 else c.subject
        return f"{c.short_hash} {tag:<8} {subject}"

    if args.select:
        try:
            from InquirerPy import inquirer
            from InquirerPy.separator import Separator
        except ImportError:
            print("--select requires InquirerPy: pip install InquirerPy", file=sys.stderr)
            return 1

        choices = []
        prev_source = None
        for c in commits:
            if c.source != prev_source:
                choices.append(Separator(f"--- {c.source} ---"))
                prev_source = c.source
            choices.append({
                "name": _commit_display(c),
                "value": c,
                "enabled": True,
            })

        try:
            commits = inquirer.checkbox(
                message="Select commits to benchmark (Space to toggle, Enter to confirm):",
                choices=choices,
                validate=lambda r: len(r) >= 1 or "Select at least one commit",
                instruction="(Space toggle, Ctrl+A toggle all, Enter confirm)",
                keybindings={
                    "toggle-all": [{"key": "c-a"}],
                    "toggle-all-true": [],
                },
            ).execute()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1

        if not commits:
            print("No commits selected.", file=sys.stderr)
            return 0
    else:
        print()
        for i, c in enumerate(commits):
            print(f"  [{i}] {_commit_display(c)}")
        print()

        try:
            answer = input(f"Run benchmarks for these {len(commits)} commits? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        if answer not in ("", "y", "yes"):
            print("Aborted.")
            return 0

    # 3. Run benchmarks
    total_start = time.monotonic()
    successes = 0
    failures: list[CommitInfo] = []

    for i, commit in enumerate(commits):
        ok = _run_bench_for_commit(
            index=i,
            total=len(commits),
            commit=commit,
            name=args.name,
            model_path=model_arg or "",
            output_dir=args.output_dir,
            prompt_size=args.prompt_size,
            tokens=args.tokens,
            runs=args.runs,
            iterations=args.iterations,
            dotllm=dotllm,
            llamacpp=llamacpp,
            llamacpp_bin=args.llamacpp_bin,
            device=args.device,
        )
        if ok:
            successes += 1
        else:
            failures.append(commit)

    # 4. Show trend
    if not args.no_trend:
        engine_filter = "dotLLM" if dotllm and not llamacpp else ("llama.cpp" if llamacpp and not dotllm else "all")
        print()
        _show_trend(args.name, args.output_dir, engine_filter, args.md)
        print(_CV_LEGEND)

    # 5. Summary
    total_elapsed = time.monotonic() - total_start
    mins, secs = divmod(int(total_elapsed), 60)

    print(f"\n=== Summary ===")
    print(f"  {len(commits)} commits, {successes} succeeded, {len(failures)} failed")
    print(f"  Total time: {mins}m {secs:02d}s")
    if failures:
        for c in failures:
            print(f"  Failed: {c.short_hash} ({c.subject[:50]})")
    print()

    return 0 if successes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
