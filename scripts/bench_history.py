#!/usr/bin/env python3
"""
bench_history.py — Benchmark across git commits.

Creates git worktrees for each commit, runs bench_compare.py in each, and
displays a trend table at the end. The user's working tree stays undisturbed.

Usage:
    python scripts/bench_history.py cpu --last 5
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

def get_commits(last: int | None, from_hash: str | None) -> list[CommitInfo]:
    """Return commits oldest-first."""
    if from_hash:
        # Check if from_hash is the root commit (has no parent)
        parent_check = subprocess.run(
            ["git", "rev-parse", "--verify", f"{from_hash}^"],
            capture_output=True, text=True,
        )
        if parent_check.returncode != 0:
            # Root commit — include it directly
            range_spec = f"{from_hash}..HEAD"
            # Get root commit separately
            root_line = subprocess.check_output(
                ["git", "log", "--format=%H %h %s", "-1", from_hash],
                text=True,
            ).strip()
            lines = [root_line]
            extra = subprocess.check_output(
                ["git", "log", "--format=%H %h %s", range_spec],
                text=True,
            ).strip()
            if extra:
                lines.extend(extra.splitlines())
        else:
            lines = subprocess.check_output(
                ["git", "log", "--format=%H %h %s", f"{from_hash}^..HEAD"],
                text=True,
            ).strip().splitlines()
    else:
        n = last or 5
        lines = subprocess.check_output(
            ["git", "log", "--format=%H %h %s", f"-{n}", "HEAD"],
            text=True,
        ).strip().splitlines()

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
# Trend display
# ---------------------------------------------------------------------------

def _show_trend(
    name: str,
    count: int,
    output_dir: str,
    engine_filter: str,
    use_md: bool,
) -> None:
    """Call bench_trend.py with the generated files."""
    bench_trend = str(Path(__file__).resolve().parent / "bench_trend.py")

    files = []
    for i in range(count):
        f = os.path.join(output_dir, f"{name}_{i}.json")
        if os.path.isfile(f):
            files.append(f)

    if len(files) < 2:
        print("  Not enough results for a trend table.")
        return

    cmd = [sys.executable, bench_trend, *files, "--engine", engine_filter]
    if use_md:
        cmd.append("--md")

    subprocess.run(cmd)


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
    parser.add_argument("--md", action="store_true",
                        help="Markdown output for trend table")
    parser.add_argument("--no-trend", action="store_true",
                        help="Skip final trend display")

    args = parser.parse_args()

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
    commits = get_commits(last, args.from_hash)
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
    print(f'\n=== bench_history: "{args.name}" across {len(commits)} commits ===')
    if model_paths:
        for mp in model_paths:
            print(f"  Model: {mp}")
    print(f"  Output: {args.output_dir}/{args.name}_*.json")
    print(f"  Commits: {commits[0].short_hash}..{commits[-1].short_hash}")

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
        )
        if ok:
            successes += 1
        else:
            failures.append(commit)

    # 4. Show trend
    if not args.no_trend:
        engine_filter = "dotLLM" if dotllm and not llamacpp else ("llama.cpp" if llamacpp and not dotllm else "all")
        print()
        _show_trend(args.name, len(commits), args.output_dir, engine_filter, args.md)

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
