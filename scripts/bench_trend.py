#!/usr/bin/env python3
"""
bench_trend.py — Interactive benchmark comparison tool.

Scans a results directory for JSON files exported by bench_compare.py --export-json,
lets you interactively select which runs and models to compare, and displays a
formatted comparison table.

Usage:
    # Interactive mode (default — scans benchmarks/results/, dotLLM engine)
    python scripts/bench_trend.py

    # Show llama.cpp results
    python scripts/bench_trend.py --engine llama.cpp --all

    # Show all engines
    python scripts/bench_trend.py --engine all --all

    # Interactive with custom folder
    python scripts/bench_trend.py --folder path/to/results

    # Non-interactive: compare two specific files
    python scripts/bench_trend.py baseline.json step23.json

    # Non-interactive: show all as trend table
    python scripts/bench_trend.py --all

    # GitHub Markdown output
    python scripts/bench_trend.py --md

Dependencies (install once):
    pip install rich InquirerPy
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional dependency imports with graceful fallback
# ---------------------------------------------------------------------------

_has_rich = False
_has_inquirer = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    _has_rich = True
except ImportError:
    pass

try:
    from InquirerPy import inquirer
    from InquirerPy.separator import Separator
    _has_inquirer = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class BenchEntry:
    """Parsed benchmark result file."""
    path: Path
    label: str
    timestamp: str
    commit: str
    branch: str
    dirty: bool
    models: list[str]
    engines: list[str]
    results: list[dict]
    config: dict
    system: dict
    raw: dict

    @staticmethod
    def load(path: Path) -> BenchEntry | None:
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        git = data.get("git", {})
        results = data.get("results", [])
        models = list(dict.fromkeys(r.get("model", "?") for r in results))
        engines = list(dict.fromkeys(r.get("engine", "dotLLM") for r in results))

        return BenchEntry(
            path=path,
            label=data.get("label", path.stem),
            timestamp=data.get("timestamp", ""),
            commit=git.get("commit", "?")[:7],
            branch=git.get("branch", "?"),
            dirty=git.get("dirty", False),
            models=models,
            engines=engines,
            results=results,
            config=data.get("config", {}),
            system=data.get("system", {}),
            raw=data,
        )

    def get_result(self, model_filter: str | None = None) -> dict | None:
        for r in self.results:
            if model_filter and model_filter.lower() not in r.get("model", "").lower():
                continue
            return r
        return self.results[0] if self.results and not model_filter else None

    @property
    def date(self) -> str:
        return self.timestamp[:10] if self.timestamp else "?"

    @property
    def display_name(self) -> str:
        dirty = "*" if self.dirty else ""
        return f"{self.label} ({self.commit}{dirty})"

    def filtered(self, engine: str | None = None) -> BenchEntry:
        """Return a copy with results filtered to a specific engine (substring match)."""
        if not engine:
            return self
        eng_lower = engine.lower()
        filt = [r for r in self.results if eng_lower in r.get("engine", "").lower()]
        return replace(
            self,
            models=list(dict.fromkeys(r.get("model", "?") for r in filt)),
            engines=list(dict.fromkeys(r.get("engine", "?") for r in filt)),
            results=filt,
        )


def scan_directory(directory: Path) -> list[BenchEntry]:
    """Load all benchmark JSON files from a directory, sorted by timestamp."""
    if not directory.is_dir():
        return []
    entries = []
    for jf in sorted(directory.glob("*.json")):
        entry = BenchEntry.load(jf)
        if entry:
            entries.append(entry)
    entries.sort(key=lambda e: e.timestamp)
    return entries


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_delta(old_val: float, new_val: float, higher_is_better: bool) -> tuple[str, str]:
    """Format delta as (text, style). style is 'green'/'red'/'dim'."""
    if old_val == 0:
        return "N/A", "dim"
    pct = (new_val - old_val) / old_val * 100
    if not higher_is_better:
        pct = -pct
    sign = "+" if pct >= 0 else ""
    text = f"{sign}{pct:.1f}%"
    if abs(pct) < 0.5:
        return text, "dim"
    return text, "green" if pct > 0 else "red"


METRICS = [
    ("Prefill tok/s", "prefill_tok_per_sec", True, ".1f"),
    ("Decode tok/s", "decode_tok_per_sec", True, ".1f"),
    ("Decode ms/tok", "decode_ms_per_tok", False, ".2f"),
    ("Prefill ms", "prefill_ms", False, ".1f"),
    ("Decode ms", "decode_ms", False, ".1f"),
    ("Total tok/s", "total_tok_per_sec", True, ".1f"),
]


# ---------------------------------------------------------------------------
# Rich output
# ---------------------------------------------------------------------------

def rich_trend_table(entries: list[BenchEntry], model_filter: str | None) -> None:
    """Display a trend table using rich."""
    console = Console()

    table = Table(title="Benchmark Trend", show_lines=False, pad_edge=True)
    table.add_column("Label", style="bold cyan", no_wrap=True)
    table.add_column("Commit", style="dim")
    table.add_column("Date", style="dim")
    table.add_column("Model", style="dim")
    table.add_column("Prefill tok/s", justify="right")
    table.add_column("Decode tok/s", justify="right")
    table.add_column("Decode ms/tok", justify="right")

    prev_result: dict | None = None
    for entry in entries:
        result = entry.get_result(model_filter)
        if not result:
            continue

        pf = result.get("prefill_tok_per_sec", 0)
        dc = result.get("decode_tok_per_sec", 0)
        ms = result.get("decode_ms_per_tok", 0)
        model = result.get("model", "?")

        # Color-code values relative to previous entry
        if prev_result:
            _, pf_style = format_delta(prev_result.get("prefill_tok_per_sec", 0), pf, True)
            _, dc_style = format_delta(prev_result.get("decode_tok_per_sec", 0), dc, True)
            _, ms_style = format_delta(prev_result.get("decode_ms_per_tok", 0), ms, False)
        else:
            pf_style = dc_style = ms_style = ""

        dirty = "*" if entry.dirty else ""
        table.add_row(
            entry.label,
            f"{entry.commit}{dirty}",
            entry.date,
            model,
            Text(f"{pf:.1f}", style=pf_style),
            Text(f"{dc:.1f}", style=dc_style),
            Text(f"{ms:.2f}", style=ms_style),
        )
        prev_result = result

    console.print()
    console.print(table)
    console.print()


def rich_comparison_table(base: BenchEntry, current: BenchEntry, model_filter: str | None) -> None:
    """Display a pairwise comparison table using rich."""
    console = Console()

    base_r = base.get_result(model_filter)
    curr_r = current.get_result(model_filter)

    if not base_r or not curr_r:
        console.print("[red]No matching results found in one or both files.[/red]")
        return

    table = Table(
        title=f"Comparison: {base.display_name} -> {current.display_name}",
        show_lines=False,
        pad_edge=True,
    )
    table.add_column("Metric", style="bold", no_wrap=True)
    table.add_column(base.display_name, justify="right", style="dim")
    table.add_column(current.display_name, justify="right", style="bold")
    table.add_column("Delta", justify="right")

    for label, key, higher_better, fmt in METRICS:
        bv = base_r.get(key, 0)
        cv = curr_r.get(key, 0)
        delta_text, delta_style = format_delta(bv, cv, higher_better)

        table.add_row(
            label,
            f"{bv:{fmt}}",
            f"{cv:{fmt}}",
            Text(delta_text, style=f"bold {delta_style}"),
        )

    console.print()
    console.print(table)

    model = curr_r.get("model", "?")
    config = current.config
    console.print(
        f"  Model: [cyan]{model}[/cyan] | "
        f"Prompt: {config.get('prompt_size', '?')} | "
        f"Tokens: {config.get('max_tokens', '?')}"
    )
    console.print()


# ---------------------------------------------------------------------------
# Plain-text fallback output
# ---------------------------------------------------------------------------

def plain_trend_table(entries: list[BenchEntry], model_filter: str | None) -> None:
    """Plain-text trend table when rich is not installed."""
    print(f"\n{'Label':<20} {'Commit':<9} {'Date':<12} {'Model':<30} {'Prefill tok/s':>14} {'Decode tok/s':>13} {'Decode ms/tok':>14}")
    print("-" * 115)
    for entry in entries:
        result = entry.get_result(model_filter)
        if not result:
            continue
        dirty = "*" if entry.dirty else ""
        print(
            f"{entry.label:<20} {entry.commit + dirty:<9} {entry.date:<12} "
            f"{result.get('model', '?'):<30} "
            f"{result.get('prefill_tok_per_sec', 0):>14.1f} "
            f"{result.get('decode_tok_per_sec', 0):>13.1f} "
            f"{result.get('decode_ms_per_tok', 0):>14.2f}"
        )
    print()


def plain_comparison_table(base: BenchEntry, current: BenchEntry, model_filter: str | None) -> None:
    """Plain-text comparison when rich is not installed."""
    base_r = base.get_result(model_filter)
    curr_r = current.get_result(model_filter)
    if not base_r or not curr_r:
        print("No matching results found.", file=sys.stderr)
        return

    title = f"Comparison: {base.display_name} -> {current.display_name}"
    print(f"\n{title}\n")
    header = f"{'Metric':<20} {base.display_name:>22} {current.display_name:>22} {'Delta':>10}"
    print(header)
    print("-" * len(header))
    for label, key, higher_better, fmt in METRICS:
        bv = base_r.get(key, 0)
        cv = curr_r.get(key, 0)
        delta_text, _ = format_delta(bv, cv, higher_better)
        print(f"{label:<20} {bv:>22{fmt}} {cv:>22{fmt}} {delta_text:>10}")
    print()


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def md_trend_table(entries: list[BenchEntry], model_filter: str | None) -> None:
    print("## Benchmark Trend\n")
    print("| Label | Commit | Date | Model | Prefill tok/s | Decode tok/s | Decode ms/tok |")
    print("|-------|--------|------|-------|---------------|--------------|---------------|")
    for entry in entries:
        result = entry.get_result(model_filter)
        if not result:
            continue
        dirty = "*" if entry.dirty else ""
        print(
            f"| {entry.label} | {entry.commit}{dirty} | {entry.date} | "
            f"{result.get('model', '?')} | "
            f"{result.get('prefill_tok_per_sec', 0):.1f} | "
            f"{result.get('decode_tok_per_sec', 0):.1f} | "
            f"{result.get('decode_ms_per_tok', 0):.2f} |"
        )


def md_comparison_table(base: BenchEntry, current: BenchEntry, model_filter: str | None) -> None:
    base_r = base.get_result(model_filter)
    curr_r = current.get_result(model_filter)
    if not base_r or not curr_r:
        return

    print(f"## Comparison: {base.display_name} -> {current.display_name}\n")
    print(f"| Metric | {base.display_name} | {current.display_name} | Delta |")
    print("|--------|--------|--------|-------|")
    for label, key, higher_better, fmt in METRICS:
        bv = base_r.get(key, 0)
        cv = curr_r.get(key, 0)
        delta_text, _ = format_delta(bv, cv, higher_better)
        print(f"| {label} | {bv:{fmt}} | {cv:{fmt}} | {delta_text} |")
    print()
    model = curr_r.get("model", "?")
    config = current.config
    print(f"Model: {model} | Prompt: {config.get('prompt_size', '?')} | Tokens: {config.get('max_tokens', '?')}")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_select(entries: list[BenchEntry]) -> tuple[list[BenchEntry], str | None]:
    """Interactively select entries and model filter. Returns (selected_entries, model_filter)."""
    if not _has_inquirer:
        print("Interactive mode requires InquirerPy: pip install InquirerPy", file=sys.stderr)
        sys.exit(1)

    # Build choices for entry selection
    choices = []
    for entry in entries:
        models_str = ", ".join(entry.models)
        dirty = " [dirty]" if entry.dirty else ""
        display = (
            f"{entry.label:<20} {entry.commit}{dirty:<10} "
            f"{entry.date}   {models_str}"
        )
        choices.append({"name": display, "value": entry, "enabled": True})

    selected_entries: list[BenchEntry] = inquirer.checkbox(
        message="Select benchmark runs to compare (Space to toggle, Enter to confirm):",
        choices=choices,
        validate=lambda result: len(result) >= 1 or "Select at least one entry",
        instruction="(↑↓ move, Space toggle, Ctrl+A toggle all, Enter confirm)",
        keybindings={
            "toggle-all": [{"key": "c-a"}],
            "toggle-all-true": [],
        },
    ).execute()

    if not selected_entries:
        print("No entries selected.", file=sys.stderr)
        sys.exit(0)

    # Collect all unique models across selected entries
    all_models: list[str] = []
    seen: set[str] = set()
    for entry in selected_entries:
        for model in entry.models:
            if model not in seen:
                all_models.append(model)
                seen.add(model)

    model_filter: str | None = None
    if len(all_models) > 1:
        model_choices = [{"name": f"All models", "value": None}]
        model_choices += [{"name": m, "value": m} for m in all_models]

        model_filter = inquirer.select(
            message="Filter by model:",
            choices=model_choices,
        ).execute()

    return selected_entries, model_filter


# ---------------------------------------------------------------------------
# Display dispatcher
# ---------------------------------------------------------------------------

def _collect_models(entries: list[BenchEntry]) -> list[str]:
    """Collect unique model names across all entries, preserving order."""
    seen: set[str] = set()
    models: list[str] = []
    for entry in entries:
        for r in entry.results:
            m = r.get("model", "?")
            if m not in seen:
                seen.add(m)
                models.append(m)
    return models


def _collect_engines(entries: list[BenchEntry]) -> list[str]:
    """Collect unique engine names across all entries, preserving order."""
    seen: set[str] = set()
    engines: list[str] = []
    for entry in entries:
        for e in entry.engines:
            if e not in seen:
                seen.add(e)
                engines.append(e)
    return engines


def _apply_engine_filter(entries: list[BenchEntry], engine: str | None) -> list[BenchEntry]:
    """Pre-filter entries by engine. Returns entries that still have results."""
    if not engine:
        return entries
    filtered = [e.filtered(engine) for e in entries]
    return [e for e in filtered if e.results]


def display(entries: list[BenchEntry], model_filter: str | None, use_md: bool) -> None:
    """Display results using the best available renderer.

    When model_filter is None and entries contain multiple models,
    renders one table per model.
    """
    # Determine which models to show
    if model_filter:
        model_list = [model_filter]
    else:
        model_list = _collect_models(entries)

    for model in model_list:
        _display_one(entries, model, use_md)


def _display_one(entries: list[BenchEntry], model_filter: str, use_md: bool) -> None:
    """Display a single table for one model filter."""
    if use_md:
        if len(entries) == 2:
            md_comparison_table(entries[0], entries[1], model_filter)
        else:
            md_trend_table(entries, model_filter)
        return

    if _has_rich:
        if len(entries) == 2:
            rich_comparison_table(entries[0], entries[1], model_filter)
        else:
            rich_trend_table(entries, model_filter)
    else:
        if len(entries) == 2:
            plain_comparison_table(entries[0], entries[1], model_filter)
        else:
            plain_trend_table(entries, model_filter)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_FOLDER = "benchmarks/results"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive benchmark comparison tool for dotLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/bench_trend.py                              # interactive (dotLLM)\n"
            "  python scripts/bench_trend.py --all                        # show all results\n"
            "  python scripts/bench_trend.py --engine llama.cpp --all     # llama.cpp results\n"
            "  python scripts/bench_trend.py --engine all --all           # all engines\n"
            "  python scripts/bench_trend.py base.json cur.json           # compare two files\n"
            "  python scripts/bench_trend.py --md --all                   # markdown output\n"
        ),
    )
    parser.add_argument("files", nargs="*",
                        help="JSON files to compare (non-interactive mode)")
    parser.add_argument("--folder", type=str, default=DEFAULT_FOLDER,
                        help=f"Results directory (default: {DEFAULT_FOLDER})")
    parser.add_argument("--all", action="store_true",
                        help="Show all results in folder as trend table (non-interactive)")
    parser.add_argument("--md", action="store_true",
                        help="Output as GitHub Markdown")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter results by model name substring")
    parser.add_argument("--engine", type=str, default="dotLLM",
                        help='Filter by engine (default: dotLLM). Use "all" for all engines.')

    args = parser.parse_args()

    # Resolve engine filter: "all" means no filter
    engine_filter = args.engine if args.engine.lower() != "all" else None

    # Mode 1: Explicit files on command line
    if args.files:
        entries = []
        for f in args.files:
            entry = BenchEntry.load(Path(f))
            if not entry:
                print(f"Failed to load: {f}", file=sys.stderr)
                return 1
            entries.append(entry)
        entries = _apply_engine_filter(entries, engine_filter)
        display(entries, args.model, args.md)
        return 0

    # Load entries from folder
    folder = Path(args.folder)
    entries = scan_directory(folder)

    if not entries:
        print(f"No benchmark results found in {folder}/", file=sys.stderr)
        print(f"Run: python scripts/bench_compare.py --model ... --dotllm --export-json {folder}/run.json --label run",
              file=sys.stderr)
        return 1

    # Mode 2: --all flag — show everything non-interactively
    if args.all:
        entries = _apply_engine_filter(entries, engine_filter)
        display(entries, args.model, args.md)
        return 0

    # Mode 3: Interactive selection
    if not _has_inquirer:
        print("Tip: pip install InquirerPy  (for interactive run/model selection)\n",
              file=sys.stderr)
        if not _has_rich:
            print("Tip: pip install rich  (for colored tables)\n",
                  file=sys.stderr)
        entries = _apply_engine_filter(entries, engine_filter)
        display(entries, args.model, args.md)
        return 0

    # Interactive: prompt for engine if multiple exist and --engine wasn't explicitly given
    all_engines = _collect_engines(entries)
    if len(all_engines) > 1 and args.engine == "dotLLM":
        engine_choices = [{"name": e, "value": e} for e in all_engines]
        engine_choices.append({"name": "All engines", "value": None})
        engine_filter = inquirer.select(
            message="Select engine:",
            choices=engine_choices,
            default="dotLLM",
        ).execute()

    entries = _apply_engine_filter(entries, engine_filter)

    selected, model_filter = interactive_select(entries)
    if args.model:
        model_filter = args.model  # CLI override
    display(selected, model_filter, args.md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
