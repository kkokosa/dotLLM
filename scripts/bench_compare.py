#!/usr/bin/env python3
"""
bench_compare.py — Run dotLLM BDN benchmarks and optionally compare against llama.cpp.

Extensible: add new engines by implementing a run_<engine>() function and registering it
in the ENGINES dict.

Usage:
    python scripts/bench_compare.py --model QuantFactory/SmolLM-135M-GGUF
    python scripts/bench_compare.py --model bartowski/Llama-3.2-3B-Instruct-GGUF --quant Q8_0
    python scripts/bench_compare.py --model path/to/model.gguf --llamacpp
    python scripts/bench_compare.py --model path/to/model.gguf --llamacpp --dotllm
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import median


# ---------------------------------------------------------------------------
# Predefined prompts for fair benchmarking at different lengths
# ---------------------------------------------------------------------------
PROMPTS = {
    "short": "The capital of France is",  # ~5 tokens
    "medium": (
        "The transformer architecture, introduced in the 2017 paper 'Attention Is All You Need' "
        "by Vaswani et al., revolutionized natural language processing by replacing recurrent "
        "neural networks with a self-attention mechanism. Unlike RNNs, which process tokens "
        "sequentially, transformers can attend to all positions in a sequence simultaneously, "
        "enabling massive parallelism during training. The architecture consists of an encoder "
        "and decoder, each built from stacked layers of multi-head self-attention and "
        "position-wise feed-forward networks. Layer normalization and residual connections "
        "stabilize training. Positional encodings, originally sinusoidal, inject sequence order "
        "information since the model has no inherent notion of position. Modern variants like "
        "GPT use only the decoder stack with causal masking, while BERT uses only the encoder "
        "with bidirectional attention. Rotary Position Embeddings (RoPE) have largely replaced "
        "sinusoidal encodings, enabling better length generalization. Grouped-Query Attention "
        "(GQA) reduces the KV-cache memory footprint by sharing key-value heads across multiple "
        "query heads, which is critical for efficient inference at long context lengths."
    ),  # ~256 tokens
    "large": (
        "Large language models (LLMs) have become a cornerstone of modern artificial intelligence, "
        "demonstrating remarkable capabilities in text generation, reasoning, and code synthesis. "
        "The journey began with early neural language models that used simple recurrent architectures, "
        "but the field was transformed by the introduction of the transformer architecture in 2017. "
        "The key innovation was the self-attention mechanism, which allows every token in a sequence "
        "to attend to every other token, capturing long-range dependencies without the vanishing "
        "gradient problems that plagued RNNs and LSTMs.\n\n"
        "Training these models requires enormous computational resources. A typical LLM training run "
        "involves processing trillions of tokens from web crawls, books, code repositories, and "
        "curated datasets. The training objective is usually next-token prediction (causal language "
        "modeling), where the model learns to predict each token given all preceding tokens. This "
        "simple objective, scaled up with more data and parameters, produces emergent capabilities "
        "like in-context learning, chain-of-thought reasoning, and instruction following.\n\n"
        "Inference efficiency is a critical concern for deploying LLMs. The autoregressive nature of "
        "text generation means that each new token requires a forward pass through the entire model. "
        "The KV-cache optimization stores previously computed key and value tensors so they don't need "
        "to be recomputed at each step, trading memory for computation. Quantization reduces the "
        "precision of model weights from 16-bit floating point to 8-bit, 4-bit, or even lower, "
        "dramatically reducing memory requirements and improving throughput with minimal quality loss.\n\n"
        "Speculative decoding is another technique that accelerates inference by using a smaller draft "
        "model to propose multiple tokens at once, which the larger target model then verifies in a "
        "single forward pass. This can provide 2-3x speedups without any change in output quality. "
        "Continuous batching allows the inference engine to dynamically add and remove requests from "
        "a batch as they complete, maximizing GPU utilization compared to static batching.\n\n"
        "The GGUF file format has become the standard for distributing quantized models. It is a "
        "self-contained binary format that includes model weights, tokenizer data, and metadata in "
        "a single file. GGUF supports memory-mapped loading, which means the operating system can "
        "demand-page model weights directly from disk without copying them into managed memory. This "
        "enables a 7-billion parameter model to begin inference within milliseconds of opening the "
        "file, as only the pages that are actually accessed are loaded into RAM.\n\n"
        "Modern inference engines like llama.cpp, vLLM, and TensorRT-LLM have pushed the boundaries "
        "of what is possible on consumer hardware. A quantized 70-billion parameter model can now "
        "run on a single GPU with 24GB of VRAM, generating text at interactive speeds. CPU inference "
        "has also improved dramatically, with SIMD-optimized kernels using AVX2 and AVX-512 "
        "instructions to perform quantized matrix multiplications at near-theoretical throughput. "
        "The combination of better quantization methods, optimized memory access patterns, and "
        "hardware-aware kernel design continues to democratize access to powerful language models."
    ),  # ~1024 tokens
}


@dataclass
class EngineResult:
    """Standardized result from any inference engine."""
    engine: str
    model: str
    prefill_ms: float
    decode_ms: float
    prefill_tokens: int
    decode_tokens: int
    prefill_tok_per_sec: float
    decode_tok_per_sec: float
    decode_ms_per_tok: float
    total_tok_per_sec: float
    # BDN-only fields (None for non-BDN engines)
    bdn_mean_ns: float | None = None
    bdn_stddev_ns: float | None = None


def _default_models_dir() -> Path:
    """Return the default models directory, matching HuggingFaceDownloader.DefaultModelsDirectory."""
    env = os.environ.get("DOTLLM_MODELS_DIR")
    if env:
        return Path(env)
    return Path.home() / ".dotllm" / "models"


def _hf_list_gguf_files(repo_id: str) -> list[dict]:
    """List .gguf files in a HuggingFace repo via the API. Returns list of {path, size} dicts."""
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dotLLM-bench/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            entries = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"[model] HF API error for {repo_id}: {e.code} {e.reason}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"[model] Network error querying HF API: {e.reason}", file=sys.stderr)
        sys.exit(1)

    return [
        {"path": entry["path"], "size": entry.get("size", 0)}
        for entry in entries
        if isinstance(entry, dict) and entry.get("path", "").endswith(".gguf")
    ]


def _download_with_progress(url: str, dest: Path, total_size: int | None = None) -> None:
    """Download a file with progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")

    # Resume support
    downloaded = 0
    if tmp_path.exists():
        downloaded = tmp_path.stat().st_size

    headers = {"User-Agent": "dotLLM-bench/1.0"}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            if total_size is None:
                content_length = resp.headers.get("Content-Length")
                if content_length:
                    total_size = int(content_length) + downloaded

            with open(tmp_path, "ab" if downloaded > 0 else "wb") as f:
                chunk_size = 1024 * 1024  # 1 MB
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = downloaded * 100 / total_size
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r[model] Downloading: {mb_done:.0f}/{mb_total:.0f} MB ({pct:.0f}%)", end="", flush=True)
                    else:
                        mb_done = downloaded / (1024 * 1024)
                        print(f"\r[model] Downloading: {mb_done:.0f} MB", end="", flush=True)
    except (urllib.error.URLError, OSError) as e:
        print(f"\n[model] Download failed: {e}", file=sys.stderr)
        sys.exit(1)

    print()  # newline after progress
    tmp_path.replace(dest)


def _apply_quant_filter(files: list[str], quant: str | None) -> list[str]:
    """Filter filenames by quant substring if specified."""
    if not quant:
        return files
    return [f for f in files if quant in f]


def resolve_model(model_arg: str, quant: str | None) -> Path:
    """
    Resolve a model argument to a local .gguf path.

    Accepts:
    - Local path ending in .gguf → used as-is
    - HF repo ID (e.g. "QuantFactory/SmolLM-135M-GGUF") → check cache, then download
    """
    # 1. Local .gguf path
    if model_arg.endswith(".gguf"):
        p = Path(model_arg)
        if p.exists():
            return p
        print(f"[model] File not found: {model_arg}", file=sys.stderr)
        sys.exit(1)

    # 2. HF repo ID
    repo_id = model_arg
    models_dir = _default_models_dir()
    repo_dir = models_dir / repo_id.replace("/", os.sep)

    # 2a. Check local cache
    if repo_dir.exists():
        cached = [f.name for f in repo_dir.iterdir() if f.suffix == ".gguf"]
        cached = _apply_quant_filter(cached, quant)
        if len(cached) == 1:
            result = repo_dir / cached[0]
            print(f"[model] Using cached: {result}")
            return result
        if len(cached) > 1:
            print(f"[model] Multiple cached .gguf files in {repo_dir}:")
            for i, name in enumerate(sorted(cached), 1):
                size_mb = (repo_dir / name).stat().st_size / (1024 * 1024)
                print(f"  {i}. {name} ({size_mb:.0f} MB)")
            print(f"[model] Use --quant to narrow (e.g. --quant Q8_0)", file=sys.stderr)
            sys.exit(1)

    # 2b. Query HF API
    print(f"[model] Querying HF API for {repo_id}...")
    hf_files = _hf_list_gguf_files(repo_id)
    if not hf_files:
        print(f"[model] No .gguf files found in {repo_id}", file=sys.stderr)
        sys.exit(1)

    filenames = [f["path"] for f in hf_files]
    size_map = {f["path"]: f["size"] for f in hf_files}
    filenames = _apply_quant_filter(filenames, quant)

    if not filenames:
        all_names = [f["path"] for f in hf_files]
        print(f"[model] No .gguf files matching --quant '{quant}'. Available:", file=sys.stderr)
        for name in sorted(all_names):
            size_mb = size_map.get(name, 0) / (1024 * 1024)
            print(f"  - {name} ({size_mb:.0f} MB)", file=sys.stderr)
        sys.exit(1)

    if len(filenames) > 1:
        print(f"[model] Multiple .gguf files in {repo_id}:")
        for i, name in enumerate(sorted(filenames), 1):
            size_mb = size_map.get(name, 0) / (1024 * 1024)
            print(f"  {i}. {name} ({size_mb:.0f} MB)")
        print(f"[model] Use --quant to narrow (e.g. --quant Q8_0)", file=sys.stderr)
        sys.exit(1)

    # 2c. Download the single match
    filename = filenames[0]
    dest = repo_dir / filename
    if dest.exists():
        print(f"[model] Using cached: {dest}")
        return dest

    size = size_map.get(filename, 0)
    size_mb = size / (1024 * 1024) if size else 0
    print(f"[model] Downloading {repo_id}/{filename} (~{size_mb:.0f} MB)...")
    download_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    _download_with_progress(download_url, dest, total_size=size if size else None)
    print(f"[model] Saved to: {dest}")
    return dest


def find_benchmark_project() -> Path:
    """Auto-detect the benchmark .csproj relative to this script."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    csproj = repo_root / "benchmarks" / "DotLLM.Benchmarks" / "DotLLM.Benchmarks.csproj"
    if csproj.exists():
        return csproj
    raise FileNotFoundError(f"Cannot find benchmark project. Tried: {csproj}")


def run_dotllm(
    model_path: str | None,
    prompt: str,
    max_tokens: int,
    runs: int,
    bdn_filter: str,
    bdn_project: str | None,
    skip_bdn_build: bool,
    **kwargs,
) -> list[EngineResult]:
    """Run dotLLM BDN benchmarks and parse results."""
    project = Path(bdn_project) if bdn_project else find_benchmark_project()
    project_dir = project.parent

    # When env var overrides the model, narrow to a single enum value to avoid
    # running 3 identical benchmarks (all [ParamsAllValues] load the same file).
    effective_filter = bdn_filter
    if model_path and bdn_filter == "*InferenceBenchmarks*":
        effective_filter = "*SmolLM_135M*"

    cmd = [
        "dotnet", "run",
        "--project", str(project),
        "-c", "Release",
        "--",
        "--filter", effective_filter,
        "--exporters", "json",
    ]

    # Set env vars so BDN uses the centrally-resolved model, prompt, and token count
    env = os.environ.copy()
    if model_path:
        env["DOTLLM_BENCH_MODEL_PATH"] = model_path
    env["DOTLLM_BENCH_PROMPT"] = prompt
    env["DOTLLM_BENCH_MAX_TOKENS"] = str(max_tokens)

    print(f"[dotLLM] Running BDN: {' '.join(cmd)}")
    if model_path:
        print(f"[dotLLM] DOTLLM_BENCH_MODEL_PATH={model_path}")
    prompt_preview = prompt[:60] + "..." if len(prompt) > 60 else prompt
    print(f'[dotLLM] DOTLLM_BENCH_PROMPT="{prompt_preview}"')
    print(f"[dotLLM] DOTLLM_BENCH_MAX_TOKENS={max_tokens}")
    result = subprocess.run(cmd, cwd=str(project_dir), capture_output=False, env=env)
    if result.returncode != 0:
        print(f"[dotLLM] BDN exited with code {result.returncode}", file=sys.stderr)
        return []

    # Find BDN JSON report (BDN names it *-report-full-compressed.json by default)
    artifacts_dir = project_dir / "BenchmarkDotNet.Artifacts" / "results"
    json_files = list(artifacts_dir.glob("*InferenceBenchmarks*-report*.json"))
    if not json_files:
        json_files = list(artifacts_dir.glob("*-report*.json"))
    if not json_files:
        print(f"[dotLLM] No BDN JSON report found in {artifacts_dir}", file=sys.stderr)
    else:
        print(f"[dotLLM] Found BDN report: {json_files[0].name}")

    bdn_results: list[EngineResult] = []

    # Determine metrics key: filename stem when using env var override
    metrics_key = None
    if model_path:
        metrics_key = Path(model_path).stem

    for json_file in json_files:
        try:
            with open(json_file) as f:
                report = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[dotLLM] Failed to parse {json_file}: {e}", file=sys.stderr)
            continue

        for bench in report.get("Benchmarks", []):
            stats = bench.get("Statistics", {})
            mean_ns = stats.get("Mean", 0)
            stddev_ns = stats.get("StandardDeviation", 0)

            # Determine the model name for display and metrics file lookup
            model_name = metrics_key
            if model_name is None:
                # Fallback: extract from BDN parameters
                model_name = "unknown"
                for param in bench.get("Parameters", "").split(", "):
                    if param.startswith("Model="):
                        model_name = param.split("=", 1)[1]
                        break

            # Read custom metrics from file bridge
            metrics_dir = Path(tempfile.gettempdir()) / "dotllm-bdn-metrics"
            metrics_file = metrics_dir / f"{model_name}.json"

            prefill_ms = 0.0
            decode_ms = 0.0
            prefill_tokens = 0
            decode_tokens = 0
            prefill_tok_s = 0.0
            decode_tok_s = 0.0

            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    prefill_ms = metrics.get("medianPrefillMs", 0)
                    decode_ms = metrics.get("medianDecodeMs", 0)
                    prefill_tokens = metrics.get("prefillTokenCount", 0)
                    decode_tokens = metrics.get("decodeTokenCount", 0)
                    prefill_tok_s = metrics.get("medianPrefillTokPerSec", 0)
                    decode_tok_s = metrics.get("medianDecodeTokPerSec", 0)
                except (json.JSONDecodeError, OSError):
                    pass

            total_ms = prefill_ms + decode_ms
            total_tokens = prefill_tokens + decode_tokens
            total_tok_s = total_tokens / (total_ms / 1000.0) if total_ms > 0 else 0
            ms_per_tok = decode_ms / decode_tokens if decode_tokens > 0 else 0

            bdn_results.append(EngineResult(
                engine="dotLLM",
                model=model_name,
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
                prefill_tok_per_sec=prefill_tok_s,
                decode_tok_per_sec=decode_tok_s,
                decode_ms_per_tok=ms_per_tok,
                total_tok_per_sec=total_tok_s,
                bdn_mean_ns=mean_ns,
                bdn_stddev_ns=stddev_ns,
            ))

    return bdn_results


def _run_llamacpp_once(cmd: list[str], run_num: int) -> str | None:
    """Run llama-completion once with -no-cnv (exits after generation) and return stderr."""
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print(f"[llama.cpp] Run {run_num} timed out", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"[llama.cpp] Binary not found: {cmd[0]}", file=sys.stderr)
        return None

    # Print perf lines for transparency
    for line in result.stderr.splitlines():
        if "common_perf_print" in line:
            print(f"  {line}")

    return result.stderr


def run_llamacpp(
    model_path: str | None,
    prompt: str,
    max_tokens: int,
    runs: int,
    llamacpp_bin: str | None,
    **kwargs,
) -> list[EngineResult]:
    """Run llama.cpp llama-completion N times and parse perf output."""
    if not llamacpp_bin:
        print("[llama.cpp] No llama-completion binary found. Use --llamacpp-bin or set LLAMACPP_BIN.", file=sys.stderr)
        return []
    if not model_path:
        print("[llama.cpp] No --model path provided, skipping.", file=sys.stderr)
        return []

    # Regex patterns for llama.cpp common_perf_print output
    re_prefill = re.compile(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second"
    )
    re_decode = re.compile(
        r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([\d.]+)\s*tokens per second"
    )

    prefill_ms_list: list[float] = []
    decode_ms_list: list[float] = []
    prefill_tokens_list: list[int] = []
    decode_tokens_list: list[int] = []
    prefill_tok_s_list: list[float] = []
    decode_tok_s_list: list[float] = []

    model_name = Path(model_path).stem

    cmd = [
        llamacpp_bin,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", "0",
        "--no-display-prompt",
        "-no-cnv",          # disable conversation mode — exit after generation
        "--simple-io",      # safe for subprocesses
        "--perf",           # enable timing output (off by default)
        "--mlock",          # lock model in RAM — eliminates mmap page faults during timing
    ]

    # Print command for manual reproduction (truncate long prompts)
    display_cmd = list(cmd)
    prompt_idx = display_cmd.index("-p") + 1
    if len(display_cmd[prompt_idx]) > 80:
        display_cmd[prompt_idx] = display_cmd[prompt_idx][:80] + "..."
    quoted_cmd = " ".join(f'"{c}"' if " " in c else c for c in display_cmd)
    print(f"[llama.cpp] Command: {quoted_cmd}")

    # Warmup run — fault in mmap pages and warm caches before measured runs
    print("[llama.cpp] Warmup run (discarded)...")
    warmup_out = _run_llamacpp_once(cmd, 0)
    if warmup_out is None:
        return []  # binary not found or warmup failed

    for i in range(runs):
        print(f"[llama.cpp] Run {i + 1}/{runs}...")

        output = _run_llamacpp_once(cmd, i + 1)
        if output is None:
            if i == 0:
                return []  # binary not found or first run failed hard
            continue

        m_prefill = re_prefill.search(output)
        m_decode = re_decode.search(output)

        if m_prefill and m_decode:
            prefill_ms_list.append(float(m_prefill.group(1)))
            prefill_tokens_list.append(int(m_prefill.group(2)))
            prefill_tok_s_list.append(float(m_prefill.group(3)))
            decode_ms_list.append(float(m_decode.group(1)))
            decode_tokens_list.append(int(m_decode.group(2)))
            decode_tok_s_list.append(float(m_decode.group(3)))
        else:
            print(f"[llama.cpp] Run {i + 1}: could not parse perf output", file=sys.stderr)
            print(f"  output (last 500 chars): {output[-500:]}", file=sys.stderr)

    if not prefill_ms_list:
        print("[llama.cpp] No successful runs.", file=sys.stderr)
        return []

    pf_ms = median(prefill_ms_list)
    dc_ms = median(decode_ms_list)
    pf_tok = prefill_tokens_list[0]
    dc_tok = decode_tokens_list[0]
    pf_tok_s = median(prefill_tok_s_list)
    dc_tok_s = median(decode_tok_s_list)
    total_ms = pf_ms + dc_ms
    total_tokens = pf_tok + dc_tok
    total_tok_s = total_tokens / (total_ms / 1000.0) if total_ms > 0 else 0
    ms_per_tok = dc_ms / dc_tok if dc_tok > 0 else 0

    return [EngineResult(
        engine="llama.cpp",
        model=model_name,
        prefill_ms=pf_ms,
        decode_ms=dc_ms,
        prefill_tokens=pf_tok,
        decode_tokens=dc_tok,
        prefill_tok_per_sec=pf_tok_s,
        decode_tok_per_sec=dc_tok_s,
        decode_ms_per_tok=ms_per_tok,
        total_tok_per_sec=total_tok_s,
    )]


# Engine registry — add new engines here
ENGINES: dict[str, callable] = {
    "dotllm": run_dotllm,
    "llamacpp": run_llamacpp,
}


def print_comparison(results: list[EngineResult], prompt: str, max_tokens: int) -> None:
    """Print a formatted comparison table."""
    if not results:
        print("No results to display.")
        return

    model_name = results[0].model
    print()
    prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
    print(f"=== Engine Comparison: {model_name} ===")
    print(f'Prompt: "{prompt_preview}", Max tokens: {max_tokens}')
    print()

    # Header
    hdr = f"{'Engine':<14} {'Prefill':>10} {'':>10} {'Decode':>10} {'':>10} {'':>8} {'Total':>8}  {'Notes'}"
    sub = f"{'':14} {'ms':>10} {'tok/s':>10} {'ms':>10} {'tok/s':>10} {'ms/tok':>8} {'tok/s':>8}"
    sep = "\u2500" * 100

    print(hdr)
    print(sub)
    print(sep)

    for r in results:
        notes = ""
        if r.bdn_mean_ns is not None:
            mean_ms = r.bdn_mean_ns / 1_000_000
            stddev_ms = (r.bdn_stddev_ns or 0) / 1_000_000
            notes = f"(BDN: {mean_ms:.1f}\u00b1{stddev_ms:.1f} ms)"
        elif r.engine == "llama.cpp":
            notes = "(median of runs)"

        print(
            f"{r.engine:<14} "
            f"{r.prefill_ms:>10.1f} {r.prefill_tok_per_sec:>10.1f} "
            f"{r.decode_ms:>10.1f} {r.decode_tok_per_sec:>10.1f} "
            f"{r.decode_ms_per_tok:>8.2f} {r.total_tok_per_sec:>8.1f}  {notes}"
        )

    print(sep)

    # Ratio rows: dotLLM as baseline, one row per other engine.
    # >1.0x consistently means dotLLM is faster.
    # For time (ms, ms/tok): other/dotllm  — other took more time = dotLLM faster.
    # For throughput (tok/s): dotllm/other — dotLLM has more throughput.
    dotllm_results = [r for r in results if r.engine == "dotLLM"]
    other_results = [r for r in results if r.engine != "dotLLM"]

    if dotllm_results and other_results:
        base = dotllm_results[0]

        def speedup(base_val: float, other_val: float, higher_is_better: bool) -> str:
            """Return speedup string. >1.0x means dotLLM is faster."""
            if base_val == 0 or other_val == 0:
                return "N/A"
            if higher_is_better:
                return f"{base_val / other_val:.2f}x"
            else:
                return f"{other_val / base_val:.2f}x"

        for comp in other_results:
            label = f"vs {comp.engine}"
            print(
                f"{label:<14} "
                f"{speedup(base.prefill_ms, comp.prefill_ms, False):>10} "
                f"{speedup(base.prefill_tok_per_sec, comp.prefill_tok_per_sec, True):>10} "
                f"{speedup(base.decode_ms, comp.decode_ms, False):>10} "
                f"{speedup(base.decode_tok_per_sec, comp.decode_tok_per_sec, True):>10} "
                f"{speedup(base.decode_ms_per_tok, comp.decode_ms_per_tok, False):>8} "
                f"{speedup(base.total_tok_per_sec, comp.total_tok_per_sec, True):>8}  "
                f"(>1 = dotLLM faster)"
            )
        print(sep)

    print()


def _find_llamacpp_bin(explicit: str | None) -> str | None:
    """Resolve llama-completion binary: explicit path > LLAMACPP_BIN env var > PATH lookup."""
    if explicit:
        return explicit
    env = os.environ.get("LLAMACPP_BIN")
    if env:
        return env
    return shutil.which("llama-completion")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run dotLLM benchmarks and compare against other engines."
    )
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace repo ID (e.g. 'QuantFactory/SmolLM-135M-GGUF') or local .gguf path")
    parser.add_argument("--quant", type=str, default=None,
                        help="Quantization filter substring (e.g. 'Q8_0') to narrow GGUF selection")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt text (overrides --prompt-size)")
    parser.add_argument("--prompt-size", type=str, default="short",
                        choices=list(PROMPTS.keys()),
                        help="Predefined prompt size: short (~5 tok), medium (~256 tok), large (~1024 tok)")
    parser.add_argument("--tokens", type=int, default=20,
                        help="Max tokens to generate (default: 20)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of llama.cpp runs (default: 5; BDN has its own iteration config)")

    # Engine selection flags — if none specified, all engines run
    parser.add_argument("--dotllm", action="store_true",
                        help="Run dotLLM BDN benchmarks")
    parser.add_argument("--llamacpp", action="store_true",
                        help="Run llama.cpp benchmarks")

    # llama.cpp binary path (auto-detected from LLAMACPP_BIN env var or PATH if omitted)
    parser.add_argument("--llamacpp-bin", type=str, default=None,
                        help="Path to llama-completion binary (default: LLAMACPP_BIN env var or PATH)")

    parser.add_argument("--bdn-filter", type=str, default="*InferenceBenchmarks*",
                        help="BDN filter pattern (default: *InferenceBenchmarks*)")
    parser.add_argument("--bdn-project", type=str, default=None,
                        help="Path to benchmark .csproj (default: auto-detect)")
    parser.add_argument("--skip-bdn-build", action="store_true",
                        help="Skip BDN build (use existing artifacts)")

    args = parser.parse_args()

    # Resolve prompt: --prompt overrides --prompt-size
    if args.prompt is not None:
        prompt = args.prompt
        prompt_size_label = "custom"
    else:
        prompt_size_label = args.prompt_size
        prompt = PROMPTS[prompt_size_label]

    # Resolve model: download from HF if needed, get a single local path
    resolved_model: str | None = None
    if args.model:
        resolved_path = resolve_model(args.model, args.quant)
        resolved_model = str(resolved_path)
        print(f"[model] Resolved: {resolved_model}")
    else:
        # No --model: list cached models and exit
        models_dir = _default_models_dir()
        cached = list(models_dir.rglob("*.gguf")) if models_dir.exists() else []
        if not cached:
            print("[model] No --model specified and no cached models found.", file=sys.stderr)
            print(f"[model] Download dir: {models_dir}", file=sys.stderr)
            print("[model] Usage: bench_compare.py --model QuantFactory/SmolLM-135M-GGUF", file=sys.stderr)
            return 1
        print("[model] No --model specified. Available cached models:")
        for p in sorted(cached):
            size_mb = p.stat().st_size / (1024 * 1024)
            # Show path relative to models dir for easy copy-paste as --model
            rel = p.relative_to(models_dir)
            print(f"  {rel}  ({size_mb:.0f} MB)")
        print()
        print("[model] Pass one with --model <path-or-repo-id>")
        return 1

    # Print configuration banner
    estimated_tokens = max(1, len(prompt.split()) * 4 // 3)  # rough word→token estimate
    prompt_preview = prompt[:60] + "..." if len(prompt) > 60 else prompt
    print()
    print(f"[config] Threads: {os.cpu_count()} (auto)")
    print(f'[config] Prompt ({prompt_size_label}): "{prompt_preview}" (~{estimated_tokens} tokens est.)')
    print(f"[config] Max tokens: {args.tokens}")
    print()

    # Determine which engines to run: explicit flags, or all if none specified
    explicit_engines = args.dotllm or args.llamacpp
    engine_names: list[str] = []
    if explicit_engines:
        if args.dotllm:
            engine_names.append("dotllm")
        if args.llamacpp:
            engine_names.append("llamacpp")
    else:
        engine_names = list(ENGINES.keys())

    # Resolve llama.cpp binary
    llamacpp_bin = _find_llamacpp_bin(args.llamacpp_bin)

    all_results: list[EngineResult] = []

    for name in engine_names:
        if name not in ENGINES:
            print(f"Unknown engine: {name}. Available: {', '.join(ENGINES.keys())}", file=sys.stderr)
            return 1

        fn = ENGINES[name]
        results = fn(
            model_path=resolved_model,
            prompt=prompt,
            max_tokens=args.tokens,
            runs=args.runs,
            llamacpp_bin=llamacpp_bin,
            bdn_filter=args.bdn_filter,
            bdn_project=args.bdn_project,
            skip_bdn_build=args.skip_bdn_build,
        )
        all_results.extend(results)

    print_comparison(all_results, prompt, args.tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
