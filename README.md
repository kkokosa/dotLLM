<div align="center">

# dotLLM

**High-performance LLM inference engine written natively in C#/.NET**

[![CI](https://github.com/kkokosa/dotLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/kkokosa/dotLLM/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![.NET](https://img.shields.io/badge/.NET-10-purple.svg)](https://dotnet.microsoft.com/)

[Documentation](docs/) · [Roadmap](docs/ROADMAP.md) · [Discussions](https://github.com/kkokosa/dotLLM/discussions)

</div>

---

## About

dotLLM is a ground-up LLM inference engine for .NET — not a wrapper around llama.cpp or Python libraries. All orchestration, model loading, tokenization, sampling, and CPU compute are implemented in pure C#, with a thin C/CUDA native library for GPU kernels. It targets transformer-based models (Llama, Mistral, Phi, Qwen, DeepSeek) with SIMD-optimized CPU and CUDA GPU backends.

> **Status**: Phase 2 complete — dotLLM supports Q4_K_M quantization, chat templates, streaming generation, multi-threaded CPU inference, and multiple architectures (Llama, Mistral, Phi, Qwen). See [Roadmap](#roadmap) for Phase 3 (CPU performance optimization).

## Key Features

### Performance
- **Zero-GC inference** — unmanaged memory (`NativeMemory.AlignedAlloc`, 64-byte aligned) for all tensor data; no managed heap allocations on the hot path
- **SIMD vectorization** — `TensorPrimitives` + hand-tuned `System.Runtime.Intrinsics` for quantized matmul, RMSNorm, RoPE, softmax
- **Memory-mapped model loading** — GGUF files loaded via `MemoryMappedFile`; OS demand-paging means multi-GB models load in milliseconds
- **Quantized inference** — FP16, Q8_0, Q4_K_M and other GGUF quantization formats; fused scale×int dot-product kernels operating directly on quantized blocks

### Architecture Support
- **Transformer models** — Llama, Mistral, Phi, Qwen, DeepSeek via parameterized `TransformerBlock` and `ModelConfig`
- **Attention mechanisms** — MHA, MQA, GQA, MLA through `IAttentionMechanism` + `IAttentionStrategy` separation
- **Position encoding** — RoPE, ALiBi, absolute, none — pluggable via `IPositionEncoding`
- **Composable sampling** — `ISamplerStep` chain: repetition penalty → temperature → top-k → top-p → min-p → categorical sample

### Serving
- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/completions`, tool calling, streaming via ASP.NET
- **Continuous batching** — iteration-level scheduling with preemption and priority queuing
- **Paged KV-cache** — PagedAttention with block-level allocation, prefix caching, and copy-on-write
- **Speculative decoding** — draft-verify-accept with KV-cache rollback for higher throughput
- **Structured output** — FSM/PDA-based constrained decoding guaranteeing valid JSON, JSON Schema, regex, and grammar

### Extensibility
- **Pluggable backends** — `IBackend` interface with separate packages per backend (CPU, CUDA, ROCm)
- **LoRA adapters** — runtime loading, no weight merging, concurrent multi-adapter serving
- **Diagnostic hooks** — zero-cost `IInferenceHook` points for activation capture, logit lens, SAE integration
- **OpenTelemetry observability** — `System.Diagnostics.Metrics` + `Activity` for throughput, latency, and per-request tracing

## Architecture Overview

dotLLM is organized as a layered architecture where each layer depends only on the layers below it:

```
┌─────────────────────────────────────────┐
│            DotLLM.Server                │  ASP.NET OpenAI-compatible API
├─────────────────────────────────────────┤
│            DotLLM.Engine                │  KV-cache, scheduler, samplers,
│                                         │  constraints, speculative decoding
├──────────┬──────────┬───────────────────┤
│ DotLLM.  │ DotLLM.  │ DotLLM.Cpu/Cuda   │  GGUF/SafeTensors, BPE/SPM,
│ Models   │Tokenizers│ (backends)        │  SIMD kernels / CUDA kernels
├──────────┴──────────┴───────────────────┤
│            DotLLM.Core                  │  Interfaces, tensor types, config
└─────────────────────────────────────────┘
```

Each project ships as a separate NuGet package, so users pull in only what they need. `DotLLM.Core` defines all abstractions (`ITensor`, `IBackend`, `IModel`, `ISamplerStep`, etc.) while concrete implementations live in their respective projects.

## Getting Started

### Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- [Python 3.10+](https://www.python.org/) with `pip install rich InquirerPy` (for benchmark scripts)
- *Optional:* [llama.cpp](https://github.com/ggerganov/llama.cpp) for comparison benchmarks (see [llama.cpp setup](#llamacpp-setup) below)

### Build

```bash
git clone https://github.com/kkokosa/dotLLM.git
cd dotLLM
dotnet build
```

### Tests

**Unit and integration tests:**

```bash
dotnet test
```

> Integration tests automatically download [SmolLM-135M](https://huggingface.co/QuantFactory/SmolLM-135M-GGUF) Q8_0 (~145 MB) to `~/.dotllm/test-cache/`.

**Model correctness smoke tests** (`scripts/test_models.py`) run dotLLM CLI with greedy decoding across architectures (Llama, Mistral, Phi, Qwen) and verify expected output:

```bash
# Build CLI first
dotnet build src/DotLLM.Cli -c Release

# List available test cases and which models are cached
python scripts/test_models.py --list

# Run tests for all cached models
python scripts/test_models.py

# Download missing models and run all tests
python scripts/test_models.py --download

# Run only specific architectures
python scripts/test_models.py --filter phi,qwen
```

Models are downloaded from HuggingFace to `~/.dotllm/models/` on first use and cached for subsequent runs.

Sample output:

```
Test                                Arch       Result      Time  Details
=====================================================================================================
SmolLM-135M                         Llama      PASS        2.1s  Paris  (163.3 tok/s)
Llama-3.2-1B-Instruct-Q4            Llama      PASS        5.7s  Paris  (31.0 tok/s)
Qwen2.5-0.5B-Instruct               Qwen       PASS        3.2s  Paris  (78.5 tok/s)
Phi-3-mini-4k-instruct              Phi        PASS       12.4s  Paris  (14.2 tok/s)
=====================================================================================================

4/4 passed, 0 failed, 0 skipped
```

### Benchmarks

Three scripts in `scripts/` provide benchmarking at different levels:

**`bench_compare.py`** -- Single-point benchmark. Runs dotLLM (via [BenchmarkDotNet](https://benchmarkdotnet.org/)) and optionally [llama.cpp](https://github.com/ggerganov/llama.cpp) on one or more models, reports best-of-N throughput with CV (coefficient of variation):

```bash
# Benchmark dotLLM on SmolLM-135M (auto-downloads from HuggingFace)
python scripts/bench_compare.py --model QuantFactory/SmolLM-135M-GGUF --quant Q8_0

# Benchmark multiple models and quantizations
python scripts/bench_compare.py \
    --model QuantFactory/SmolLM-135M-GGUF,bartowski/Llama-3.2-1B-Instruct-GGUF \
    --quant Q4_K_M,Q8_0

# Compare dotLLM vs llama.cpp side-by-side
python scripts/bench_compare.py --model QuantFactory/SmolLM-135M-GGUF --dotllm --llamacpp

# Export results to JSON for later comparison
python scripts/bench_compare.py --model QuantFactory/SmolLM-135M-GGUF \
    --export-json benchmarks/results/baseline.json --label baseline
```

Sample output:

```
=== dotLLM Benchmark Results ===

  Model                  Prefill tok/s   Decode tok/s   Decode ms/tok   Total tok/s     CV
  SmolLM-135M.Q8_0             229.2          182.7           5.47         175.3      14.7%
  SmolLM-135M.Q4_K_M           165.0          230.1           4.35         198.2      20.5%

All values are best-of-N (max tok/s, min ms). CV is the coefficient of variation
across N iterations -- lower means more stable measurements.
```

**`bench_trend.py`** -- Interactive comparison of exported JSON results. Displays color-coded delta tables with noise-aware highlighting:

```bash
# Interactive mode: select runs and models to compare
python scripts/bench_trend.py

# Compare two specific result files
python scripts/bench_trend.py benchmarks/results/baseline.json benchmarks/results/optimized.json

# Show all results as a trend table
python scripts/bench_trend.py --all
```

Sample output (comparing two runs):

```
Comparison: baseline (a062743) -> optimized (572179d)

  Metric               baseline (a062743)     optimized (572179d)        Delta
  Prefill tok/s                      44.8                    48.8       +8.9%
  Decode tok/s                       24.2                    31.0      +28.1%
  Decode ms/tok                     41.30                   32.30      +21.8%

  Model: Llama-3.2-1B-Instruct-Q4_K_M | Prompt: short | Tokens: 20 | CV: 10.6%
```

**`bench_history.py`** -- Benchmark across git commits. Creates worktrees for each commit, runs bench_compare in each, and displays trend tables with per-commit deltas:

```bash
# Benchmark last 5 commits on main
python scripts/bench_history.py myrun --last 5

# Benchmark from a specific commit to HEAD
python scripts/bench_history.py myrun --from f3d3bf8

# Show results from a previous run (no benchmarking)
python scripts/bench_history.py myrun --show

# Interactively select which commits to benchmark
python scripts/bench_history.py myrun --last 10 --select
```

Sample output:

```
                     Benchmark History -- Llama-3.2-3B-Instruct-Q8_0
 Label                  Date        Prefill tok/s   %chg pf   Decode tok/s   %chg dc     CV
 uber_run5_0 (f3d3bf8)  2026-03-16           24.6                      8.1                 -
 uber_run5_1 (c12ba0a)  2026-03-16           24.3     -1.3%            8.0    ~-0.9%       -
 uber_run5_2 (cdb5234)  2026-03-16           24.9     +2.8%            8.0    ~-0.1%       -
 uber_run5_3 (5531fa4)  2026-03-16           24.7    ~-0.9%            7.8     -1.9%       -
 uber_run5_4 (a062743)  2026-03-16           24.9    ~+0.9%            7.8    ~-0.7%       -
 uber_run5_5 (f50cefe)  2026-03-16           24.7    ~-0.7%            7.9     +1.6%       -
 uber_run5_6 (6c06fbf)  2026-03-16           24.6    ~-0.6%            7.8     -1.9%       -
 uber_run5_7 (d1978d2)  2026-03-16           25.4     +3.4%            7.0    -10.4%       -
 uber_run5_8 (572179d)  2026-03-16           25.5    ~+0.1%            7.8    +12.2%    4.2%
```

> `%chg` columns show commit-to-commit deltas. `~` prefix means the change is within noise (CV threshold). CV requires multiple [BenchmarkDotNet](https://benchmarkdotnet.org/) iterations (controlled by `--runs` in bench_compare).

**Why best-of-N instead of median?** On a non-isolated machine (laptop, desktop with background processes), run-to-run noise is typically 6--30%. The median includes runs degraded by OS scheduling jitter, thermal throttling, and background I/O. Best-of-N (maximum throughput) represents what the hardware *can* achieve and is more stable across sessions. CV is reported alongside so you can judge measurement quality -- if CV is high, the environment was noisy and even the best-of-N value should be taken with a grain of salt.

### [llama.cpp](https://github.com/ggerganov/llama.cpp) setup

To run comparison benchmarks against [llama.cpp](https://github.com/ggerganov/llama.cpp):

1. **Get llama.cpp** -- either download a [prebuilt release](https://github.com/ggerganov/llama.cpp/releases) or build from source:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release
   ```

2. **Point bench_compare to the binary** -- either:
   - Set `LLAMACPP_BIN` environment variable to the path of `llama-cli`
   - Or pass `--llamacpp-bin /path/to/llama-cli` on each invocation

3. **Run comparison:**
   ```bash
   python scripts/bench_compare.py --model QuantFactory/SmolLM-135M-GGUF --dotllm --llamacpp
   ```

> [llama.cpp](https://github.com/ggerganov/llama.cpp) is optional. All dotLLM benchmarks work without it. The `--llamacpp` flag simply adds a side-by-side comparison column.

There is no NuGet package yet -- the project is in early development. Follow the [Roadmap](#roadmap) for progress toward the first release.

## News

- **2026-03** — NUMA-aware threading: adaptive spin-wait dispatch (generation counter with event fallback), NUMA topology detection (Windows/Linux), P-core/E-core awareness, CPU affinity pinning, auto-reduced decode thread count ([#57](https://github.com/kkokosa/dotLLM/issues/57))
- **2026-03** — Operator fusion: fused RMSNorm+quantize (decode-only, eliminates normOut intermediate buffer) and tiled SwiGLU (1KB L1-resident sigmoid buffer) reduce DRAM roundtrips on the decode hot path ([#56](https://github.com/kkokosa/dotLLM/issues/56))
- **2026-03** — Fast approximate exp/softmax: Schraudolph IEEE-754 bit-manipulation trick replaces polynomial exp (~3 SIMD ops vs ~12) in attention softmax. AVX2/AVX-512 fused shift+exp+sum pass eliminates 3 separate TensorPrimitives calls. Sampling softmax keeps full precision ([#55](https://github.com/kkokosa/dotLLM/issues/55))
- **2026-03** — Tiled attention with online softmax: O(N) memory flash-attention-style algorithm replaces O(N²) score matrix materialization, eliminates 64 MB/head allocations at ctx 4096, uses ~1 KB stack per head ([#54](https://github.com/kkokosa/dotLLM/issues/54))
- **2026-03** — Row-interleaved weight repacking: R4 layout stores 4 consecutive rows' blocks contiguously at model load time, improving cache/TLB locality for all quantized GEMV kernels ([#52](https://github.com/kkokosa/dotLLM/issues/52))
- **2026-03** — Q8_1 input quantization: precomputed block sums for Q5_0 kernels, 2-block loop unrolling, eliminates ~4 SIMD ops/block from Q5_0 vec_dot hot path ([#51](https://github.com/kkokosa/dotLLM/issues/51))
- **2026-03** — Fused decode dispatch: Q/K/V (3→1) and Gate/Up (2→1) projection fusion saves ~72 dispatches/layer, ~4% decode throughput improvement ([#50](https://github.com/kkokosa/dotLLM/issues/50))
- **2026-03** — **Phase 2 complete**: additional model architectures (Mistral, Phi, Qwen), sliding window attention, fused QKV support, `IModel` interface, `ModelLoader` helper ([#34](https://github.com/kkokosa/dotLLM/issues/34))
- **2026-03** — Streaming token generation: `IAsyncEnumerable<GenerationToken>` API with UTF-8-safe incremental text, `CancellationToken` support, and per-token finish reason/timings ([#31](https://github.com/kkokosa/dotLLM/issues/31))
- **2026-03** — Chat template engine: Jinja2-subset interpreter (lexer→parser→evaluator), `IChatTemplate` implementation, `GgufChatTemplateFactory`, `dotllm chat` REPL command ([#30](https://github.com/kkokosa/dotLLM/issues/30))
- **2026-03** — Mixed quantization + Q8_K: Q8_K input quantization (float32 scale, 256-element blocks, precomputed bsums), true 4-row fused K-quant kernels, re-enabled Q4_K×Q8_K/Q5_K×Q8_K/Q6_K×Q8_K fused GEMV/GEMM ([#29](https://github.com/kkokosa/dotLLM/issues/29))
- **2026-03** — Q4_K_M dequantization and vec_dot kernels: Q4_K, Q5_K, Q6_K scalar + AVX2 dequant and fused matmul kernels with full model-level dispatch ([#28](https://github.com/kkokosa/dotLLM/issues/28))
- **2026-03** — BDN inference benchmarks: end-to-end benchmarks with custom tok/s columns, auto model download, llama.cpp comparison script ([#42](https://github.com/kkokosa/dotLLM/issues/42))
- **2026-03** — Engine inference timings: `InferenceTimings` on `InferenceResponse`, `onTokenGenerated` callback, CLI refactored to use `TextGenerator` ([#41](https://github.com/kkokosa/dotLLM/issues/41))
- **2026-03** — Multi-threaded CPU inference: zero-alloc `ComputeThreadPool` with `delegate*` dispatch, parallel GEMV/GEMM and head-parallel attention ([#36](https://github.com/kkokosa/dotLLM/issues/36))
- **2026-03** — SIMD kernel tuning: FMA float accumulation, 4-row batched GEMV, AVX-512 paths, SIMD quantization ([#26](https://github.com/kkokosa/dotLLM/issues/26))
- **2026-03** — Phase 1 complete: sampling pipeline + stop conditions — first coherent multi-token generation ([#24](https://github.com/kkokosa/dotLLM/pull/24))
- **2026-03** — KV-cache: eval drops from 1091 ms/token to 227 ms/token (~4.8× speedup)
- **2026-03** — Llama forward pass: first token generation from embedding to logits
- **2026-02** — BPE Tokenizer with SentencePiece and tiktoken support ([#16](https://github.com/kkokosa/dotLLM/pull/16))

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **1 — End-to-End Generation** | GGUF loading, dequantization, CPU ops, tokenizer, attention, forward pass, KV-cache, sampling | Done (9/9) |
| **2 — Practical Local Inference** | Engine metrics, benchmarks, Q4_K_M, chat templates, streaming, multi-threading, more architectures | Done (10/10) |
| **3 — CPU Performance** | Decode dispatch, Q8_1 input, weight repacking, outer-product GEMM, tiled attention, fast exp, fusion, NUMA | In Progress (7/8) |
| **4 — GPU Acceleration** | CUDA backend, CPU/GPU hybrid, KV-cache quantization | Planned |
| **5 — Constrained Decoding & API** | JSON mode, JSON Schema, regex/CFG, tool calling, logit bias, OpenAI API server | Planned |
| **6 — Production Serving** | Continuous batching, paged KV-cache, prompt caching, speculative decoding, metrics | Planned |
| **7 — Expand** | Hooks, logit lens, LoRA, MLA, SAE, multi-GPU, ROCm | Planned |

See [docs/ROADMAP.md](docs/ROADMAP.md) for detailed steps, dependencies, and milestones.

## Documentation

- [Architecture & data flow](docs/ARCHITECTURE.md)
- [GGUF binary format](docs/GGUF_FORMAT.md)
- [Quantization formats](docs/QUANTIZATION.md)
- [Attention mechanisms](docs/ATTENTION.md)
- [Position encoding](docs/POSITION_ENCODING.md)
- [Tokenizers & chat templates](docs/TOKENIZERS.md)
- [Sampling pipeline](docs/SAMPLING.md)
- [Constrained decoding](docs/CONSTRAINED_DECODING.md)
- [KV-cache management](docs/KV_CACHE.md)
- [Batch scheduling](docs/SCHEDULING.md)
- [Full roadmap](docs/ROADMAP.md)

## Contributing

Contributions are welcome! dotLLM uses an issue-driven workflow — every change starts with a [GitHub issue](https://github.com/kkokosa/dotLLM/issues) describing the work. Pick an existing issue or open a new one, then submit a PR targeting `main`.

## Contact

Questions, ideas, or feedback? Open a thread in [GitHub Discussions](https://github.com/kkokosa/dotLLM/discussions).

## License

dotLLM is licensed under the [GNU General Public License v3.0](LICENSE).

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — reference for GGUF format, quantization kernels, and CUDA implementations
- [Hugging Face](https://huggingface.co/) — model ecosystem, transformers reference implementations, tokenizer specs
- [.NET team](https://github.com/dotnet/runtime) — `TensorPrimitives`, `System.Runtime.Intrinsics`, `MemoryMappedFile`, and the runtime that makes this possible
