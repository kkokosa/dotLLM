# Benchmarks — dotLLM

## Approach

All benchmarks use **greedy decoding** (argmax / top-k=1, temp=0) to ensure deterministic, comparable output across engines. The reference model is **SmolLM-135M Q8_0** — small enough to run on any machine, large enough to exercise the full pipeline.

Each engine is measured on the same machine, same model file, same prompt, same token count. We compare:

| Metric | What it measures |
|--------|-----------------|
| Load time | GGUF open + config + tokenizer + weight loading |
| Prompt eval | First forward pass processing all prompt tokens |
| Eval | Subsequent decode steps (1 token each, with KV-cache) |
| Total tokens/s | End-to-end throughput (prompt + generated) / wall time |

## Test Setup

- **Model**: `QuantFactory/SmolLM-135M-GGUF` / `SmolLM-135M.Q8_0.gguf` (136 MiB)
- **Prompt**: `"The capital of France is"`
- **Max tokens**: 2
- **Expected output**: `Paris.`
- **Hardware**: AMD Ryzen 9 7950X, 64 GB DDR5, Windows 11, CPU-only

## Commands

### llama.cpp

```
llama-completion.exe ^
  -m C:\Users\kkoko\.dotllm\models\QuantFactory\SmolLM-135M-GGUF\SmolLM-135M.Q8_0.gguf ^
  -p "The capital of France is" ^
  -n 2 ^
  --samplers "top_k" --top-k 1 --repeat-penalty 1.0 --temp 0 ^
  --verbose-prompt
```

### dotLLM

```
DotLLM.Cli.exe run QuantFactory/SmolLM-135M-GGUF -p "The capital of France is" -n 2
```

## Results

### Baseline — Before KV-cache (2025-03-04)

#### llama.cpp (b5291)

```
common_perf_print:        load time =     227.69 ms
common_perf_print: prompt eval time =      11.28 ms /     5 tokens (    2.25 ms per token,   443.46 tokens per second)
common_perf_print:        eval time =       8.27 ms /     1 runs   (    8.27 ms per token,   120.93 tokens per second)
common_perf_print:       total time =      21.27 ms /     6 tokens
```

#### dotLLM — without KV-cache (commit 2f5616d)

```
                    Performance Summary
╭─────────────┬────────────┬────────┬──────────┬──────────╮
│ Phase       │       Time │ Tokens │ ms/token │ tokens/s │
├─────────────┼────────────┼────────┼──────────┼──────────┤
│ Load        │  203.24 ms │      — │        — │        — │
│ Prompt eval │ 1177.39 ms │      5 │   235.48 │     4.25 │
│ Eval        │ 1090.74 ms │      1 │  1090.74 │     0.92 │
│ Sampling    │    0.26 ms │      2 │     0.13 │        — │
│ Total       │ 2269.11 ms │      7 │        — │     3.08 │
╰─────────────┴────────────┴────────┴──────────┴──────────╯

               Memory Breakdown
╭───────────────┬────────────────────────────╮
│ Component     │                       Size │
├───────────────┼────────────────────────────┤
│ Model weights │ 136.4 MiB  (memory-mapped) │
│ Compute       │                    0.9 MiB │
│ Total         │                  137.3 MiB │
╰───────────────┴────────────────────────────╯
```

### Current — With KV-cache (2026-03-04)

#### dotLLM — with KV-cache

```
                    Performance Summary
╭─────────────┬────────────┬────────┬──────────┬──────────╮
│ Phase       │       Time │ Tokens │ ms/token │ tokens/s │
├─────────────┼────────────┼────────┼──────────┼──────────┤
│ Load        │  198.03 ms │      — │        — │        — │
│ Prompt eval │ 1191.75 ms │      5 │   238.35 │     4.20 │
│ Eval        │  227.12 ms │      1 │   227.12 │     4.40 │
│ Sampling    │    0.26 ms │      2 │     0.13 │        — │
│ Total       │ 1419.95 ms │      7 │        — │     4.93 │
╰─────────────┴────────────┴────────┴──────────┴──────────╯

               Memory Breakdown
╭───────────────┬────────────────────────────╮
│ Component     │                       Size │
├───────────────┼────────────────────────────┤
│ Model weights │ 136.4 MiB  (memory-mapped) │
│ Compute       │                    0.9 MiB │
│ KV-cache      │         0.3 MiB  (7 slots) │
│ Total         │                  137.6 MiB │
╰───────────────┴────────────────────────────╯
```

### Analysis

#### KV-cache impact (dotLLM before → after)

| Metric | Before (no cache) | After (KV-cache) | Improvement |
|--------|-------------------|-------------------|-------------|
| Eval per token | 1091 ms | 227 ms | **4.8× faster** |
| Total time | 2269 ms | 1420 ms | **1.6× faster** |
| Total tokens/s | 3.08 | 4.93 | **1.6× faster** |

The eval speedup is modest here because this benchmark only generates 1 decode token. The KV-cache avoids reprocessing the 5 prompt tokens during decode — a single GEMV per weight matrix instead of 6. For longer generations (N decode tokens), the speedup grows linearly: each step is O(1) instead of O(N), so total decode time drops from O(N²) to O(N).

#### dotLLM vs llama.cpp (current)

| Metric | llama.cpp | dotLLM | Ratio |
|--------|-----------|--------|-------|
| Load time | 228 ms | 198 ms | 0.87× (dotLLM faster) |
| Prompt eval (5 tokens) | 11.3 ms | 1192 ms | ~106× slower |
| Eval per token | 8.3 ms | 227 ms | ~27× slower |
| Total tokens/s | ~282 | 4.93 | ~57× slower |

**Load time** is comparable — both memory-map the GGUF file. dotLLM is slightly faster here.

**Prompt eval** remains the main bottleneck. Key factors:

1. **No GEMM batching** — dotLLM uses per-token GEMV (matrix-vector), while llama.cpp batches prompt tokens into a GEMM (matrix-matrix) call. This explains most of the prompt eval gap.
2. **Q8_0 dequant + dot product** — llama.cpp's Q8_0 kernels are heavily SIMD-optimized with fused dequant-dot. dotLLM's current Q8_0 GEMV is functional but not yet tuned.
3. **Thread parallelism** — llama.cpp parallelizes across cores. dotLLM is currently single-threaded.

**Eval per token** dropped from ~132× to ~27× slower with KV-cache. The remaining gap is kernel performance (SIMD tuning, threading).

### Roadmap to Parity

| Optimization | Expected impact | Roadmap step |
|-------------|----------------|--------------|
| ~~KV-cache~~ | ~~eval speedup~~ | ~~Phase 1, Step 7~~ :white_check_mark: |
| SIMD-tuned Q8_0 kernels | ~2-4× kernel speedup | Phase 2, Step 10 |
| Batched GEMM for prefill | ~5-10× prefill speedup | Phase 2, Step 11 |
| Multi-threaded inference | ~4-8× on multi-core | Phase 2, Step 20 |
| CUDA GPU backend | 10-50× prefill, 3-10× decode | Phase 3, Step 21 |
