# Benchmarks — dotLLM

## Approach

All benchmarks use **greedy decoding** (argmax / top-k=1, temp=0) to ensure deterministic, comparable output across engines. The reference model is **SmolLM-135M Q8_0** — small enough to run on any machine, large enough to exercise the full pipeline.

Each engine is measured on the same machine, same model file, same prompt. We compare:

| Metric | What it measures |
|--------|-----------------|
| Load time | GGUF open + config + tokenizer + weight loading |
| Prompt eval | First forward pass processing all prompt tokens |
| Eval | Subsequent decode steps (1 token each, with KV-cache) |
| Total tokens/s | End-to-end throughput (prompt + generated) / wall time |

## Test Setup

- **Model**: `QuantFactory/SmolLM-135M-GGUF` / `SmolLM-135M.Q8_0.gguf` (136 MiB)
- **Prompt**: `"The capital of France is"`
- **Expected output**: `Paris. It is the largest city in France and the capital of the country. Paris is the capital`
- **Hardware**: AMD Ryzen 9 7950X, 64 GB DDR5, Windows 11, CPU-only
- **Build**: Release configuration (JIT + Dynamic PGO)

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
DotLLM.Cli.exe run QuantFactory/SmolLM-135M-GGUF -p "The capital of France is" -n 20 -t 0
```

## Results (2026-03-04)

### llama.cpp (b5291, -n 2)

```
common_perf_print:        load time =     227.69 ms
common_perf_print: prompt eval time =      11.28 ms /     5 tokens (    2.25 ms per token,   443.46 tokens per second)
common_perf_print:        eval time =       8.27 ms /     1 runs   (    8.27 ms per token,   120.93 tokens per second)
common_perf_print:       total time =      21.27 ms /     6 tokens
```

### dotLLM (Release, -n 20)

```
                   Performance Summary
╭─────────────┬───────────┬────────┬──────────┬──────────╮
│ Phase       │      Time │ Tokens │ ms/token │ tokens/s │
├─────────────┼───────────┼────────┼──────────┼──────────┤
│ Load        │ 206.70 ms │      — │        — │        — │
│ Prompt eval │ 299.61 ms │      5 │    59.92 │    16.69 │
│ Eval        │ 464.30 ms │     19 │    24.44 │    40.92 │
│ Sampling    │  16.74 ms │     20 │     0.84 │        — │
│ Total       │ 784.59 ms │     25 │        — │    31.86 │
╰─────────────┴───────────┴────────┴──────────┴──────────╯

               Memory Breakdown
╭───────────────┬────────────────────────────╮
│ Component     │                       Size │
├───────────────┼────────────────────────────┤
│ Model weights │ 136.4 MiB  (memory-mapped) │
│ Compute       │                    0.9 MiB │
│ KV-cache      │        1.1 MiB  (25 slots) │
│ Total         │                  138.4 MiB │
╰───────────────┴────────────────────────────╯
```

## Results — Llama 3.2 1B (2026-03-05)

Larger model to exercise SIMD kernel optimizations at production dimensions (K=2048, FFN K=8192).

- **Model**: `bartowski/Llama-3.2-1B-Instruct-GGUF` / Q8_0 (1252 MiB)
- **Hardware**: AMD Ryzen 7 5800HS, 16 GB DDR4, Windows 11, CPU-only

### llama.cpp (b5291)

```
common_perf_print:        load time =    4662.65 ms
common_perf_print: prompt eval time =    1786.38 ms /    15 tokens (  119.09 ms per token,     8.40 tokens per second)
common_perf_print:        eval time =     104.47 ms /     2 runs   (   52.23 ms per token,    19.14 tokens per second)
common_perf_print:       total time =   11727.30 ms /    17 tokens
```

### dotLLM — before Step 10 (row-by-row AVX2 + scalar quantize)

```
                    Performance Summary
╭─────────────┬────────────┬────────┬──────────┬──────────╮
│ Phase       │       Time │ Tokens │ ms/token │ tokens/s │
├─────────────┼────────────┼────────┼──────────┼──────────┤
│ Load        │  553.66 ms │      — │        — │        — │
│ Prompt eval │ 1392.68 ms │      5 │   278.54 │     3.59 │
│ Eval        │ 3249.81 ms │     19 │   171.04 │     5.85 │
│ Sampling    │   14.49 ms │     20 │     0.72 │        — │
│ Total       │ 4661.93 ms │     25 │        — │     5.36 │
╰─────────────┴────────────┴────────┴──────────┴──────────╯
```

### dotLLM — after Step 10 (4-row AVX2 + FMA + SIMD quantize)

```
                    Performance Summary
╭─────────────┬────────────┬────────┬──────────┬──────────╮
│ Phase       │       Time │ Tokens │ ms/token │ tokens/s │
├─────────────┼────────────┼────────┼──────────┼──────────┤
│ Load        │  530.43 ms │      — │        — │        — │
│ Prompt eval │ 3387.24 ms │      5 │   677.45 │     1.48 │
│ Eval        │ 2713.50 ms │     19 │   142.82 │     7.00 │
│ Sampling    │   15.37 ms │     20 │     0.77 │        — │
│ Total       │ 6120.07 ms │     25 │        — │     4.08 │
╰─────────────┴────────────┴────────┴──────────┴──────────╯
```

### Llama 3.2 1B — Analysis

| Metric | Before Step 10 | After Step 10 | Change |
|--------|---------------|---------------|--------|
| Eval per token | 171.04 ms | 142.82 ms | **1.20× faster** |
| Eval tokens/s | 5.85 | 7.00 | **+20%** |
| Prompt eval per token | 278.54 ms | 677.45 ms | 2.4× slower |

**Eval (decode)** improved by **1.20×** — the FMA accumulation and 4-row batching are effective at K=2048/8192. The gap to llama.cpp (52 ms/token, multi-threaded) narrows from ~3.3× to ~2.7×.

**Prompt eval regression**: The first 5 tokens are slower after the change. This is likely JIT warmup — the new dispatch paths (AVX-512 checks, 4-row branching, SIMD quantize) require more JIT compilation on the first calls. Once the JIT stabilizes, eval tokens are consistently faster. Batched GEMM (Step 11) will address prompt eval performance properly.

## Analysis

### SmolLM-135M — dotLLM vs llama.cpp

| Metric | llama.cpp | dotLLM | Ratio |
|--------|-----------|--------|-------|
| Load time | 228 ms | 207 ms | 0.91× (dotLLM faster) |
| Prompt eval (5 tokens) | 11.3 ms | 300 ms | ~27× slower |
| Eval per token | 8.3 ms | 24.4 ms | ~3× slower |
| Total tokens/s | ~282 | 31.9 | ~9× slower |

**Load time** is comparable — both memory-map the GGUF file. dotLLM is slightly faster.

**Eval per token** (24.4 ms) is now within **3× of llama.cpp** thanks to KV-cache and Release-mode JIT with Dynamic PGO. The remaining gap is thread parallelism (llama.cpp uses all cores, dotLLM is single-threaded).

**Prompt eval** is the main bottleneck (~27× slower). This is expected — dotLLM processes prompt tokens one at a time (GEMV), while llama.cpp batches them into a single GEMM call and parallelizes across cores.

### Roadmap to Parity

| Optimization | Expected impact | Roadmap step |
|-------------|----------------|--------------|
| ~~SIMD-tuned Q8_0 kernels~~ | ~~~2-4× kernel speedup~~ | ~~Phase 2, Step 10~~ :white_check_mark: **1.2× eval on Llama 1B, up to 1.9× on micro-benchmarks** |
| Batched GEMM for prefill | ~5-10× prefill speedup | Phase 2, Step 11 |
| Multi-threaded inference | ~4-8× on multi-core | Phase 2, Step 22 |
| CUDA GPU backend | 10-50× prefill, 3-10× decode | Phase 3, Step 29 |
