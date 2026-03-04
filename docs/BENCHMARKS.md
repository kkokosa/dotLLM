# Benchmarks — dotLLM

## Approach

All benchmarks use **greedy decoding** (argmax / top-k=1, temp=0) to ensure deterministic, comparable output across engines. The reference model is **SmolLM-135M Q8_0** — small enough to run on any machine, large enough to exercise the full pipeline.

Each engine is measured on the same machine, same model file, same prompt, same token count. We compare:

| Metric | What it measures |
|--------|-----------------|
| Load time | GGUF open + config + tokenizer + weight loading |
| Prompt eval | First forward pass processing all prompt tokens |
| Eval | Subsequent decode steps (1 token each, no KV-cache in dotLLM yet) |
| Total tokens/s | End-to-end throughput (prompt + generated) / wall time |

> **Note**: dotLLM currently reprocesses the full context each step (no KV-cache). This makes eval dramatically slower than llama.cpp's cached single-token decode. The comparison is useful for validating correctness and measuring kernel-level performance, not end-to-end throughput parity — that comes with KV-cache (Phase 1 Step 8).

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

## Results (2025-03-04)

### llama.cpp (b5291)

```
common_perf_print:        load time =     227.69 ms
common_perf_print: prompt eval time =      11.28 ms /     5 tokens (    2.25 ms per token,   443.46 tokens per second)
common_perf_print:        eval time =       8.27 ms /     1 runs   (    8.27 ms per token,   120.93 tokens per second)
common_perf_print:       total time =      21.27 ms /     6 tokens
```

### dotLLM (commit 2f5616d)

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

### Analysis

| Metric | llama.cpp | dotLLM | Ratio |
|--------|-----------|--------|-------|
| Load time | 228 ms | 203 ms | 0.89× (dotLLM faster) |
| Prompt eval (5 tokens) | 11.3 ms | 1177 ms | ~104× slower |
| Eval per token | 8.3 ms | 1091 ms | ~132× slower |
| Total tokens/s | ~282 | 3.08 | ~92× slower |

**Load time** is comparable — both memory-map the GGUF file. dotLLM is slightly faster here.

**Prompt eval and eval** show the expected gap. Key factors:

1. **No KV-cache** — dotLLM reprocesses the full context each step. llama.cpp caches K/V and only computes the new token. This alone accounts for a large portion of the eval gap.
2. **No GEMM batching** — dotLLM uses per-token GEMV (matrix-vector), while llama.cpp batches prompt tokens into a GEMM (matrix-matrix) call. This explains most of the prompt eval gap.
3. **Q8_0 dequant + dot product** — llama.cpp's Q8_0 kernels are heavily SIMD-optimized with fused dequant-dot. dotLLM's current Q8_0 GEMV is functional but not yet tuned.
4. **Thread parallelism** — llama.cpp parallelizes across cores. dotLLM is currently single-threaded.

### Roadmap to Parity

| Optimization | Expected impact | Roadmap step |
|-------------|----------------|--------------|
| KV-cache | ~N× speedup on eval (N = seq length) | Phase 1, Step 7 |
| SIMD-tuned Q8_0 kernels | ~2-4× kernel speedup | Phase 2, Step 10 |
| Batched GEMM for prefill | ~5-10× prefill speedup | Phase 2, Step 11 |
| Multi-threaded inference | ~4-8× on multi-core | Phase 2, Step 20 |
| CUDA GPU backend | 10-50× prefill, 3-10× decode | Phase 3, Step 21 |
