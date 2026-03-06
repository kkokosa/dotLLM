# Benchmarks — dotLLM

## Approach

All benchmarks use **greedy decoding** (temp=0) to ensure deterministic, comparable output across engines. Each engine is measured on the same machine, same model file, same prompt, same thread count.

| Metric | What it measures |
|--------|-----------------|
| Prefill | Forward pass processing all prompt tokens (GEMM) |
| Decode | Autoregressive token generation with KV-cache (GEMV) |
| Total tok/s | End-to-end throughput (prompt + generated) / wall time |

### Methodology

Comparisons are run via `scripts/bench_compare.py`, which ensures fairness:

- **Same prompt & token count** — both engines receive identical inputs via env vars
- **Warm pages** — llama.cpp runs with `--mlock` to lock model weights in RAM (eliminates mmap page faults during timing); dotLLM uses BDN warmup iterations for the same effect
- **Warmup run** — llama.cpp gets a discarded warmup invocation before measured runs
- **Statistical rigor** — dotLLM uses BenchmarkDotNet (2 warmup + 5 measured iterations); llama.cpp uses median of 5 runs
- **Thread parity** — both engines default to all available cores (`ThreadingConfig.Auto` / llama.cpp default)

```bash
# Run comparison
python scripts/bench_compare.py --model QuantFactory/SmolLM-135M-GGUF --prompt-size short

# Available prompt sizes: short (~5 tok), medium (~256 tok), large (~1024 tok)
python scripts/bench_compare.py --model bartowski/Llama-3.2-3B-Instruct-GGUF --quant Q8_0 --prompt-size large
```

## Current Results (2026-03-06)

All results on: **AMD Ryzen 7 5800HS**, 16 GB DDR4, 8 cores / 16 threads, Windows 11, CPU-only, .NET 10, llama.cpp b5291.

### SmolLM-135M Q8_0 (136 MiB)

```
Engine            Prefill                Decode                        Total  Notes
                       ms      tok/s         ms      tok/s   ms/tok    tok/s
────────────────────────────────────────────────────────────────────────────────────────────────────
dotLLM               29.4      170.0      175.2      108.5     9.22    117.3  (BDN: 199.3±2.1 ms)
llama.cpp             9.6      523.7      130.4      145.8     6.86    171.5  (median of runs)
────────────────────────────────────────────────────────────────────────────────────────────────────
vs llama.cpp        0.32x      0.32x      0.74x      0.74x    0.74x    0.68x  (>1 = dotLLM faster)
────────────────────────────────────────────────────────────────────────────────────────────────────
```

### Llama 3.2 1B Instruct Q8_0 (1252 MiB)

```
Engine            Prefill                Decode                        Total  Notes
                       ms      tok/s         ms      tok/s   ms/tok    tok/s
────────────────────────────────────────────────────────────────────────────────────────────────────
dotLLM              120.3       41.6     1065.2       17.8    56.06     20.2  (BDN: 1185.5±10.8 ms)
llama.cpp            70.9       84.6     1008.8       18.8    53.09     23.2  (median of runs)
────────────────────────────────────────────────────────────────────────────────────────────────────
vs llama.cpp        0.59x      0.49x      0.95x      0.95x    0.95x    0.87x  (>1 = dotLLM faster)
────────────────────────────────────────────────────────────────────────────────────────────────────
```

### Llama 3.2 3B Instruct Q8_0 (3400 MiB)

```
Engine            Prefill                Decode                        Total  Notes
                       ms      tok/s         ms      tok/s   ms/tok    tok/s
────────────────────────────────────────────────────────────────────────────────────────────────────
dotLLM              303.1       16.5     2800.0        6.8   147.37      7.7  (BDN: 3032.0±86.1 ms)
llama.cpp           205.9       29.1     2832.6        6.7   149.08      8.2  (median of runs)
────────────────────────────────────────────────────────────────────────────────────────────────────
vs llama.cpp        0.68x      0.57x      1.01x      1.01x    1.01x    0.94x  (>1 = dotLLM faster)
────────────────────────────────────────────────────────────────────────────────────────────────────
```

## Analysis

### Decode (memory-bandwidth bound)

| Model | dotLLM tok/s | llama.cpp tok/s | Ratio |
|-------|-------------|----------------|-------|
| SmolLM 135M | 108.5 | 145.8 | 0.74× |
| Llama 1B | 17.8 | 18.8 | 0.95× |
| Llama 3B | 6.8 | 6.7 | **1.01×** |

Decode is memory-bandwidth bound (GEMV). On larger models where memory bandwidth dominates over kernel overhead, dotLLM is essentially **at parity with llama.cpp**. The remaining gap on small models is per-token overhead (function call dispatch, sampler pipeline) that gets amortized on larger models.

### Prefill (compute bound)

| Model | dotLLM tok/s | llama.cpp tok/s | Ratio |
|-------|-------------|----------------|-------|
| SmolLM 135M | 170.0 | 523.7 | 0.32× |
| Llama 1B | 41.6 | 84.6 | 0.49× |
| Llama 3B | 16.5 | 29.1 | 0.57× |

Prefill is compute-bound (GEMM). dotLLM uses batched GEMM but llama.cpp's matmul kernels are more heavily optimized. The gap narrows on larger models (0.32× → 0.57×) as compute dominates overhead.

### Roadmap to Parity

| Optimization | Expected impact | Roadmap step |
|-------------|----------------|--------------|
| ~~SIMD-tuned Q8_0 kernels~~ | ~~2-4× kernel speedup~~ | ~~Phase 2, Step 10~~ :white_check_mark: |
| ~~Batched GEMM for prefill~~ | ~~5-10× prefill speedup~~ | ~~Phase 2, Step 11~~ :white_check_mark: |
| ~~Multi-threaded inference~~ | ~~4-8× on multi-core~~ | ~~Phase 2, Step 20~~ :white_check_mark: |
| Cache-tiled GEMM | ~1.5-2× prefill improvement | Phase 2, optimization |
| CUDA GPU backend | 10-50× prefill, 3-10× decode | Phase 3, Step 29 |

---

## Historical Results

<details>
<summary>Earlier measurements (pre-Step 13, manual runs)</summary>

### SmolLM-135M (2026-03-04)

- **Hardware**: AMD Ryzen 9 7950X, 64 GB DDR5, Windows 11, CPU-only

#### llama.cpp (b5291, -n 2)

```
common_perf_print:        load time =     227.69 ms
common_perf_print: prompt eval time =      11.28 ms /     5 tokens (    2.25 ms per token,   443.46 tokens per second)
common_perf_print:        eval time =       8.27 ms /     1 runs   (    8.27 ms per token,   120.93 tokens per second)
common_perf_print:       total time =      21.27 ms /     6 tokens
```

#### dotLLM (Release, -n 20)

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
```

### Llama 3.2 1B — Step 10 Optimization (2026-03-05)

- **Hardware**: AMD Ryzen 7 5800HS, 16 GB DDR4, Windows 11, CPU-only

#### Llama 3.2 1B — before Step 10 (row-by-row AVX2 + scalar quantize)

```
╭─────────────┬────────────┬────────┬──────────┬──────────╮
│ Phase       │       Time │ Tokens │ ms/token │ tokens/s │
├─────────────┼────────────┼────────┼──────────┼──────────┤
│ Load        │  553.66 ms │      — │        — │        — │
│ Prompt eval │ 1392.68 ms │      5 │   278.54 │     3.59 │
│ Eval        │ 3249.81 ms │     19 │   171.04 │     5.85 │
│ Total       │ 4661.93 ms │     25 │        — │     5.36 │
╰─────────────┴────────────┴────────┴──────────┴──────────╯
```

#### Llama 3.2 1B — after Step 10 (4-row AVX2 + FMA + SIMD quantize)

```
╭─────────────┬────────────┬────────┬──────────┬──────────╮
│ Phase       │       Time │ Tokens │ ms/token │ tokens/s │
├─────────────┼────────────┼────────┼──────────┼──────────┤
│ Load        │  530.43 ms │      — │        — │        — │
│ Prompt eval │ 3387.24 ms │      5 │   677.45 │     1.48 │
│ Eval        │ 2713.50 ms │     19 │   142.82 │     7.00 │
│ Total       │ 6120.07 ms │     25 │        — │     4.08 │
╰─────────────┴────────────┴────────┴──────────┴──────────╯
```

| Metric | Before Step 10 | After Step 10 | Change |
|--------|---------------|---------------|--------|
| Eval per token | 171.04 ms | 142.82 ms | **1.20× faster** |
| Eval tokens/s | 5.85 | 7.00 | **+20%** |

</details>
