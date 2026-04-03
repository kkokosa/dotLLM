# Warm-up — dotLLM

## Overview

At server startup, dotLLM runs configurable warm-up inference passes before marking itself as ready. This eliminates the "cold start" latency penalty on the first real request by triggering .NET JIT compilation of hot paths and exercising CUDA/cuBLAS kernel pipelines.

## Why Warm-up Matters

### .NET JIT Compilation

.NET uses tiered JIT compilation:

1. **Tier-0** (Quick JIT): On first call, methods are compiled quickly with minimal optimization. This is fast to compile but produces slower code.
2. **Tier-1** (Optimized JIT + Dynamic PGO): After ~30 invocations, methods are recompiled with full optimizations using profiling data collected during Tier-0 execution.

Dynamic PGO (Profile-Guided Optimization), enabled by default since .NET 8, uses Tier-0 profiling data for:
- **Devirtualization**: Replacing virtual/interface calls with direct calls based on observed types
- **Guarded Devirtualization (GDV)**: Speculatively inlining the most common implementation with a type-check guard
- **Hot/cold block layout**: Reordering code blocks based on observed branch frequencies
- **Loop cloning**: Creating optimized versions of loops for common iteration patterns

Without warm-up, the first inference request pays the full Tier-0 JIT cost for every hot-path method: `Forward`, `MatMul`, `Attention`, `RmsNorm`, `SiLU`, the entire sampling pipeline, and hundreds of supporting methods. This adds ~200-500ms to the first request on CPU inference.

### CUDA Kernel Loading

The CUDA backend has two sources of first-call overhead:

1. **PTX JIT compilation**: dotLLM ships PTX (parallel thread execution) intermediate code that the CUDA driver JIT-compiles to SASS (device-native assembly) on first load. This happens in the `CudaKernels` constructor during model loading — before warm-up — so the PTX→SASS cost is already paid at load time. The CUDA compute cache (`~/.nv/ComputeCache`) persists compiled kernels across process restarts.

2. **cuBLAS first-call overhead**: cuBLAS performs internal JIT optimization on the first GEMM call — workspace allocation, algorithm selection, and heuristic tuning for the specific matrix dimensions and GPU architecture. This adds ~100-300ms to the first forward pass. A warm-up inference pass exercises all cuBLAS code paths, paying this cost before any real request arrives.

### Combined Effect

Without warm-up, the first request experiences:

| Source | Penalty |
|--------|---------|
| .NET Tier-0 JIT (CPU paths) | ~200-500ms |
| cuBLAS first-call (GPU) | ~100-300ms |
| Attention/sampling pipeline JIT | ~50-100ms |
| **Total cold-start penalty** | **~300-900ms** |

With 3 warm-up iterations, the first real request runs at steady-state performance.

## Configuration

### WarmupOptions

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `Enabled` | `bool` | `true` | Enable/disable warm-up |
| `DummyPrompt` | `string` | `"The quick brown fox..."` | Prompt text for warm-up passes |
| `MaxTokens` | `int` | `16` | Tokens to generate per iteration |
| `Iterations` | `int` | `3` | Number of warm-up passes |

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--no-warmup` | `false` | Disable warm-up |
| `--warmup-iterations N` | `3` | Set number of warm-up iterations |

### Usage

```bash
# Default: warm-up enabled with 3 iterations
dotllm serve model.gguf

# Disable warm-up (faster startup, slower first request)
dotllm serve model.gguf --no-warmup

# Single iteration (minimal warm-up)
dotllm serve model.gguf --warmup-iterations 1

# More iterations for large models (ensures Tier-1 promotion)
dotllm serve model.gguf --warmup-iterations 5
```

### Example Output

```
[dotllm] Loading model from llama-3.2-1b.Q8_0.gguf...
[dotllm] CPU inference (8 threads)
[dotllm] Warming up (3 iterations, 10 prompt tokens, 16 max gen tokens)...
[dotllm]   Iteration 1/3: 1250ms
[dotllm]   Iteration 2/3: 380ms
[dotllm]   Iteration 3/3: 340ms
[dotllm] Warm-up complete in 1970ms
```

Note the large drop between iteration 1 and 2 — this is the Tier-0→Tier-1 JIT transition.

## Why 3 Iterations?

- **Iteration 1**: All hot-path methods JIT-compiled at Tier-0 (slow compilation, unoptimized code). CUDA/cuBLAS first-call overhead paid. Slowest iteration by far.
- **Iteration 2**: Methods with high call counts within a single inference pass (inner loops in MatMul, Attention) get promoted to Tier-1 with Dynamic PGO profiles. Significant speedup.
- **Iteration 3**: Most remaining hot-path methods reach Tier-1. Near steady-state performance.
- **Beyond 3**: Diminishing returns. The critical matmul/attention inner loops are fully optimized after 2-3 passes. Additional iterations may help niche code paths but the improvement is marginal.

## Readiness Probe

The `/ready` health endpoint returns HTTP 503 while warm-up is running and HTTP 200 only after warm-up completes. This integrates naturally with:

- **Kubernetes readiness probes**: Pods only receive traffic after warm-up
- **Load balancers**: Health checks gate on warm-up completion
- **Orchestrators**: Dependent services wait for readiness

The `/health` endpoint always returns 200 (the server process is running), while `/ready` reflects whether the model is loaded and warmed.

## Hot-Swap

When a model is swapped via `POST /v1/models/load`, warm-up runs automatically for the new model as part of the `LoadModel()` call. During the swap:

1. `IsReady` is set to `false` (new requests get 503)
2. Old model is disposed
3. New model is loaded
4. Warm-up runs for the new model
5. `IsReady` is set to `true`

This ensures every loaded model is fully warmed before receiving real traffic.

## Implementation

The warm-up system consists of two components in `DotLLM.Engine`:

- **`WarmupOptions`**: Sealed record with configuration properties
- **`WarmupRunner`**: Static class that executes warm-up passes via `TextGenerator.Generate()`

The runner uses greedy sampling (temperature=0) with a short dummy prompt to exercise the full pipeline with minimal compute. Each iteration's KV-cache is managed normally by `TextGenerator` (disposed or stored in the prefix cache).
