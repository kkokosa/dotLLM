# Native AOT — dotLLM

## Status: Experimental

Native AOT publishing is an **opt-in, experimental** feature. JIT compilation with Dynamic PGO remains the default and recommended mode for best inference throughput.

## Overview

.NET Native AOT compiles the entire application ahead-of-time to native machine code, producing a single self-contained binary with no dependency on the .NET runtime. For dotLLM this means:

- **Instant startup**: ~50ms vs ~500ms with JIT (eliminates Tier-0/Tier-1 JIT compilation)
- **Single-file binary**: One `dotllm` executable, no runtime to install
- **Smaller deployment footprint**: No .NET SDK or runtime required on target machine
- **Container-friendly**: Minimal base image (no `aspnet` runtime layer needed)

## How to Build

### Prerequisites

- .NET 10 SDK
- Platform-specific AOT compiler:
  - **Windows**: Visual Studio Build Tools (cl.exe) or full Visual Studio
  - **Linux**: `clang` or `gcc`
  - **macOS**: Xcode command-line tools

### Build Commands

```bash
# Windows x64
dotnet publish src/DotLLM.Cli -c Release -p:PublishAot=true -r win-x64

# Linux x64
dotnet publish src/DotLLM.Cli -c Release -p:PublishAot=true -r linux-x64

# macOS Apple Silicon
dotnet publish src/DotLLM.Cli -c Release -p:PublishAot=true -r osx-arm64
```

Or use the provided publish profiles:

```bash
dotnet publish src/DotLLM.Cli -c Release -p:PublishProfile=aot-win-x64
dotnet publish src/DotLLM.Cli -c Release -p:PublishProfile=aot-linux-x64
dotnet publish src/DotLLM.Cli -c Release -p:PublishProfile=aot-osx-arm64
```

The output binary is in `src/DotLLM.Cli/bin/Release/net10.0/<rid>/publish/`.

## Performance Trade-offs

### Startup (AOT wins)

| Metric | JIT | Native AOT |
|--------|-----|------------|
| Process start → ready | ~500ms | ~50ms |
| JIT warm-up needed | Yes (Tier-0 → Tier-1 recompilation) | No |
| First inference latency | High (JIT + Dynamic PGO ramp-up) | Low (code is pre-compiled) |

### Steady-state Inference (JIT wins on CPU, neutral on GPU)

| Metric | JIT + Dynamic PGO | Native AOT |
|--------|-------------------|------------|
| Interface dispatch | Devirtualized via GDV | Virtual (indirect call) |
| Hot-path optimization | Profile-guided (branch layout, inlining) | Static analysis only |
| Expected CPU throughput delta | Baseline | ~10–40% slower on models ≥1B |
| Expected GPU throughput delta | Baseline | Identical (GPU-bound) |

**Why JIT is faster at steady-state on CPU**: .NET's Dynamic PGO (Profile-Guided Optimization) observes runtime behavior during Tier-0 execution and applies:

- **Guarded Devirtualization (GDV)**: Speculatively inlines the most common `IBackend`, `IAttentionMechanism`, `ISamplerStep` implementations with a type-check guard. In dotLLM this matters because the entire inference pipeline dispatches through interfaces.
- **Hot/cold code layout**: Reorders basic blocks based on observed branch frequencies in compute kernels.
- **Loop cloning**: Creates optimized loop variants for common iteration patterns in SIMD kernels.

Native AOT cannot do any of this — it has only static whole-program analysis at compile time.

**Why GPU decode is unaffected**: GPU inference is dominated by CUDA kernel execution and cuBLAS GEMM — the .NET host code is not on the GPU hot path. The only .NET code in the GPU decode loop is thin P/Invoke dispatch (microseconds vs milliseconds for the GPU kernels). JIT vs AOT makes no measurable difference here.

### CUDA Warm-up

CUDA warm-up is still required with Native AOT. The cuBLAS first-call overhead (~100–300ms for workspace allocation and algorithm selection) is a GPU runtime cost, not a .NET JIT cost.

## Benchmark Results

Measured with `scripts/test_models_aot.py` on Windows 11, AMD Ryzen, RTX GPU. Single run per binary, greedy decode (temperature=0), 2 generated tokens. The wall-clock time includes process startup, model loading (mmap), and inference.

### CPU Inference

| Model | Arch | Quant | JIT wall | JIT tok/s | AOT wall | AOT tok/s | Startup | Decode |
|-------|------|-------|----------|-----------|----------|-----------|---------|--------|
| SmolLM-135M | Llama | Q8_0 | 0.83s | 29.1 | 0.33s | 114.6 | 2.50x | 3.93x |
| SmolLM2-135M-Instruct | Llama | Q8_0 | 0.76s | 19.4 | 0.35s | 49.7 | 2.15x | 2.56x |
| Llama-3.2-1B-Q4 | Llama | Q4_K_M | 1.98s | 13.7 | 1.56s | 8.2 | 1.27x | 0.60x |
| Llama-3.2-1B-Q8 | Llama | Q8_0 | 1.74s | 16.0 | 1.56s | 14.0 | 1.11x | 0.87x |
| Llama-3.2-3B-Q4 | Llama | Q4_K_M | 2.28s | 8.5 | 3.33s | 2.8 | 0.69x | 0.33x |
| Llama-3.2-3B-Q8 | Llama | Q8_0 | 7.84s | 5.7 | 3.75s | 5.4 | 2.09x | 0.95x |
| Bielik-1.5B | Llama | Q8_0 | 1.95s | 12.7 | 1.82s | 8.0 | 1.07x | 0.62x |
| Bielik-11B | Llama | Q4_K_M | 14.81s | 3.8 | 12.48s | 0.9 | 1.19x | 0.25x |
| Qwen2.5-0.5B | Qwen | Q8_0 | 1.56s | 21.9 | 0.86s | 33.9 | 1.81x | 1.55x |
| Qwen3-0.6B | Qwen | Q8_0 | 1.24s | 21.3 | 1.00s | 29.1 | 1.24x | 1.36x |
| Phi-3-mini | Phi | default | 4.01s | 7.4 | 3.40s | 2.9 | 1.18x | 0.39x |
| Phi-4-mini | Phi | Q8_0 | 8.40s | 6.2 | 4.21s | 4.4 | 1.99x | 0.71x |
| Ministral-3-3B | Mistral | default | 7.77s | 7.2 | 4.11s | 4.2 | 1.89x | 0.58x |
| Mistral-7B-Q8 | Mistral | Q8_0 | 15.65s | 3.2 | 7.15s | 2.6 | 2.19x | 0.82x |
| **Average** | | | | | | | **1.60x** | **1.11x** |

Correctness: JIT 14/14, AOT 14/14.

**Startup speedup** = JIT wall-clock / AOT wall-clock (>1 means AOT is faster overall).
**Decode ratio** = AOT tok/s / JIT tok/s (>1 means AOT decodes faster, <1 means JIT decodes faster).

### GPU Inference

| Model | Arch | Quant | GPU layers | JIT wall | JIT tok/s | AOT wall | AOT tok/s | Startup | Decode |
|-------|------|-------|------------|----------|-----------|----------|-----------|---------|--------|
| SmolLM-135M | Llama | Q8_0 | 30 | 0.73s | 43.6 | 0.53s | 43.8 | 1.37x | 1.00x |
| SmolLM2-135M | Llama | Q8_0 | 30 | 0.90s | 43.6 | 0.51s | 43.8 | 1.77x | 1.00x |
| Llama-3.2-1B-Q4 | Llama | Q4_K_M | 16 | 7.57s | 1.6 | 6.72s | 1.6 | 1.13x | 1.00x |
| Llama-3.2-1B-Q8 | Llama | Q8_0 | 16 | 6.28s | 4.3 | 5.46s | 4.3 | 1.15x | 1.00x |
| Bielik-1.5B | Llama | Q8_0 | 28 | 10.65s | 3.7 | 7.56s | 3.9 | 1.41x | 1.07x |
| **Average** | | | | | | | | **1.36x** | **1.02x** |

Correctness: JIT 5/5, AOT 5/5.

### Analysis

**Startup**: AOT consistently wins — avg **1.60x on CPU**, **1.36x on GPU**. The AOT advantage is larger on CPU because the ~500ms JIT compilation overhead is a bigger proportion of total time. On GPU, CUDA context creation and cuBLAS initialization add fixed overhead that dilutes the .NET startup improvement.

**CPU decode — model size matters**:
- **Small models (< 1B)**: AOT is *faster* (1.36x–3.93x). These models fit in CPU cache, so the bottleneck is instruction dispatch. With only 2 generated tokens, JIT never reaches Tier-1 (Dynamic PGO needs ~30 invocations), so AOT's fully-optimized native code outperforms JIT's unoptimized Tier-0 code.
- **Medium models (1B–3B)**: JIT starts to win (0.33x–0.95x decode ratio). Weight memory bandwidth becomes the bottleneck, and Dynamic PGO's devirtualization of the interface-heavy decode loop pays off.
- **Large models (7B–11B)**: JIT wins decisively (0.25x–0.82x). The PGO advantage compounds across more transformer layers.

In production with longer generation runs (hundreds of tokens), the JIT steady-state advantage would be larger because Tier-1 recompilation would have completed. The 2-token benchmark actually *undersells* the JIT decode advantage on larger models.

**GPU decode — identical**: 1.00x–1.07x across all models. The GPU kernels (PTX/cuBLAS) are the bottleneck, not .NET host code. JIT vs AOT is irrelevant for GPU-bound workloads.

### Reproducing

```bash
# CPU comparison
python scripts/test_models_aot.py --device cpu --save aot-cpu.json

# GPU comparison
python scripts/test_models_aot.py --device gpu --save aot-gpu.json

# Both devices, small models only
python scripts/test_models_aot.py --device both --size "<3B"

# Redisplay saved results
python scripts/test_models_aot.py --show aot-cpu.json
```

## What Works

All core inference features are fully functional under Native AOT:

- CPU inference (all SIMD kernels, multi-threaded)
- CUDA GPU inference (all PTX kernels, cuBLAS)
- CPU/GPU hybrid layer offloading
- All quantization formats (Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0)
- KV-cache (simple, quantized)
- All model architectures (Llama, Mistral, Phi, Qwen, DeepSeek)
- Streaming generation (`IAsyncEnumerable`)
- Chat templates (Jinja2-subset interpreter)
- Sampling pipeline (temperature, top-k, top-p, min-p, repetition penalty)
- Constrained decoding (JSON, JSON Schema, regex, grammar)
- Tool calling
- ASP.NET server (all OpenAI-compatible endpoints)
- Web chat UI (`serve` command)
- Prompt caching
- HuggingFace model download

## Known Limitations

### Spectre.Console.Cli

The CLI framework (Spectre.Console.Cli) uses reflection to discover command settings properties. dotLLM preserves these types via `TrimmerRoots.xml`, which works for the known command set. If you add new CLI commands, they must also be added to `TrimmerRoots.xml`.

### Binary Size

Expect 30–80MB depending on the platform and which backends are linked. This is larger than a framework-dependent JIT deployment (~5MB app + shared runtime) but smaller than a self-contained JIT publish (~70MB + runtime).

### Future Features

Some planned features may be incompatible with AOT:

- **Runtime LoRA loading**: Dynamic weight shape instantiation may require runtime generic specialization
- **Plugin system**: If implemented with runtime assembly loading

These would require JIT fallback when enabled.

## When to Use AOT vs JIT

| Scenario | Recommended |
|----------|-------------|
| Local development / experimentation | JIT |
| Benchmarking / performance tuning | JIT (Dynamic PGO gives best throughput) |
| Production server (long-running) | JIT (steady-state throughput matters more than startup) |
| CLI tool distribution | AOT (instant startup, no runtime dependency) |
| Docker / container deployment | AOT (minimal image, fast cold start) |
| Edge / embedded deployment | AOT (single binary, no SDK needed) |
| Serverless / scale-to-zero | AOT (startup latency dominates) |

## Feature Compatibility Matrix

| Feature | JIT | Native AOT | Notes |
|---------|-----|------------|-------|
| CPU inference | Yes | Yes | |
| CUDA inference | Yes | Yes | |
| Hybrid CPU/GPU | Yes | Yes | |
| All quant formats | Yes | Yes | |
| Chat templates | Yes | Yes | Tree-walking interpreter, no reflection |
| Streaming | Yes | Yes | |
| Server endpoints | Yes | Yes | Source-generated JSON |
| Tool calling | Yes | Yes | |
| Constrained decoding | Yes | Yes | FSM/PDA, no reflection |
| Prompt caching | Yes | Yes | |
| HuggingFace download | Yes | Yes | Source-generated JSON |
| Dynamic PGO | Yes | No | CPU decode: ~15–75% slower on models ≥1B; GPU: unaffected |
| JIT warm-up benefit | Yes | N/A | AOT doesn't need warm-up |
| Runtime LoRA (future) | Yes | TBD | May require JIT |

## Troubleshooting

### Trimming Warnings

If you see IL2026/IL2046/IL2075 warnings during AOT publish, a type accessed via reflection is being trimmed. Add it to `src/DotLLM.Cli/TrimmerRoots.xml`:

```xml
<assembly fullname="YourAssembly">
    <type fullname="Your.Namespace.YourType" preserve="all" />
</assembly>
```

### Adding New CLI Commands

When adding a new Spectre.Console.Cli command, add both the command class and its `Settings` nested type to `TrimmerRoots.xml`:

```xml
<type fullname="DotLLM.Cli.Commands.NewCommand" preserve="all" />
<type fullname="DotLLM.Cli.Commands.NewCommand+Settings" preserve="all" />
```

### Runtime Errors

If the AOT binary crashes with `MissingMethodException` or `TypeInitializationException`, it typically means a type or method was trimmed that is used via reflection. Check the publish output for trimming warnings and add the affected types to `TrimmerRoots.xml`.

### Build Failures

Native AOT requires a platform-native C/C++ linker:
- **Windows**: Install "Desktop development with C++" workload in Visual Studio
- **Linux**: `sudo apt install clang` or `sudo apt install gcc`
- **macOS**: `xcode-select --install`
