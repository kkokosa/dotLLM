# CLAUDE.md — dotLLM Project Guide

## Project Identity

**dotLLM** is an open-source, high-performance LLM inference engine written natively in C#/.NET. It targets transformer-based models (Llama, Mistral, Phi, Qwen, DeepSeek) with CPU (SIMD-optimized) and CUDA GPU backends. Not a wrapper — a ground-up implementation.

- **Repository**: https://github.com/kkokosa/dotLLM
- **License**: GPLv3
- **Target framework**: .NET 10
- **Languages**: C# + thin C/CUDA native library for GPU kernels

## Design Philosophy

1. **Native .NET first** — All orchestration, model loading, tokenization, sampling, scheduling, CPU compute in pure C#.
2. **Unmanaged memory for tensors** — `NativeMemory.AlignedAlloc` (64-byte). Zero GC allocations on inference hot path.
3. **Hybrid GPU architecture** — Thin native C/CUDA lib via `[LibraryImport]`. GPU memory as opaque `IntPtr` — tensor data never crosses P/Invoke boundary.
4. **Backend-pluggable** — `IBackend` interface, separate NuGet packages per backend (CPU, CUDA, ROCm).
5. **Parameterized architectures** — Single `TransformerBlock` handles Llama/Mistral/Phi/Qwen/DeepSeek via `ModelConfig`.
6. **Interpretability first-class** — Zero-cost hook points for activation capture, logit lens, SAE integration.
7. **Multi-GPU aware from day one** — Explicit device placement in all tensor operations, even before multi-GPU is implemented.

## Solution Structure

```
dotLLM/
├── CLAUDE.md                          # This file — always read
├── docs/                              # Detailed specs — read on demand per task
├── src/
│   ├── DotLLM.Core/                   # Interfaces, abstractions, tensor types, config
│   │   ├── Tensors/                   # ITensor, TensorShape, DType
│   │   ├── Backends/                  # IBackend, IKernelRunner, DevicePlacement
│   │   ├── Models/                    # IModel, IModelArchitecture, ModelConfig
│   │   ├── Attention/                 # IAttentionMechanism, IAttentionStrategy
│   │   ├── PositionEncoding/          # IPositionEncoding
│   │   ├── Sampling/                  # ISamplerStep, ILogitProcessor, IStopCondition
│   │   ├── Constraints/              # IDecodingConstraint, TokenMask
│   │   ├── Diagnostics/              # IInferenceHook, HookPoint
│   │   ├── Telemetry/                # IInferenceMetrics, IRequestTracer
│   │   └── Configuration/             # InferenceOptions, QuantizationType
│   ├── DotLLM.Models/                 # GGUF/SafeTensors loaders, model architectures, LoRA adapters
│   ├── DotLLM.Tokenizers/             # BPE, SentencePiece, HF parsers, chat templates
│   ├── DotLLM.Cpu/                    # CPU backend, SIMD kernels
│   ├── DotLLM.Cuda/                   # CUDA backend, P/Invoke interop
│   ├── DotLLM.Engine/                 # KV-cache, scheduler, samplers, constraints, speculative decoding, prefix cache
│   ├── DotLLM.Diagnostics/           # Hooks, activation capture, logit lens, SAE
│   ├── DotLLM.Telemetry/             # Metrics, tracing
│   └── DotLLM.Server/                 # ASP.NET OpenAI-compatible API, tool calling, rate limiting
├── native/                            # C/CUDA shared library (CMake)
│   ├── include/                       # dotllm_native.h
│   ├── src/cuda/                      # CUDA kernels
│   ├── src/cublas/                    # cuBLAS wrappers
│   ├── src/nccl/                      # Multi-GPU communication
│   └── hip/                           # ROCm compat (future)
├── tests/
│   ├── DotLLM.Tests.Unit/
│   └── DotLLM.Tests.Integration/
├── benchmarks/
│   └── DotLLM.Benchmarks/
└── samples/
    ├── DotLLM.Sample.Console/
    ├── DotLLM.Sample.Server/
    ├── DotLLM.Sample.ToolCalling/
    └── DotLLM.Sample.Interpretability/
```

## Code Style & Conventions (applies to ALL code)

- File-scoped namespaces.
- `readonly record struct` for small value types (TensorShape, DType, TokenId).
- `Span<T>` / `ReadOnlySpan<T>` over arrays in method signatures.
- `[MethodImpl(MethodImplOptions.AggressiveInlining)]` on small hot-path methods.
- `[SkipLocalsInit]` on performance-critical methods.
- XML doc comments on all public APIs.
- `<Nullable>enable</Nullable>` project-wide.
- GPU handles: `devicePtr`/`dPtr`. CPU: `hostPtr`/`hPtr`.
- `IAsyncEnumerable<T>` for streaming token generation.
- Composability over inheritance. Interfaces and records, not deep class hierarchies.

## Memory Management Rules

- **NEVER** allocate managed arrays for tensor data. Use `NativeMemory.AlignedAlloc` (64-byte for AVX-512, 32-byte for AVX2).
- `ArrayPool<T>.Shared` for temporary scratch buffers — return promptly.
- Model weights: memory-mapped via `MemoryMappedFile`. No managed heap copies. OS page cache provides cross-process weight sharing.
- All unmanaged memory wrapped in `IDisposable` with deterministic cleanup.
- Tensor metadata (shape, stride, pointer): structs, not classes.
- GC: Server GC, `SustainedLowLatency` mode during inference.

## SIMD & Vectorization Rules

- Foundation: `System.Numerics.Tensors.TensorPrimitives` for standard ops.
- Hot inner loops (quantized matmul, RoPE): `System.Runtime.Intrinsics` — prefer cross-platform `Vector128<T>`/`Vector256<T>`, use platform-specific (`Fma.MultiplyAdd`, `Avx2.MultiplyAddAdjacent`) only when measurably faster.
- Always provide scalar fallback.
- Align data: 64 bytes for AVX-512, 32 bytes for AVX2.

## GPU Interop Rules

- Native library: flat C API only (no C++ classes across boundary).
- `[LibraryImport]` (source-generated, .NET 7+), not `[DllImport]`.
- `[SuppressGCTransition]` for trivially short native calls.
- GPU memory: `nint` handles in C#. Native side owns pointer semantics.
- Coarse-grained API: `LoadTensors`, `Forward`, `Attention` — each call takes ms.
- Ship native binaries under `runtimes/{rid}/native/` per .NET RID conventions.

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| License | GPLv3 | Strong copyleft. |
| GPU interop | P/Invoke to custom C/CUDA lib | ManagedCUDA=GPLv3+unmaintained, ILGPU can't cuBLAS, Silk.NET no CUDA. |
| Tensor memory | `NativeMemory.AlignedAlloc` | Zero GC pressure. Enables mmap loading. |
| Model loading | Memory-mapped GGUF | OS demand-paging. 7GB loads in ms. |
| SIMD | TensorPrimitives + Intrinsics | Standard ops via TP; hand-tuned for quantized loops. |
| Runtime | JIT + Dynamic PGO default | Better steady-state than NativeAOT. AOT opt-in for edge. |
| Model format | GGUF primary | Self-contained, quantization, HF ecosystem. |
| Attention | `IAttentionMechanism` + `IAttentionStrategy` | Mechanism (GQA/MLA) separate from kernel (naive/flash/paged). |
| Position encoding | `IPositionEncoding`, optional | RoPE, ALiBi, absolute, none. |
| Sampler | Composable `ISamplerStep` chain | Independent, reorderable, extensible. |
| Structured output | FSM/PDA logit masking | Guaranteed valid JSON/schema/regex. |
| Chat templates | Jinja2-subset interpreter | All formats, no per-model hardcoding. |
| Multi-GPU | TP via NCCL | Abstractions ready before implementation. |
| LoRA | Runtime, no merge | Instant switching, concurrent adapters. |
| Diagnostics | Hooks, zero-cost when off | HF-level interpretability in .NET. |
| Observability | `System.Diagnostics.Metrics` + `Activity` | Native OpenTelemetry, zero-cost. |

## Documentation Index

**Read the relevant doc(s) before starting work on a module.**

| Topic | Document | Read when working on... |
|-------|----------|------------------------|
| System architecture & data flow | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Any major feature, onboarding |
| GGUF binary format & parsing | [docs/GGUF_FORMAT.md](docs/GGUF_FORMAT.md) | Model loading, GGUF parser |
| Quantization formats & kernels | [docs/QUANTIZATION.md](docs/QUANTIZATION.md) | Dequant, quantized matmul, model loading |
| Attention mechanisms | [docs/ATTENTION.md](docs/ATTENTION.md) | GQA, MHA, MQA, MLA, flash attention |
| Position encoding | [docs/POSITION_ENCODING.md](docs/POSITION_ENCODING.md) | RoPE, ALiBi, context extension |
| Model config & arch pattern | [docs/MODEL_CONFIG.md](docs/MODEL_CONFIG.md) | Adding model architectures |
| Tokenizers & chat templates | [docs/TOKENIZERS.md](docs/TOKENIZERS.md) | BPE, SentencePiece, templates, tool calling |
| Sampling pipeline | [docs/SAMPLING.md](docs/SAMPLING.md) | Sampler steps, beam search, stop conditions |
| Constrained decoding | [docs/CONSTRAINED_DECODING.md](docs/CONSTRAINED_DECODING.md) | JSON, schema, regex, grammar |
| KV-cache management | [docs/KV_CACHE.md](docs/KV_CACHE.md) | Paged attention, KV quant, prefix sharing |
| Batch scheduling | [docs/SCHEDULING.md](docs/SCHEDULING.md) | Continuous batching, preemption, priority |
| Speculative decoding | [docs/SPECULATIVE.md](docs/SPECULATIVE.md) | Draft-verify-accept, rollback |
| LoRA adapters | [docs/LORA.md](docs/LORA.md) | Adapter loading, multi-adapter serving |
| Diagnostics & interpretability | [docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md) | Hooks, logit lens, SAE |
| Telemetry & observability | [docs/TELEMETRY.md](docs/TELEMETRY.md) | Metrics, request tracing |
| Server & API | [docs/SERVER.md](docs/SERVER.md) | Endpoints, rate limiting, warm-up |
| Multi-GPU | [docs/MULTI_GPU.md](docs/MULTI_GPU.md) | Tensor/pipeline parallelism, NCCL |
| Implementation roadmap | [docs/ROADMAP.md](docs/ROADMAP.md) | Planning, task sequencing |

## Development Workflow

All development follows an issue-driven workflow.

1. **Every task starts with a GitHub issue.** Before writing code, a GitHub issue must exist describing the work with acceptance criteria and relevant roadmap/doc references.
2. **Dedicated branch per issue.** Branch from `main` using: `issue/{number}-{short-kebab-description}` (e.g., `issue/7-gguf-loader`).
3. **Commits reference the issue.** Include `(#{number})` in commit messages.
4. **PRs close the issue.** Use `Closes #{N}` in the PR description. PR targets `main`.
5. **One issue, one branch, one PR.** If scope grows, split into a new issue.
6. **Read the relevant docs first.** Before starting, read docs listed in the Documentation Index for the module being implemented.
7. **Keep README in sync.** When a PR completes a roadmap step, update `docs/ROADMAP.md` (add `:white_check_mark:` to the step) and `README.md` (bump the step count in the Roadmap table, e.g., "4/9" → "5/9"; when a phase completes change status to "Done"; add a News entry for significant milestones).

## What Claude Should Know

- **Author**: Konrad — expert .NET dev, "Pro .NET Memory Management" (2nd ed.), MVP, 20+ yrs perf. AI/agents at Nethermind. Do NOT over-explain .NET basics.
- **Priority**: Correctness then Performance then Extensibility. Every allocation matters. Benchmark before/after kernel changes.
- **SIMD kernels**: Verify numerical accuracy against scalar reference first, then optimize.
- **GGUF**: Spec is source of truth. Check llama.cpp when in doubt.
- **CUDA**: Reference llama.cpp `ggml-cuda/` for proven kernels.
- **New architectures**: Verify against HuggingFace transformers for correctness.
- **Native API**: Keep minimal and stable. Changes require cross-platform rebuild.
- **Diagnostics/hooks**: Zero overhead when disabled. Null check, not event pattern.
- **Telemetry**: Zero overhead when no listener.
- **Device placement**: Always explicit. Never implicit "current device."
- **Before implementing**: Read the relevant doc from the index above.