# dotLLM

**High-performance LLM inference engine written natively in C#/.NET**

[![CI](https://github.com/kkokosa/dotLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/kkokosa/dotLLM/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![.NET](https://img.shields.io/badge/.NET-10-purple.svg)](https://dotnet.microsoft.com/)

[Documentation](docs/) · [Roadmap](docs/ROADMAP.md) · [Discussions](https://github.com/kkokosa/dotLLM/discussions)

---

## About

dotLLM is a ground-up LLM inference engine for .NET — not a wrapper around llama.cpp or Python libraries. All orchestration, model loading, tokenization, sampling, and CPU compute are implemented in pure C#, with a thin C/CUDA native library for GPU kernels. It targets transformer-based models (Llama, Mistral, Phi, Qwen, DeepSeek) with SIMD-optimized CPU and CUDA GPU backends.

> **Status**: Phase 1 complete — dotLLM can load a GGUF model, tokenize a prompt, run the full Llama forward pass with KV-cache, sample tokens, and generate coherent multi-token output on CPU. See [Roadmap](#roadmap) for Phase 2 (Q4_K_M, chat templates, streaming, hooks).

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

dotLLM requires [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0).

**Build from source:**

```bash
git clone https://github.com/kkokosa/dotLLM.git
cd dotLLM
dotnet build
```

**Run tests:**

```bash
dotnet test
```

> Integration tests automatically download [SmolLM-135M](https://huggingface.co/QuantFactory/SmolLM-135M-GGUF) Q8_0 (~145 MB) to `~/.dotllm/test-cache/`.

There is no NuGet package yet — the project is in early development. Follow the [Roadmap](#roadmap) for progress toward the first release.

## News

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
| **2 — Practical Local Inference** | Engine metrics, inference benchmarks, Q4_K_M, chat templates, streaming, hooks, logit lens, more architectures | In Progress (8/13) |
| **2B — CPU Performance** | Decode dispatch optimization, Q8_1 input format, weight repacking, outer-product GEMM, tiled attention, fast exp, operator fusion, NUMA | Planned (0/8) |
| **3 — GPU Acceleration** | CUDA backend, CPU/GPU hybrid, KV-cache quantization | Planned |
| **4 — Production Serving** | OpenAI API, continuous batching, paged KV-cache, structured output, tool calling | Planned |
| **5 — Expand** | LoRA, MLA, SAE, multi-GPU tensor parallelism, ROCm | Planned |

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
