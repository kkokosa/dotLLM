# Architecture — dotLLM

## Component Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│  DotLLM.Server (ASP.NET Minimal API)                            │
│  ├── /v1/chat/completions, /v1/completions, /v1/embeddings      │
│  ├── /v1/models, /v1/tokenize, /v1/detokenize                  │
│  ├── Tool calling protocol handler                              │
│  └── Rate limiting, API key auth, request priority              │
├─────────────────────────────────────────────────────────────────┤
│  DotLLM.Engine (Orchestration)                                  │
│  ├── InferenceEngine          — Main entry point                │
│  ├── IScheduler               — Continuous batch scheduling     │
│  ├── PagedKvCacheManager      — Block allocation, prefix cache  │
│  ├── SamplerPipeline          — Composable ISamplerStep chain   │
│  ├── ConstraintEngine         — FSM/PDA for structured output   │
│  ├── ISpeculativeDecoder      — Draft-verify-accept loop        │
│  └── IAdapterManager          — LoRA runtime management         │
├─────────────────────────────────────────────────────────────────┤
│  DotLLM.Models                     DotLLM.Tokenizers            │
│  ├── GGUF loader (mmap)            ├── BPE (tiktoken-style)     │
│  ├── SafeTensors loader            ├── SentencePiece            │
│  ├── LlamaModel                    ├── HuggingFace tokenizer    │
│  ├── MistralModel                  └── Chat template engine     │
│  ├── PhiModel, QwenModel                                        │
│  └── DeepSeekModel (MLA)                                        │
├─────────────────────────────────────────────────────────────────┤
│  DotLLM.Diagnostics              DotLLM.Telemetry               │
│  ├── Hook registry                ├── IInferenceMetrics          │
│  ├── Activation capture           └── IRequestTracer             │
│  ├── Logit lens                                                  │
│  └── SAE integration                                             │
├─────────────────────────────────────────────────────────────────┤
│  DotLLM.Core (Interfaces & Abstractions)                        │
│  ├── ITensor, TensorShape, DType                                │
│  ├── IBackend, IKernelRunner, DevicePlacement                   │
│  ├── IAttentionMechanism, IAttentionStrategy                    │
│  ├── IPositionEncoding                                           │
│  ├── ISamplerStep, ILogitProcessor, IStopCondition              │
│  ├── IDecodingConstraint, TokenMask                             │
│  ├── IInferenceHook, HookPoint                                  │
│  └── ModelConfig, InferenceOptions                              │
├─────────────────────────────────────────────────────────────────┤
│  DotLLM.Cpu              │  DotLLM.Cuda                         │
│  ├── CpuBackend          │  ├── CudaBackend                     │
│  ├── SIMD kernels        │  ├── P/Invoke interop                │
│  └── TensorPrimitives    │  └── Handle management               │
├──────────────────────────┼──────────────────────────────────────┤
│  P/Invoke boundary       │                                      │
├──────────────────────────┼──────────────────────────────────────┤
│  Native C/CUDA Library   │                                      │
│  ├── cuBLAS GEMM         │  ├── Flash attention .cu             │
│  ├── Quantized matmul    │  ├── Fused RoPE/RMSNorm/SiLU .cu    │
│  ├── NCCL wrappers       │  └── GPU memory pool                 │
└──────────────────────────┴──────────────────────────────────────┘
```

## Data Flow: Model Loading

```
GGUF file on disk
  │
  ├─ Header parsing ──→ magic, version, tensor count, metadata count
  ├─ Metadata parsing ──→ ModelConfig (architecture, dims, vocab, RoPE params)
  │                       ChatTemplate (Jinja2 string)
  │                       Tokenizer vocabulary + merges + scores
  │
  └─ Tensor data section
       │
       MemoryMappedFile.CreateFromFile()
       │
       ├─ Tensor descriptors ──→ (name, shape, quantization type, offset)
       │
       └─ Memory-mapped region ──→ OS demand-pages from disk
            │                       No managed heap allocation
            │
            ├─ CPU tensors: raw pointer via SafeMemoryMappedViewHandle
            └─ GPU tensors: cudaMemcpy from mmap'd host → device memory
```

## Data Flow: Inference Request

```
HTTP POST /v1/chat/completions
  │
  ├─ Parse request (messages, tools, sampling params, constraints)
  ├─ Apply chat template ──→ IChatTemplate.Apply(messages) ──→ prompt string
  ├─ Tokenize ──→ ITokenizer.Encode(prompt) ──→ int[] token_ids
  ├─ Prefix cache lookup ──→ match existing KV-cache blocks
  ├─ Enqueue in scheduler with priority
  │
  └─ Scheduler admits request when KV-cache capacity available
       │
       ├─ PREFILL (compute-bound)
       │    For each layer:
       │      Norm → Q/K/V projection → RoPE → Attention → Residual
       │      → Norm → FFN (+LoRA delta) → Residual
       │      [Hooks fire at each stage if registered]
       │    Store K, V in KV-cache blocks
       │
       ├─ DECODE LOOP (memory-bandwidth-bound)
       │    Each iteration:
       │      Forward pass for single token (using cached K, V)
       │      → Sampler pipeline: logit_bias → constraint → penalties
       │        → temperature → top_k → top_p → min_p → sample
       │      → Check stop conditions
       │      → Advance constraint FSM
       │      → Yield token via SSE stream
       │
       └─ Response: tokens + usage + finish_reason
```

## Data Flow: Speculative Decoding

```
Draft model generates K candidates → Target model verifies in single forward pass
→ Accept left-to-right via rejection sampling → Rollback rejected tokens
  (KV-cache entries + constraint state rolled back)
```

See [SPECULATIVE.md](SPECULATIVE.md) for full design.

## NuGet Package Graph

```
DotLLM (pure .NET) ─── DotLLM.Server (ASP.NET)
├── Core, Models, Tokenizers, Cpu, Engine, Diagnostics, Telemetry

DotLLM.Backend.Cuda12 (native binaries) ── depends on DotLLM.Core
DotLLM.Backend.ROCm (future) ── depends on DotLLM.Core
```

## Threading Model

- **Server**: ASP.NET thread pool, fully async.
- **Scheduler**: Single dedicated thread, communicates via `Channel<T>`.
- **Inference**: Synchronous compute on scheduler thread. GPU ops async (kernel launch + stream sync).
- **Hooks**: Synchronous on inference thread — must be fast.
- **Streaming**: Tokens pushed via `Channel<T>` to `IAsyncEnumerable<string>`.