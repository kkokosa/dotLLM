# Implementation Roadmap — dotLLM

## Overview

The roadmap is organized into 6 phases, each building on the previous. The principle is **end-to-end first, then optimize** — get a working inference pipeline as quickly as possible (Phase 1), then add features and performance iteratively.

Each step is designed to be a discrete unit of work suitable for a single implementation session.

## Phase 1 — End-to-End Single-Token Generation

**Goal**: Load a model, tokenize a prompt, run a forward pass, generate one token. The "Hello World" of LLM inference.

| Step | Feature | Description | Key Files | Depends On |
|------|---------|-------------|-----------|------------|
| 1 | **GGUF loader** :white_check_mark: | Parse header, metadata KV pairs, tensor descriptors. Memory-map tensor data section via `MemoryMappedFile`. | `Models/Gguf/GgufReader.cs`, `GgufMetadata.cs`, `GgufTensorDescriptor.cs` | — |
| 2 | **FP16/Q8_0 dequantization** :white_check_mark: | Dequantize FP16 (trivial: half→float) and Q8_0 (scale × int8). Validates tensor data access through mmap. | `Cpu/Kernels/Dequantize.cs` | 1 |
| 3 | **Basic CPU tensor ops** :white_check_mark: | MatMul (f32 GEMV for single-token decode, then quantized GEMV operating directly on Q8_0/Q4_K blocks — no dequantization to f32, fused scale×int dot-product into accumulator, as llama.cpp does), RMSNorm, SiLU, Softmax. Use `TensorPrimitives` + SIMD intrinsics. Scalar reference implementations for correctness validation. | `Cpu/Kernels/MatMul.cs`, `RmsNorm.cs`, `SiLu.cs`, `Softmax.cs` | — |
| 4 | **BPE tokenizer** :white_check_mark: | Parse vocabulary and merges from GGUF metadata (`tokenizer.ggml.tokens`, `tokenizer.ggml.scores`). Trie-based encode, simple decode. | `Tokenizers/Bpe/BpeTokenizer.cs`, `Tokenizers/Trie.cs` | 1 |
| 5 | **GQA attention + RoPE** :white_check_mark: | Grouped-query attention with pre-computed cos/sin tables. Single implementation covering MHA/MQA/GQA via `num_kv_heads`. | `Cpu/Kernels/Attention.cs`, `Cpu/Kernels/RoPE.cs` | 3 |
| 6 | **Llama forward pass** :white_check_mark: | Wire together: embedding lookup → N × (RMSNorm → attention → residual → RMSNorm → FFN → residual) → RMSNorm → LM head. Generate logits for one token. | `Models/Architectures/LlamaModel.cs` | 1–5 |
| 7 | **Simple KV-cache** :white_check_mark: | Pre-allocated fixed-size KV-cache. Store K and V after each attention layer. On subsequent tokens, concatenate with cached K/V. | `Engine/KvCache/SimpleKvCache.cs` | 5, 6 |
| 8 | **Sampling pipeline** :white_check_mark: | Composable `ISamplerStep` chain: repetition penalty → temperature → top-k → top-p → min-p → categorical sample. Greedy (argmax) as special case of temperature=0. | `Engine/Samplers/` | 6 |
| 9 | **Stop conditions** :white_check_mark: | EOS token, max tokens, stop strings. `IStopCondition` interface. | `Engine/Samplers/StopConditions/` | 8 |

**Milestone**: Run `dotnet run -- --model llama-3-8b.Q8_0.gguf --prompt "Hello"` and see coherent multi-token output.

**Recommended test model**: TinyLlama 1.1B (Q8_0) — small enough for CPU, uses standard Llama architecture.

## Phase 2 — Practical Local Inference

**Goal**: Support the most popular quantization format, add streaming, chat templates, diagnostics, and additional model architectures. This is the "usable for local experimentation" milestone.

| Step | Feature | Description | Depends On |
|------|---------|-------------|------------|
| 10 | **SIMD kernel tuning** :white_check_mark: | Benchmark-driven optimization of Q8_0/Q4_K GEMV kernels. Fused dequant-dot (no intermediate f32 buffer), `Fma.MultiplyAdd` accumulation, AVX-512 specialization with AVX2 fallback. Scalar reference as correctness oracle. Target: ~2-4× over current functional kernels. | 3 |
| 11 | **CPU batched GEMM** :white_check_mark: | Batched matrix-matrix multiply for prefill: process all prompt tokens in one GEMM call instead of per-token GEMV. Tiled loop with cache-friendly access patterns. Falls back to GEMV for single-token decode. Target: ~5-10× prefill speedup. | 3 |
| 12 | **Engine inference timings** | Move timing into the engine: `InferenceTimings` record (`PrefillTimeMs`, `DecodeTimeMs`, `SamplingTimeMs`, prefill/decode token counts, derived tok/s). Add `Timings` to `InferenceResponse`. Instrument `TextGenerator.Generate()` with `Stopwatch.GetTimestamp()`. Refactor CLI to consume engine timings. Prerequisite for Step 43 (metrics) and Step 13 (benchmarks). | 6 |
| 13 | **BenchmarkDotNet inference benchmarks** | End-to-end inference benchmarks via BDN: SmolLM-135M, Llama-3.2-1B, Llama-3.2-3B (Q8_0). Custom `IColumn` for prefill tok/s and decode tok/s from `InferenceTimings`. Models auto-downloaded via `HuggingFaceDownloader`. Optional llama.cpp comparison via separate script. | 12 |
| 14 | **Q4_K_M dequantization** | K-quant with super-blocks, double quantization. See `docs/QUANTIZATION.md` for block layout. | 2 |
| 15 | **Mixed quantization** | Handle heterogeneous per-tensor quantization types (common in Q4_K_M files: attention Q6_K, FFN Q4_K). Dispatch correct kernel per tensor. | 14 |
| 16 | **Chat template engine** | Jinja2-subset interpreter. Parse `chat_template` from GGUF metadata or `tokenizer_config.json`. Compile to `IChatTemplate`. | 4 |
| 17 | **Streaming generation** | `IAsyncEnumerable<string>` token-by-token output. Yield each decoded token as it's generated. | 8 |
| 18 | **Hook system** | `IInferenceHook` interface, `HookPoint` enum, hook registry on `InferenceEngine`. Fire at 8 pipeline locations. Zero-cost when no hooks registered. | 6 |
| 19 | **Logit lens** | Built on hook system. Capture `PostLayer(i)` hidden states, project through LM head, produce per-layer token probabilities. | 18 |
| 20 | **Additional architectures** | Mistral (add sliding window attention mask), Phi, Qwen. Should be mostly `ModelConfig` parameterization, minimal new code. | 6 |
| 21 | **Logit bias** | Per-request `logit_bias` map applied as `ISamplerStep` at the start of the sampling pipeline. | 8 |
| 22 | **Multi-threaded CPU inference** :white_check_mark: | Parallelize GEMV/GEMM, attention, and FFN across cores. Custom zero-alloc `ComputeThreadPool` with `delegate*` dispatch for compute-bound loops in `MatMul`, `Attention`, per-layer token processing. Thread count configurable via `--threads` CLI option and `ThreadingConfig`. Target: ~4-8× speedup on multi-core CPUs. | 6 |

**Milestone**: Chat interactively with Q4_K_M models, stream responses, inspect layer activations via logit lens.

## Phase 2B — CPU Performance Optimizations

**Goal**: Close the performance gap with llama.cpp and exceed it on prefill. Research (ik_llama.cpp, llamafile/tinyBLAS, QuAKE, IntAttention, T-MAC) shows llama.cpp leaves 2-7× prefill performance on the table. These steps implement proven CPU optimization techniques.

| Step | Feature | Description | Depends On |
|------|---------|-------------|------------|
| 23 | **Row-interleaved weight repacking** | At model load time, repack quantized weight matrices from row-major into row-interleaved format (e.g., `Q8_0_R8`: 8 rows' blocks packed contiguously). SIMD lanes processing multiple output rows access contiguous memory instead of striding. Repacked buffer via `NativeMemory.AlignedAlloc` alongside mmap'd weights. Key files: `LlamaWeights.cs`, new `Cpu/Kernels/WeightRepacking.cs`. | 14 |
| 24 | **Outer-product tiled matmul kernels** | Replace inner-product GEMM with outer-product formulation: unroll M×N output tile, share one activation load across all weight-row dot products. AVX2: 3×4 tile (12 ymm accumulators). AVX-512: 5×6 or 6×8 (32 zmm). Requires row-interleaved weights (23). Prefill only — decode uses existing GEMV. **Single biggest win — 3-7× prefill speedup.** Key files: `Cpu/Kernels/MatMul.cs` (new `OuterProductGemmQ8_0`). | 23 |
| 25 | **Tiled attention (flash-attention style)** | Replace full QK^T materialization with tiled algorithm: iterate over KV tiles sized to fit in L2, compute partial scores, maintain running max/sum-of-exp for online softmax, accumulate weighted V. Fuses softmax + value weighting. seqQ=1 skips tiling. Eliminates O(n²) score allocation: 4096 context drops from 64MB to ~32KB. Key files: `Cpu/Kernels/Attention.cs` (new `ExecuteTiled`). | 22 |
| 26 | **Fast approximate exp/softmax** | IEEE-754 bit-manipulation fast exp (QuAKE-style). AVX2/AVX-512 vectorized. Use in attention softmax where full precision is unnecessary. Keep standard path for sampling softmax. **10-35% total inference speedup** (QuAKE paper). Key files: new `Cpu/Kernels/FastMath.cs`, `Cpu/Kernels/Attention.cs`. | — |
| 27 | **Operator fusion (FFN + attention pipeline)** | Fuse adjacent ops to eliminate intermediate DRAM roundtrips: (1) RMSNorm + projection, (2) FFN SwiGLU — gate+up+SiLU+mul+down in single tiled pass, (3) QKV+RoPE while results still in registers. **15-25% decode throughput improvement** from eliminated memory passes. Key files: new `Cpu/Kernels/FusedOps.cs`, `LlamaModel.cs`. | 24 |
| 28 | **NUMA-aware threading and CPU pinning** | Extend `ComputeThreadPool`: detect NUMA topology, pin workers to P-cores on hybrid CPUs, replicate weights per NUMA node on multi-socket, auto-reduce decode threads to memory channel count. **62% speedup on multi-socket.** Key files: `Cpu/Threading/ComputeThreadPool.cs`, new `Cpu/Threading/NumaTopology.cs`. | 22 |

**Dependency graph:**
```
Step 14 (Q4_K_M) ───► Step 23 (Repacking) ───► Step 24 (Outer-Product) ───► Step 27 (Fusion)
Step 22 (done) ──────► Step 25 (Tiled Attention)
(independent) ──────► Step 26 (Fast Exp)
Step 22 (done) ──────► Step 28 (NUMA)
```

**Milestone**: Prefill throughput exceeds llama.cpp on equivalent hardware. Outer-product GEMM reaches >800 GFLOPS on AVX2.

## Phase 3 — GPU Acceleration

**Goal**: GPU inference for dramatically higher throughput. Target: 10-50× speedup over CPU for prefill, 3-10× for decode.

| Step | Feature | Description | Depends On |
|------|---------|-------------|------------|
| 29 | **CUDA backend** | Native C/CUDA library: cuBLAS GEMM for matmul, custom kernels for flash attention, RMSNorm+SiLU fused, RoPE, quantized matmul (Q4_K_M, Q8_0). `CudaBackend` implementing `IBackend`. | Phase 1–2 |
| 30 | **CPU/GPU hybrid** | Layer offloading: specify N layers on GPU, remainder on CPU. Automatic tensor transfer at layer boundaries. Useful when model doesn't fully fit in VRAM. | 29 |
| 31 | **KV-cache quantization** | FP8 (E4M3) and INT8 KV-cache compression. Configurable per-model via `KvCacheConfig`. Extends effective context length. | 29 |

**Milestone**: Run Llama 3 8B at >50 tokens/sec decode on a single consumer GPU.

## Phase 4 — Production Serving

**Goal**: A production-ready API server with OpenAI compatibility, concurrent request handling, and structured output. This is the "deploy behind an API" milestone.

| Step | Feature | Description | Depends On |
|------|---------|-------------|------------|
| 32 | **ASP.NET server** | Minimal API endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/v1/tokenize`, `/v1/detokenize`. Health + readiness probes. | 16, 17 |
| 33 | **Continuous batching** | `IScheduler` with iteration-level scheduling. Prefill/decode separation. Request admission based on KV-cache capacity. Sequence eviction on completion. | 7, 32 |
| 34 | **Paged KV-cache** | PagedAttention: block-based allocation, block tables, free pool, reference counting, copy-on-write. Replace simple KV-cache. | 33 |
| 35 | **Prompt caching** | Automatic prefix sharing via trie of computed KV blocks. Reference-counted shared blocks. LRU eviction. Optional explicit `prefix_id` API. | 34 |
| 36 | **Rate limiting** | Per-API-key token-bucket rate limiter via `System.Threading.RateLimiting`. Requests/min, tokens/min, concurrent limits. Priority levels. HTTP 429 responses. | 32 |
| 37 | **JSON mode** | `JsonConstraint` — FSM-based constrained decoding guaranteeing syntactically valid JSON. `response_format: {"type": "json_object"}`. | 8 |
| 38 | **JSON Schema** | `JsonSchemaConstraint` — Schema-compiled automaton. `response_format: {"type": "json_schema", ...}`. Token mask precomputation. | 37 |
| 39 | **Regex + CFG** | `RegexConstraint` (DFA-based) and `GrammarConstraint` (PDA, GBNF-style). | 37 |
| 40 | **Tool calling** | `IToolCallParser`, chat template tool integration, structured output for function arguments. `finish_reason: "tool_calls"`. Parallel tool calls. | 16, 38 |
| 41 | **Speculative decoding** | `ISpeculativeDecoder`. Draft-verify-accept loop with modified rejection sampling. KV-cache rollback. Constraint state rollback via `IDecodingConstraint.Clone()`. | 33, 34 |
| 42 | **Beam search** | N-best decoding with length normalization. COW KV-cache for beam prefix sharing. Per-beam constraint state. | 34 |
| 43 | **Metrics & tracing** | `System.Diagnostics.Metrics` for throughput/latency/utilization. `System.Diagnostics.Activity` for per-request tracing. OpenTelemetry exporters. | 12, 33 |
| 44 | **Warm-up** | JIT pre-compilation pass at startup. CUDA kernel pre-loading. Configurable `WarmupOptions`. Readiness probe gates on warm-up completion. | 32 |

**Milestone**: Serve concurrent API requests with structured output, tool calling, continuous batching, and full observability.

## Phase 5 — Expand

**Goal**: Advanced features for specialized use cases — multi-tenant serving, additional model architectures, multi-GPU, interpretability research tooling.

| Step | Feature | Description | Depends On |
|------|---------|-------------|------------|
| 45 | **LoRA adapters** | `IAdapterManager`. Runtime adapter loading from SafeTensors. Multi-adapter batching (group sequences by adapter). Per-request `lora_adapter` parameter. No weight merging. | 33 |
| 46 | **MLA attention** | DeepSeek-V2/V3 Multi-head Latent Attention. Down-project KV to latent, up-project during attention. `LatentKvCache`. | Phase 1 |
| 47 | **ALiBi position encoding** | Additive linear bias to attention scores. `AlibiPositionEncoding` implementing `IPositionEncoding`. | Phase 1 |
| 48 | **SAE integration** | Sparse autoencoder hooks. Load pre-trained SAEs from SafeTensors. Feature analysis, steering, ablation. Sample project: `DotLLM.Sample.Interpretability`. | 18 |
| 49 | **Multi-GPU tensor parallelism** | NCCL-based TP. Split attention heads and FFN columns. All-reduce after attention and FFN. `ParallelismConfig`. See `docs/MULTI_GPU.md`. | 29 |
| 50 | **ROCm backend** | HIP conditional compilation of CUDA kernels. `#ifdef __HIP_PLATFORM_AMD__`. Separate `DotLLM.Backend.ROCm` NuGet package. Same C# code, different native binary. | 29 |

**Milestone**: Multi-tenant LoRA serving, DeepSeek support, multi-GPU 70B inference, mechanistic interpretability workflows in .NET.

## Future Considerations

Not in the current roadmap, but the architecture should not preclude these:

| Feature | Description | Architectural Impact |
|---------|-------------|---------------------|
| **Runtime quantization** | Load FP16 model and quantize to Q4_K_M at load time | Add `IQuantizer` interface, quantization kernels |
| **Vision / multimodal** | Image encoders (CLIP ViT) for LLaVA, Phi-3-Vision, Qwen-VL | `IInputEncoder` abstraction mapping raw inputs → embeddings. Model arch needs to handle image token insertion. |
| **Guided generation** | Pause mid-stream, inject tokens (tool results), resume from arbitrary KV-cache state | KV-cache append API, generation continuation from checkpoint |
| **Model merging** | SLERP/TIES/DARE weight arithmetic | Utility feature, not core inference. Operates on loaded weight tensors. |
| **Pipeline warm-up profiling** | Auto-tune batch sizes, memory allocation based on warm-up profiling runs | Profiler that measures throughput at various batch sizes, selects optimal |
| **Pipeline parallelism** | Split layers across nodes for very large models (405B+) | Micro-batching scheduler, point-to-point communication via NCCL send/recv |
| **T-MAC LUT-based matmul** | `vpshufb` table lookup for 1-4 bit weights. 4× throughput for ultra-low-bit. | New quant types, LUT-compatible repacking. |
| **HNSW vocabulary projection** | ANN search replacing LM head GEMV. ~40 candidates from 128K vocab. | `IVocabProjection` interface, HNSW index at load time. |
| **JIT-specialized kernel codegen** | Source generators for format-specific kernels (QIGen-style). | `IKernelGenerator`, Roslyn source generator pipeline. |

## Version Milestones

| Version | Phase | Description |
|---------|-------|-------------|
| `v0.1.0` | Phase 1 complete | First token: CPU inference with Q8_0 Llama models |
| `v0.2.0` | Phase 2 complete | Local inference: Q4_K_M, chat, streaming, multiple architectures |
| `v0.2.5` | Phase 2B complete | CPU performance: outer-product GEMM, tiled attention, operator fusion, NUMA |
| `v0.3.0` | Phase 3 complete | GPU acceleration: CUDA backend, hybrid CPU/GPU |
| `v0.4.0` | Phase 4 complete | Production server: OpenAI API, batching, structured output, tools |
| `v0.5.0` | Phase 5 complete | Extended: LoRA, multi-GPU, SAE, ROCm |
| `v1.0.0` | Stability | API stability commitment, comprehensive benchmarks, documentation |

## Testing Checkpoints

Each phase has a validation checkpoint:

- **Phase 1**: Generate coherent text from TinyLlama 1.1B Q8_0 on CPU. Numerical accuracy within ε of llama.cpp output for the same prompt.
- **Phase 2**: Interactive chat with Llama 3 8B Q4_K_M. Logit lens produces meaningful layer-wise predictions. Mistral/Phi/Qwen models produce correct output.
- **Phase 2B**: Outer-product GEMM reaches >800 GFLOPS on AVX2. Prefill throughput exceeds llama.cpp on equivalent hardware. Tiled attention handles 4096+ context without O(n²) memory. All kernels pass numerical accuracy validation against scalar reference.
- **Phase 3**: GPU decode throughput within 2× of llama.cpp for equivalent model/quantization. Hybrid CPU/GPU matches pure-CPU quality.
- **Phase 4**: Pass OpenAI API compatibility test suite. JSON schema constraint produces 100% valid outputs over 1000 generations. Continuous batching maintains throughput under concurrent load.
- **Phase 5**: Multi-adapter serving handles mixed-adapter batches. TP=2 produces identical output to TP=1. SAE feature steering demonstrably modifies model behavior.
