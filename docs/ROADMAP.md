# Implementation Roadmap — dotLLM

## Overview

The roadmap is organized into 9 phases, each building on the previous. The principle is **end-to-end first, then optimize** — get a working inference pipeline as quickly as possible (Phase 1), then add features and performance iteratively.

Each step is designed to be a discrete unit of work suitable for a single implementation session.

> **Note:** Step numbers are not sequential within phases. The original numbering is preserved for traceability (issues, branches, commits reference step numbers), but steps have been reordered and moved between phases as the roadmap evolved.

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

**Goal**: Support the most popular quantization format, add streaming, chat templates, multi-threading, and additional model architectures. This is the "usable for local experimentation" milestone.

| Step | Feature | Description | Depends On |
|------|---------|-------------|------------|
| 10 | **SIMD kernel tuning** :white_check_mark: | Benchmark-driven optimization of Q8_0/Q4_K GEMV kernels. Fused dequant-dot (no intermediate f32 buffer), `Fma.MultiplyAdd` accumulation, AVX-512 specialization with AVX2 fallback. Scalar reference as correctness oracle. Target: ~2-4× over current functional kernels. | 3 |
| 11 | **CPU batched GEMM** :white_check_mark: | Batched matrix-matrix multiply for prefill: process all prompt tokens in one GEMM call instead of per-token GEMV. Tiled loop with cache-friendly access patterns. Falls back to GEMV for single-token decode. Target: ~5-10× prefill speedup. | 3 |
| 12 | **Engine inference timings** :white_check_mark: | Move timing into the engine: `InferenceTimings` record (`PrefillTimeMs`, `DecodeTimeMs`, `SamplingTimeMs`, prefill/decode token counts, derived tok/s). Add `Timings` to `InferenceResponse`. Instrument `TextGenerator.Generate()` with `Stopwatch.GetTimestamp()`. Refactor CLI to consume engine timings. Prerequisite for Step 45 (metrics) and Step 13 (benchmarks). | 6 |
| 13 | **BenchmarkDotNet inference benchmarks** :white_check_mark: | End-to-end inference benchmarks via BDN: SmolLM-135M, Llama-3.2-1B, Llama-3.2-3B (Q8_0). Custom `IColumn` for prefill tok/s and decode tok/s from `InferenceTimings`. Models auto-downloaded via `HuggingFaceDownloader`. Optional llama.cpp comparison via separate script. | 12 |
| 14 | **Q4_K_M dequantization** :white_check_mark: | K-quant with super-blocks, double quantization. See `docs/QUANTIZATION.md` for block layout. | 2 |
| 15 | **Mixed quantization + Q8_K** :white_check_mark: | Handle heterogeneous per-tensor quantization types (common in Q4_K_M files: attention Q6_K, FFN Q4_K). Implement Q8_K input quantization (float32 scale, 256-element super-blocks) for K-quant fused vec_dot kernels. Re-enable Q4_K×Q8_K, Q5_K×Q8_K, Q6_K×Q8_K fused GEMV/GEMM paths. True 4-row kernels with shared activation loading. | 14 |
| 16 | **Chat template engine** :white_check_mark: | Jinja2-subset interpreter. Parse `chat_template` from GGUF metadata or `tokenizer_config.json`. Compile to `IChatTemplate`. | 4 |
| 17 | **Streaming generation** :white_check_mark: | `IAsyncEnumerable<string>` token-by-token output. Yield each decoded token as it's generated. | 8 |
| 20 | **Additional architectures** :white_check_mark: | Mistral (add sliding window attention mask), Phi, Qwen. Should be mostly `ModelConfig` parameterization, minimal new code. | 6 |
| 22 | **Multi-threaded CPU inference** :white_check_mark: | Parallelize GEMV/GEMM, attention, and FFN across cores. Custom zero-alloc `ComputeThreadPool` with `delegate*` dispatch for compute-bound loops in `MatMul`, `Attention`, per-layer token processing. Thread count configurable via `--threads` CLI option and `ThreadingConfig`. Target: ~4-8× speedup on multi-core CPUs. | 6 |

**Milestone**: Chat interactively with Q4_K_M models, stream responses, support multiple model architectures.

## Phase 3 — CPU Performance

**Goal**: Close the performance gap with llama.cpp and exceed it on prefill. Research (ik_llama.cpp, llamafile/tinyBLAS, QuAKE, IntAttention, T-MAC) shows llama.cpp leaves 2-7× prefill performance on the table. These steps implement proven CPU optimization techniques.

| Step | Feature | Description | Depends On |
|------|---------|-------------|------------|
| 23 | **Decode dispatch optimization** :white_check_mark: | Reduce per-token thread pool overhead during decode. Fuse Q/K/V projections into single dispatch (3→1), fuse Gate/Up (2→1), saving ~120 dispatches per token (~4% decode throughput). Enable GEMV pre-quantization reuse (`QuantizeInput` for seqLen=1) to avoid redundant input quantization across projections sharing the same input. Key files: `TransformerModel.cs`, `MatMulFusedDecode.cs`. | 22 |
| 24 | **Q8_1 input format + Q5_0 kernel tuning** :white_check_mark: | New Q8_1 input quantization storing precomputed `d * sum(qs)` per block. Eliminates q8sum computation from Q5_0/Q4_0/Q4_1 vec_dot kernels (~4 fewer SIMD ops/block, ~22% of Q5_0 per-block cost). 2-block loop unrolling in Q5_0 AVX2 kernel for better instruction-level parallelism. Goal: make dotLLM bandwidth-bound during decode so Q4_K_M becomes faster than Q8_0 (as in llama.cpp). Key files: `MatMulQ5_0.cs`, new Q8_1 quantization in `MatMul.cs`. | 15 |
| 25 | **Row-interleaved weight repacking** :white_check_mark: | At model load time, repack quantized weight matrices from row-major into R4-interleaved format (groups of 4 rows' blocks packed contiguously column-by-column). 4-row SIMD kernels read sequentially instead of striding across rows, improving cache/TLB locality. R4-aware VecDot variants for Q8_0, Q5_0, Q4_K, Q5_K, Q6_K. Key files: `Cpu/Kernels/WeightRepacking.cs`, `TransformerWeights.cs`, `TransformerModel.cs`. | 14, 22, 23, 24 |
| 26 | **Outer-product tiled matmul kernels** | Replace inner-product GEMM with outer-product formulation: unroll M×N output tile, share one activation load across all weight-row dot products. AVX2: 4×3 tile (12 ymm accumulators). AVX-512: 4×6 tile (24 zmm accumulators). Requires row-interleaved weights (25). Prefill only — decode uses existing GEMV. **Blocked on RyuJIT register pressure** — 4×3 tile needs 23 YMM registers (only 16 available), causing spills that negate the weight-reuse benefit. May require AVX-512 (32 ZMM) or native C microkernel. Key files: `Cpu/Kernels/MatMul.cs` (new `OuterProductGemmQ8_0`), `MatMulQ5_0.cs` (`OuterProductGemmQ5_0`), `MatMulKQuants.cs` (K-quant outer-product dispatch). | 25 |
| 27 | **Tiled attention (flash-attention style)** :white_check_mark: | Replace full QK^T materialization with tiled algorithm: iterate over KV tiles sized to fit in L2, compute partial scores, maintain running max/sum-of-exp for online softmax, accumulate weighted V. Fuses softmax + value weighting. seqQ=1 skips tiling. Eliminates O(n²) score allocation: 4096 context drops from 64MB to ~32KB. Key files: `Cpu/Kernels/Attention.cs` (new `ExecuteTiled`). | 22 |
| 28 | **Fast approximate exp/softmax** :white_check_mark: | IEEE-754 bit-manipulation fast exp (QuAKE-style). AVX2/AVX-512 vectorized. Use in attention softmax where full precision is unnecessary. Keep standard path for sampling softmax. **10-35% total inference speedup** (QuAKE paper). Key files: new `Cpu/Kernels/FastMath.cs`, `Cpu/Kernels/Attention.cs`. | — |
| 29 | **Operator fusion (FFN + attention pipeline)** :white_check_mark: | Fuse adjacent ops to eliminate intermediate DRAM roundtrips: (1) RMSNorm + quantize fused into single pass (decode-only, skips normOut intermediate), (2) Tiled SwiGLU — sigmoid intermediate stays in L1 via 1KB stack buffer. Key files: new `Cpu/Kernels/FusedOps.cs`, `TransformerModel.cs`. | 26 |
| 30 | **NUMA-aware threading and CPU pinning** :white_check_mark: | Extend `ComputeThreadPool`: adaptive spin-wait dispatch (generation-counter approach with event fallback), NUMA topology detection (Windows `GetLogicalProcessorInformationEx`, Linux sysfs), P-core/E-core awareness on Intel hybrid CPUs, CPU affinity pinning (`SetThreadAffinityMask`/`sched_setaffinity`), auto-reduce decode threads to memory channel count. Key files: `Cpu/Threading/ComputeThreadPool.cs`, `Cpu/Threading/NumaTopology.cs`, `Cpu/Threading/CpuAffinity.cs`, `Cpu/Threading/DispatchMode.cs`. | 22 |

**Dependency graph:**
```
Step 22 (done) ──────► Step 23 (Dispatch Opt)
Step 15 (done) ──────► Step 24 (Q8_1 + Q5_0)
Step 14 (Q4_K_M) ───► Step 25 (Repacking) ───► Step 26 (Outer-Product) ───► Step 29 (Fusion)
Step 22 (done) ──────► Step 27 (Tiled Attention)
(independent) ──────► Step 28 (Fast Exp)
Step 22 (done) ──────► Step 30 (NUMA + Spin-wait)
```

**Milestone**: Prefill throughput exceeds llama.cpp on equivalent hardware. Outer-product GEMM reaches >800 GFLOPS on AVX2. Decode becomes bandwidth-bound: Q4_K_M faster than Q8_0.

## Phase 4 — GPU Acceleration

**Goal**: GPU inference for dramatically higher throughput. Target: 10-50× speedup over CPU for prefill, 3-10× for decode.

| Step | Feature | Description | Depends On |
|------|---------|-------------|------------|
| 31 | **CUDA backend** :white_check_mark: | PTX kernels loaded via CUDA Driver API P/Invoke — no native shared library. cuBLAS HGEMM for prefill, custom quantized GEMV for decode (Q8_0, Q4_K, Q6_K). Dequantization kernels (Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K). FP16 activation pipeline, on-the-fly weight dequantization into scratch buffer, GPU KV-cache. `CudaTransformerModel` implementing `IModel`. | Phase 1–2 |
| 32 | **CPU/GPU hybrid** :white_check_mark: | Layer offloading: specify N layers on GPU, remainder on CPU. Automatic tensor transfer at layer boundaries. Useful when model doesn't fully fit in VRAM. | 31 |
| 33 | **KV-cache quantization** :white_check_mark: | Q8_0 and Q4_0 KV-cache compression on both CPU and GPU. Separate key/value type configs (`--cache-type-k`, `--cache-type-v`). Mixed-precision window: recent W tokens in full precision, older tokens quantized. Quantize-on-write, dequant-in-attention-kernel. `KvCacheConfig { KeyDType, ValueDType, MixedPrecisionWindowSize }`. Extends effective context length 2–4×. | 31 |

**Milestone**: Run Llama 3 8B at >50 tokens/sec decode on a single consumer GPU.

## Phase 5 — Constrained Decoding & Initial API

**Goal**: Structured output guarantees, an OpenAI-compatible API server with built-in chat UI, and simple prompt caching for interactive use.

| Step | Feature | Description | Depends On |
|------|---------|-------------|------------|
| 39 | **JSON mode** :white_check_mark: | `JsonConstraint` — FSM-based constrained decoding guaranteeing syntactically valid JSON. `response_format: {"type": "json_object"}`. | 8 |
| 40 | **JSON Schema** :white_check_mark: | `JsonSchemaConstraint` — Schema-compiled automaton. `response_format: {"type": "json_schema", ...}`. Token mask precomputation. | 39 |
| 41 | **Regex + CFG** :white_check_mark: | `RegexConstraint` (DFA-based) and `GrammarConstraint` (PDA, GBNF-style). | 39 |
| 42 | **Tool calling** :white_check_mark: | `IToolCallParser`, chat template tool integration, structured output for function arguments. `finish_reason: "tool_calls"`. Parallel tool calls. | 16, 40 |
| 34 | **ASP.NET server** :white_check_mark: | Minimal API endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/v1/tokenize`, `/v1/detokenize`. Health + readiness probes. | 16, 17 |
| 53 | **Chat UI (`serve` command)** :white_check_mark: | Built-in web chat interface served by the ASP.NET host. `dotllm serve model.gguf` starts the API server and opens a browser to a bundled single-page chat UI. Streaming responses via SSE, model/parameter selection, conversation history. Similar to `ollama serve` + Open WebUI or `llama.cpp --server` built-in UI. | 34 |
| 54 | **Simple prompt caching** :white_check_mark: | Hash-based prefix cache for multi-turn conversations. Hash the token prefix, store/reuse the KV-cache state. On cache hit, skip prefill for the matching prefix and only process new tokens. Works with existing `SimpleKvCache` — no paged attention required. LRU eviction with configurable max cached sessions. `--prompt-cache` CLI flag. Dramatically reduces TTFT for multi-turn chat. | 7, 34 |

**Milestone**: Guaranteed valid JSON/schema output, tool calling, a functional OpenAI-compatible API server, browser-based chat UI, and fast multi-turn conversations via prompt caching.

## Phase 6 — Improved Serving

**Goal**: Performance and deployment improvements for single-user and small-scale serving. Faster startup, efficient memory management, and speculative decoding — all without requiring concurrent request infrastructure.

| Step | Feature | Description | Depends On |
|------|---------|-------------|------------|
| 46 | **Warm-up** :white_check_mark: | JIT pre-compilation pass at startup. CUDA kernel pre-loading. Configurable `WarmupOptions`. Readiness probe gates on warm-up completion. | 34 |
| 55 | **Native AOT (experimental)** :white_check_mark: | Trimming-safe deployment via .NET Native AOT. Audit `[DynamicallyAccessedMembers]` annotations, source-generated JSON serialization, replace reflection-based patterns. `rd.xml` for preserved types. Goal: single-file `dotllm` binary with instant startup (~50ms vs ~500ms JIT). Mark experimental — some features (runtime LoRA loading, source generators) may require JIT fallback. | 34 |
| 36 | **Paged KV-cache** | PagedAttention: block-based allocation, block tables, free pool, reference counting, copy-on-write. Replace simple KV-cache. | 7 |
| 37 | **Prompt caching (advanced)** | Automatic prefix sharing via trie of computed KV blocks. Reference-counted shared blocks. LRU eviction. Optional explicit `prefix_id` API. Upgrades simple prompt caching (step 54) to work with paged KV-cache. | 36 |
| 43 | **Speculative decoding** | `ISpeculativeDecoder`. Draft-verify-accept loop with modified rejection sampling. KV-cache rollback. Constraint state rollback via `IDecodingConstraint.Clone()`. | 36 |

**Milestone**: Sub-100ms startup via warm-up/AOT, paged KV-cache with advanced prompt caching, speculative decoding for 2-3× decode speedup.

## Phase 7 — Diagnostics & Interpretability

**Goal**: First-class diagnostic hooks, mechanistic interpretability tools, and adapter support — making dotLLM a platform for LLM research in .NET.

| Step | Feature | Description | Depends On |
|------|---------|-------------|------------|
| 18 | **Hook system** | `IInferenceHook` interface, `HookPoint` enum, hook registry on `InferenceEngine`. Fire at 8 pipeline locations. Zero-cost when no hooks registered. | 6 |
| 19 | **Logit lens** | Built on hook system. Capture `PostLayer(i)` hidden states, project through LM head, produce per-layer token probabilities. | 18 |
| 21 | **Logit bias** | Per-request `logit_bias` map applied as `ISamplerStep` at the start of the sampling pipeline. | 8 |
| 47 | **LoRA adapters** | `IAdapterManager`. Runtime adapter loading from SafeTensors. Per-request `lora_adapter` parameter. No weight merging. | Phase 1 |
| 50 | **SAE integration** | Sparse autoencoder hooks. Load pre-trained SAEs from SafeTensors. Feature analysis, steering, ablation. Sample project: `DotLLM.Sample.Interpretability`. | 18 |

**Milestone**: Diagnostic hooks, logit lens, logit bias, LoRA adapter serving, and mechanistic interpretability workflows in .NET.

## Phase 8 — Model Expansion

**Goal**: Broaden model architecture support beyond Llama/Mistral/Phi/Qwen to cover DeepSeek, SmolLM3, Gemma, and Mixture-of-Experts architectures.

| Step | Feature | Description | Depends On |
|------|---------|-------------|------------|
| 48 | **MLA attention** | DeepSeek-V2/V3 Multi-head Latent Attention. Down-project KV to latent, up-project during attention. `LatentKvCache`. | Phase 1 |
| 49 | **ALiBi position encoding** | Additive linear bias to attention scores. `AlibiPositionEncoding` implementing `IPositionEncoding`. | Phase 1 |
| 56 | **SmolLM3 architecture** | HuggingFace SmolLM3-3B. NoPE layer support in attention (skip RoPE application on marked layers). YARN context extension for 128k. GQA with 4 groups. Tool calling via `xml_tools` (Hermes-compatible) or `python_tools` (`PythonicToolCallParser`). | Phase 1 |
| 57 | **Gemma 4 architecture** | Google Gemma 4 model family. GeGLU activation, RMS pre-norm with per-layer scaling, interleaved local/global attention, logit soft-capping. `GemmaModel` implementing `IModel` via `TransformerBlock` parameterization. | Phase 1 |
| 58 | **Mixture of Experts** | MoE FFN with top-K expert routing. `IExpertRouter` interface, `MoeFFN` block replacing standard FFN. Sparse activation — only K of N experts compute per token. Shared expert support (DeepSeek-style). Memory: all expert weights loaded, only active experts computed. Covers: DeepSeek-V2 MoE, Granite hybrid MoE, Qwen-MoE. | Phase 1 |

**Milestone**: DeepSeek-V2/V3 inference, SmolLM3 with NoPE, Gemma 4, and MoE models running correctly.

## Phase 9 — Production Serving

**Goal**: Full production-grade serving infrastructure for concurrent multi-user deployments. Continuous batching, rate limiting, observability, and advanced scheduling.

| Step | Feature | Description | Depends On |
|------|---------|-------------|------------|
| 35 | **Continuous batching** | `IScheduler` with iteration-level scheduling. Prefill/decode separation. Request admission based on KV-cache capacity. Sequence eviction on completion. | 7, 34 |
| 59 | **Advanced scheduling** | Prefill/decode disaggregation — separate queues and thread pools for prefill-heavy vs decode-heavy workloads. Priority-based scheduling with preemption (swap lower-priority sequences to CPU when VRAM-constrained). Fairness constraints to prevent starvation. Chunked prefill for long prompts to avoid head-of-line blocking. | 35, 36 |
| 38 | **Rate limiting** | Per-API-key token-bucket rate limiter via `System.Threading.RateLimiting`. Requests/min, tokens/min, concurrent limits. Priority levels. HTTP 429 responses. | 34 |
| 45 | **Metrics & tracing** | `System.Diagnostics.Metrics` for throughput/latency/utilization. `System.Diagnostics.Activity` for per-request tracing. OpenTelemetry exporters. | 12, 35 |

**Milestone**: Serve concurrent API requests with continuous batching, advanced scheduling, rate limiting, and full OpenTelemetry observability.

## Future Considerations

Not in the current roadmap, but the architecture should not preclude these:

| Feature | Description | Architectural Impact |
|---------|-------------|---------------------|
| **Multi-GPU tensor parallelism** | NCCL-based TP. Split attention heads and FFN columns. All-reduce after attention and FFN. `ParallelismConfig`. See `docs/MULTI_GPU.md`. | Requires CUDA backend (31). |
| **ROCm backend** | HIP conditional compilation of CUDA kernels. `#ifdef __HIP_PLATFORM_AMD__`. Separate `DotLLM.Backend.ROCm` NuGet package. Same C# code, different native binary. | Requires CUDA backend (31). |
| **Beam search** | N-best decoding with length normalization. COW KV-cache for beam prefix sharing. Per-beam constraint state. | Requires paged KV-cache (36). |
| **Runtime quantization** | Load FP16 model and quantize to Q4_K_M at load time | Add `IQuantizer` interface, quantization kernels |
| **Vision / multimodal** | Image encoders (CLIP ViT) for LLaVA, Phi-3-Vision, Qwen-VL | `IInputEncoder` abstraction mapping raw inputs → embeddings. Model arch needs to handle image token insertion. |
| **Guided generation** | Pause mid-stream, inject tokens (tool results), resume from arbitrary KV-cache state | KV-cache append API, generation continuation from checkpoint |
| **Model merging** | SLERP/TIES/DARE weight arithmetic | Utility feature, not core inference. Operates on loaded weight tensors. |
| **Pipeline warm-up profiling** | Auto-tune batch sizes, memory allocation based on warm-up profiling runs | Profiler that measures throughput at various batch sizes, selects optimal |
| **Pipeline parallelism** | Split layers across nodes for very large models (405B+) | Micro-batching scheduler, point-to-point communication via NCCL send/recv |
| **T-MAC LUT-based matmul** | `vpshufb` table lookup for 1-4 bit weights. 4× throughput for ultra-low-bit. | New quant types, LUT-compatible repacking. |
| **HNSW vocabulary projection** | ANN search replacing LM head GEMV. ~40 candidates from 128K vocab. | `IVocabProjection` interface, HNSW index at load time. |
| **FP32 residual stream (CUDA)** | Keep residual/hidden buffers in FP32, activations in FP16. Currently the CUDA backend uses FP16 everywhere, which causes cumulative truncation in the residual stream. Measured: Qwen2.5-0.5B diverges at layer 1 (maxDiff=4.7 vs 0.3 for Llama), growing to maxDiff=14.9 by layer 24 — enough to flip the argmax token. Llama models tolerate FP16 residuals; Qwen/some architectures do not. llama.cpp uses FP32 residuals as standard practice. | `CudaForwardState`: Residual/HiddenState → FP32. New `fused_add_rmsnorm` variant: FP32 residual in/out, FP16 activation out. `LaunchRmsNorm` variant: FP32 input, FP16 output. Embedding → FP32 output. |
| **JIT-specialized kernel codegen** | Source generators for format-specific kernels (QIGen-style). | `IKernelGenerator`, Roslyn source generator pipeline. |
| **TurboQuant KV-cache** | Vector quantization for KV-cache using random orthogonal rotation + Lloyd-Max codebook (Google, ICLR 2026). 3-bit with near-zero quality loss, 5–6× compression. Calibration-free, data-oblivious. Fused attention kernel reads packed indices and gathers from codebook LUT. Enables 128K+ context on consumer GPUs. | New `TurboQuantKvCache`, rotation matrix storage, Lloyd-Max codebook (pre-computed for Beta distribution), fused attention PTX kernel with LUT gather, `--cache-type-k tq3_0` / `--cache-type-v tq2_0` CLI options. |
| **Chat template override** | `--chat-template <file\|preset>` CLI option to override the GGUF-embedded template. Enables tool calling for models whose GGUF template doesn't inject tools (e.g., Phi-4-mini), and custom formatting for fine-tuned models. Presets for common formats (chatml, llama3, mistral). | `IChatTemplate` selection logic in CLI/Server. Template preset registry. Fallback tool-injection into system prompt when template ignores `tools` variable. |
| **xLAM tool call format** | Salesforce xLAM-2-1b/3b-fc-r — best-performing sub-4B tool-calling models (65.74% BFCL v3). Output is a bare JSON array `[{"name": "...", "arguments": {...}}]` after the assistant turn, terminated by EOS. No XML wrapper, no prefix token. vLLM uses `--tool-call-parser=xlam` with custom `xlam_qwen.jinja` template. | New `XlamToolCallParser`. Custom Jinja template preset for xLAM system prompt format. The 1B model (53.97% BFCL) is the smallest competitive tool-caller. |
| **Pythonic tool call format** | Emerging convention: model generates Python-style function invocations (`get_weather(city="Copenhagen")`) instead of JSON. Used by SmolLM3 `python_tools` mode (wrapped in `<code>` tags), Gorilla OpenFunctions (raw Python calls), NexusRaven (`Call: func(args)`), and increasingly by Llama 4 and OLMo 3. Avoids JSON escaping issues, more natural for code-trained models. | New `PythonicToolCallParser` with Python AST-style parsing (regex for function name + kwargs). Supports `<code>` wrapper (SmolLM3) and bare invocation variants. |
| **Granite tool call format** | IBM Granite 4.0 Micro (3B dense, Apache 2.0) — enterprise-grade tool calling. Tools in `<tools></tools>` XML, calls as `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`, results via `<tool_response>`. Role markers use `<\|start_of_role\|>` tokens (not ChatML). Also Granite-4.0-H-Micro (3B hybrid Mamba-2) and Granite-4.0-H-Tiny (7B MoE, ~1B active). vLLM uses `--tool-call-parser=granite`. Granite-4.0-1b scores 54.8 on BFCL v3 (best in 1B class). | `GraniteToolCallParser` (close to Hermes but different role markers). Granite-specific Jinja template preset. Granite architecture may need Mamba-2/MoE support for hybrid variants. |

## Version Milestones

| Version | Phase | Description |
|---------|-------|-------------|
| `v0.1.0` | Phase 1 complete | First token: CPU inference with Q8_0 Llama models |
| `v0.2.0` | Phase 2 complete | Local inference: Q4_K_M, chat, streaming, multiple architectures |
| `v0.2.5` | Phase 3 complete | CPU performance: outer-product GEMM, tiled attention, operator fusion, NUMA |
| `v0.3.0` | Phase 4 complete | GPU acceleration: CUDA backend, hybrid CPU/GPU, KV-cache quantization |
| `v0.3.5` | Phase 5 complete | Constrained decoding: JSON/schema/regex, tool calling, API server, chat UI, simple prompt caching |
| `v0.4.0` | Phase 6 complete | Improved serving: warm-up, NativeAOT, paged KV-cache, advanced prompt caching, speculative decoding |
| `v0.5.0` | Phase 7 complete | Diagnostics: hooks, logit lens, LoRA, SAE |
| `v0.6.0` | Phase 8 complete | Model expansion: MLA, ALiBi, SmolLM3, Gemma 4, MoE |
| `v0.7.0` | Phase 9 complete | Production serving: continuous batching, scheduling, rate limiting, metrics |
| `v1.0.0` | Stability | API stability commitment, comprehensive benchmarks, documentation |

## Testing Checkpoints

Each phase has a validation checkpoint:

- **Phase 1**: Generate coherent text from TinyLlama 1.1B Q8_0 on CPU. Numerical accuracy within ε of llama.cpp output for the same prompt.
- **Phase 2**: Interactive chat with Llama 3 8B Q4_K_M. Mistral/Phi/Qwen models produce correct output.
- **Phase 3**: Outer-product GEMM reaches >800 GFLOPS on AVX2. Prefill throughput exceeds llama.cpp on equivalent hardware. Tiled attention handles 4096+ context without O(n²) memory. All kernels pass numerical accuracy validation against scalar reference. Decode is bandwidth-bound: Q4_K_M faster than Q8_0.
- **Phase 4**: GPU decode throughput within 2× of llama.cpp for equivalent model/quantization. Hybrid CPU/GPU matches pure-CPU quality. Q8_0 KV-cache produces identical output to FP16 baseline. Q4_0 KV-cache perplexity within +0.3 of baseline.
- **Phase 5**: JSON schema constraint produces 100% valid outputs over 1000 generations. Pass OpenAI API compatibility test suite. `dotllm serve` launches browser-based chat. Multi-turn TTFT drops >5× with prompt caching enabled.
- **Phase 6**: Warm-up eliminates first-request latency spike. Paged KV-cache handles long contexts without fragmentation. Speculative decoding achieves ≥1.5× decode speedup with acceptable draft model overhead.
- **Phase 7**: Logit lens produces meaningful layer-wise predictions. Logit bias modifies token probabilities correctly. LoRA adapter swaps complete in <100ms. SAE feature steering demonstrably modifies model behavior.
- **Phase 8**: DeepSeek-V2 MoE produces correct output. SmolLM3 NoPE layers handled correctly. Gemma 4 matches HuggingFace reference output. MoE expert routing matches reference implementation.
- **Phase 9**: Continuous batching maintains throughput under concurrent load. Paged KV-cache handles concurrent sequences without OOM. Advanced scheduling prevents starvation under mixed workloads.
