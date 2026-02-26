# CLAUDE.md — dotLLM Project Guide

## Project Identity

**dotLLM** is an open-source, high-performance LLM inference engine written natively in C#/.NET. It targets transformer-based models (Llama, Mistral, Phi, Qwen, DeepSeek families) with CPU (SIMD-optimized) and CUDA GPU backends. The project aims to be the first production-grade, pure-.NET inference engine — not a wrapper around llama.cpp or ONNX Runtime, but a ground-up implementation leveraging .NET's modern performance primitives.

A key differentiator beyond raw performance is **first-class support for inference diagnostics and interpretability** — activation hooks, hidden state capture, and sparse autoencoder integration — bringing HuggingFace-transformers-level research tooling to the .NET ecosystem.

- **Repository**: https://github.com/kkokosa/dotLLM
- **License**: GNU General Public License v3.0 (GPLv3)
- **Target framework**: .NET 10
- **Primary language**: C# (with a thin C/CUDA native library for GPU kernels)

## Architecture Overview

### Design Philosophy

1. **Native .NET first** — All orchestration, model loading, tokenization, sampling, scheduling, and CPU compute are pure C#. No wrapping of C++ inference engines.
2. **Unmanaged memory for tensors** — All tensor data (weights, KV-cache, activations) lives in unmanaged memory via `NativeMemory.AlignedAlloc` (64-byte alignment). Only small metadata objects live on the managed heap. Target: **zero GC allocations on the inference hot path**.
3. **Hybrid GPU architecture** — GPU compute uses a thin native C/CUDA shared library called via `[LibraryImport]` P/Invoke. GPU memory is represented as opaque `IntPtr` handles in C# — tensor data never crosses the P/Invoke boundary.
4. **Backend-pluggable** — An `IBackend` interface allows swapping CPU/CUDA/ROCm (future) without touching core inference logic. Each backend ships as a separate NuGet package.
5. **Model architectures are parameterized, not duplicated** — Transformer variants share a common block structure with pluggable attention mechanisms, position encodings, and activation functions. Do not duplicate code for new architectures — parameterize via `ModelConfig`.
6. **Interpretability as a first-class feature** — Hook points throughout the inference pipeline allow capturing activations, attention patterns, and hidden states for mechanistic interpretability research (sparse autoencoders, logit lens, probing classifiers).
7. **Multi-GPU aware from day one** — Core abstractions (`IBackend`, tensor placement, forward pass orchestration) must not preclude tensor parallelism or pipeline parallelism, even if initial implementation is single-device. Tensor placement is always explicit.

### High-Level Component Map

```
┌─────────────────────────────────────────────────────┐
│  C# Managed Code                                     │
│  ├── Models/        Model architectures & loaders    │
│  ├── Tokenizers/    BPE, SentencePiece, HF parsers   │
│  ├── Templates/     Chat template engine              │
│  ├── Engine/        Scheduler, KV-cache, batching     │
│  ├── Cpu/           SIMD-optimized CPU backend        │
│  ├── Cuda/          P/Invoke wrappers for GPU backend │
│  ├── Samplers/      Temperature, top-k/p, penalties   │
│  ├── Constraints/   Structured output (JSON, regex)   │
│  ├── Adapters/      LoRA adapter loading & application│
│  ├── Diagnostics/   Hooks, activation capture, SAE    │
│  ├── Telemetry/     Metrics, tracing, observability   │
│  └── Server/        ASP.NET OpenAI-compatible API     │
├─────────────────────────────────────────────────────┤
│  P/Invoke boundary (IntPtr handles, coarse-grained)  │
├─────────────────────────────────────────────────────┤
│  Native C/CUDA Library (native/)                     │
│  ├── cuBLAS GEMM wrappers                            │
│  ├── Flash attention kernels (.cu)                   │
│  ├── Quantized matmul kernels (.cu)                  │
│  ├── Fused kernels: RoPE, RMSNorm, SiLU (.cu)       │
│  ├── NCCL multi-GPU communication                    │
│  └── GPU memory pool (cudaMallocAsync)               │
└─────────────────────────────────────────────────────┘
```

### Solution Structure

```
dotLLM/
├── CLAUDE.md                          # This file
├── README.md
├── LICENSE                            # GPLv3
├── Directory.Build.props              # Shared build properties, .NET 10, LangVersion
├── Directory.Packages.props           # Central package management
├── dotllm.sln
├── src/
│   ├── DotLLM.Core/                   # Interfaces, abstractions, tensor types, config
│   │   ├── Tensors/                   # ITensor, TensorShape, DType, unmanaged tensor impl
│   │   ├── Backends/                  # IBackend, IKernelRunner, DevicePlacement interfaces
│   │   ├── Models/                    # IModel, IModelArchitecture, ModelConfig
│   │   ├── Attention/                 # IAttentionMechanism, IAttentionStrategy interfaces
│   │   ├── PositionEncoding/          # IPositionEncoding interface
│   │   ├── Sampling/                  # ISamplerStep, ILogitProcessor, IStopCondition interfaces
│   │   ├── Constraints/              # IDecodingConstraint, TokenMask interfaces
│   │   ├── Diagnostics/              # IInferenceHook, HookPoint, activation capture interfaces
│   │   ├── Telemetry/                # IInferenceMetrics, IRequestTracer interfaces
│   │   └── Configuration/             # InferenceOptions, QuantizationType enums
│   ├── DotLLM.Models/                 # Model loaders (GGUF, SafeTensors) & architectures
│   │   ├── Gguf/                      # GGUF parser, metadata, tensor descriptor
│   │   ├── SafeTensors/               # SafeTensors loader
│   │   ├── Adapters/                  # LoRA adapter loading, multi-adapter management
│   │   └── Architectures/             # LlamaModel, MistralModel, PhiModel, QwenModel, DeepSeekModel
│   ├── DotLLM.Tokenizers/             # Tokenizer implementations
│   │   ├── Bpe/                       # tiktoken-style BPE
│   │   ├── SentencePiece/             # SentencePiece BPE
│   │   ├── HuggingFace/               # tokenizer.json parser
│   │   └── Templates/                 # Chat template engine (IChatTemplate, Jinja2-subset)
│   ├── DotLLM.Cpu/                    # CPU backend with SIMD kernels
│   │   ├── Kernels/                   # MatMul, RmsNorm, Softmax, RoPE, SiLU, quantized ops
│   │   └── CpuBackend.cs
│   ├── DotLLM.Cuda/                   # CUDA backend (P/Invoke to native lib)
│   │   ├── Interop/                   # LibraryImport declarations, handle types
│   │   └── CudaBackend.cs
│   ├── DotLLM.Engine/                 # Inference orchestration
│   │   ├── KvCache/                   # KV-cache manager (simple → paged), KV-cache quantization
│   │   ├── Scheduler/                 # Continuous batch scheduler, priority, preemption
│   │   ├── Samplers/                  # Composable sampler pipeline, beam search
│   │   ├── Constraints/               # Structured output: JSON schema, regex, CFG constrained decoding
│   │   ├── Speculative/               # Speculative decoding engine
│   │   ├── PrefixCache/               # Prompt prefix sharing / automatic prefix caching
│   │   └── InferenceEngine.cs         # Main entry point
│   ├── DotLLM.Diagnostics/           # Interpretability & debugging tools
│   │   ├── Hooks/                     # Hook registry, activation interceptors
│   │   ├── Capture/                   # Hidden state & attention pattern capture
│   │   ├── LogitLens/                 # Project intermediate states through LM head
│   │   └── Sae/                       # Sparse autoencoder integration points
│   ├── DotLLM.Telemetry/             # Observability
│   │   ├── Metrics/                   # Token throughput, latency, cache utilization, GPU memory
│   │   └── Tracing/                   # Per-request pipeline tracing
│   └── DotLLM.Server/                 # ASP.NET Minimal API
│       ├── Endpoints/                 # /v1/chat/completions, /v1/completions, /v1/models,
│       │                              # /v1/embeddings, /v1/tokenize
│       ├── ToolCalling/               # Tool/function calling protocol handling
│       └── RateLimiting/              # Per-key rate limiting, request priority
├── native/                            # C/CUDA native library (CMake)
│   ├── CMakeLists.txt
│   ├── include/                       # Public C API header (dotllm_native.h)
│   ├── src/
│   │   ├── cuda/                      # CUDA kernels (.cu files)
│   │   ├── cublas/                    # cuBLAS GEMM wrappers
│   │   ├── nccl/                      # Multi-GPU communication wrappers
│   │   └── memory/                    # GPU memory pool
│   └── hip/                           # HIP/ROCm compat layer (future)
├── tests/
│   ├── DotLLM.Tests.Unit/
│   └── DotLLM.Tests.Integration/
├── benchmarks/
│   └── DotLLM.Benchmarks/            # BenchmarkDotNet performance tests
└── samples/
    ├── DotLLM.Sample.Console/         # Minimal console chat app
    ├── DotLLM.Sample.Server/          # Server deployment example
    ├── DotLLM.Sample.ToolCalling/     # Function calling / tool use example
    └── DotLLM.Sample.Interpretability/ # SAE / logit lens / activation analysis example
```

## Core Abstractions

### Attention Mechanisms — `IAttentionMechanism`

Attention is **not hardcoded to GQA**. The engine supports pluggable attention via `IAttentionMechanism`:

**Grouped-Query Attention (GQA)** — A single implementation that generalizes three variants through `num_kv_heads` configuration:
- `num_kv_heads == num_attention_heads` → **Multi-Head Attention (MHA)** (classic transformer, older models)
- `num_kv_heads == 1` → **Multi-Query Attention (MQA)** (Falcon, PaLM)
- `1 < num_kv_heads < num_attention_heads` → **Grouped-Query Attention (GQA)** (Llama 2/3, Mistral, Qwen2)

**Multi-head Latent Attention (MLA)** — Structurally distinct from GQA. Used by DeepSeek-V2/V3. Compresses KV projections into a low-rank latent space via down-projection, stores the compressed latent in the KV-cache (dramatically reducing cache size), then decompresses via up-projection during attention computation. Requires its own `IAttentionMechanism` implementation with separate `LatentKvCache` management.

**Sliding Window Attention** — Not a separate mechanism but a modifier applied to any attention type. Limits attention to a fixed window of recent tokens. Used by Mistral. Implemented as an attention mask strategy, not a separate `IAttentionMechanism`.

**Future candidates**: Linear attention variants, Mamba-style selective state spaces (if the project expands beyond pure transformers).

The `IAttentionMechanism` interface receives Q/K/V tensors, position encodings, KV-cache reference, attention mask, and optional hook points — and returns the attention output. Each implementation manages its own KV-cache layout requirements.

### Attention Kernel Strategy — `IAttentionStrategy`

Within each `IAttentionMechanism`, the actual compute can use different kernel strategies. The `IAttentionStrategy` selects the implementation based on backend capabilities:

- **Naive attention** — Materializes the full N×N score matrix. Used as reference implementation and fallback. O(N²) memory.
- **Flash Attention** — Tiled attention that keeps working data in on-chip SRAM (GPU) or L1/L2 cache (CPU). O(N) memory, 2–7× speedup from reduced memory I/O. FlashAttention-2 for Ampere, FlashAttention-3 for Hopper (asynchronous operations, FP8).
- **Paged Flash Attention** — Flash Attention adapted to work with PagedAttention's non-contiguous KV-cache blocks. Required for production serving with paged KV-cache.
- **CPU tiled attention** — Cache-friendly tiled implementation for CPU that processes Q/K/V in blocks fitting in L2 cache. Analogous to Flash Attention's HBM optimization but for the CPU memory hierarchy.

The backend advertises its available strategies; the engine selects the best one. On GPU, this means Flash Attention when available (compute capability ≥ 8.0), falling back to naive. On CPU, tiled by default.

### Position Encoding — `IPositionEncoding`

Position encoding is **optional and pluggable** via `IPositionEncoding`:

- **RoPE (Rotary Position Embeddings)** — Default for most modern LLMs (Llama, Mistral, Qwen, Phi-3). Pre-compute cos/sin tables up to max sequence length. Applied to Q and K tensors element-wise.
  - Standard RoPE (fixed theta, typically 10000)
  - Extended RoPE variants: **YaRN** (Yet another RoPE extensioN), **NTK-aware** scaling, **Dynamic NTK**, **CodeRoPE** — all configured via `RoPEConfig` parameters (theta, scaling_factor, scaling_type, etc.)
  - **Runtime context extension** — RoPE scaling parameters can be overridden at inference time (not just at model load) to extend effective context beyond training length. When changing scaling parameters at runtime, the KV-cache must be invalidated and the cos/sin tables recomputed.
- **ALiBi (Attention with Linear Biases)** — Adds position-dependent linear bias to attention scores. No modification to Q/K. Used by some models (BLOOM, MPT).
- **Absolute Learned Embeddings** — Adds a learned position embedding vector to token embeddings. Classic GPT-2 style.
- **None** — Some architectures may not use explicit position encoding (e.g., if baked into the architecture differently).

`ModelConfig` specifies `PositionEncodingType` and associated parameters. When `None`, no position encoding is applied. The `IPositionEncoding` is invoked by the model architecture before passing Q/K to the attention mechanism.

### Model Configuration — `ModelConfig`

`ModelConfig` is a comprehensive record describing any transformer variant:

```
ModelConfig:
  Architecture          (Llama | Mistral | Phi | Qwen | DeepSeek | ...)
  VocabSize             int
  HiddenSize            int
  IntermediateSize      int (FFN intermediate dim)
  NumLayers             int
  NumAttentionHeads     int
  NumKvHeads            int (== NumAttentionHeads for MHA, 1 for MQA, between for GQA)
  HeadDim               int (typically HiddenSize / NumAttentionHeads)
  MaxSequenceLength     int
  AttentionType         GQA | MLA
  PositionEncodingType  RoPE | ALiBi | Absolute | None
  PositionEncodingConfig  (RoPE theta, scaling, etc. — type-specific)
  ActivationFunction    SiLU | GELU | GELUTanh
  NormType              RMSNorm | LayerNorm
  NormEpsilon           float
  TiedEmbeddings        bool
  SlidingWindowSize     int? (null = no sliding window)
  MlaConfig             (latent dim, rope dim — only for MLA models)
  ChatTemplate          string? (Jinja2-style template from model metadata)
```

## Inference Engine Features

### Chat Template Engine

Models ship with specific chat templates that define how `messages[]` arrays (system, user, assistant, tool roles) are formatted into the raw token stream the model expects. Without a template engine, users must manually format prompts — which breaks the OpenAI-compatible server.

**How it works**:
- Templates are specified in Jinja2 format (the HuggingFace standard), stored in `tokenizer_config.json` as `chat_template` or embedded in GGUF metadata.
- The engine includes a **Jinja2-subset interpreter** sufficient for all known model templates. Required features: variable interpolation, `for` loops, `if`/`else` conditionals, string filters (`trim`, `strip`), `raise_exception`. Full Jinja2 is not needed.
- Templates are parsed once at model load and compiled into an efficient `IChatTemplate` callable.
- The `IChatTemplate` interface: `Apply(IReadOnlyList<ChatMessage> messages, ChatTemplateOptions options) → string`
- `ChatTemplateOptions` includes: `add_generation_prompt` (bool — append the assistant turn prefix), `tools` (list of tool definitions for tool-calling models).

**Known template formats**: Llama 3 (`<|begin_of_text|><|start_header_id|>...`), ChatML (`<|im_start|>user\n...`), Mistral (`[INST]...[/INST]`), Zephyr, Gemma, and custom per-model templates. The Jinja2 interpreter handles all of these — no need for per-model hardcoded formatters.

**Fallback**: If no template is found in model metadata, use a configurable default (ChatML is a reasonable default). Warn the user.

### Tool Calling / Function Calling

Tool calling is the integration of chat templates, structured output, and a protocol layer that enables models to invoke external functions:

**Protocol flow**:
1. User sends a request with `tools` definitions (function name, description, JSON schema for parameters).
2. The chat template formats the tool definitions into the prompt in the model's expected format.
3. The model generates a tool call: `{"name": "get_weather", "arguments": {"city": "London"}}`.
4. **Structured output** (constrained decoding) guarantees the tool call JSON conforms to the tool's parameter schema.
5. The server detects the tool call output, parses it, and returns it to the client with `finish_reason: "tool_calls"`.
6. The client executes the tool, sends the result back as a `tool` role message.
7. The chat template formats the tool result into the prompt; the model generates a final response.

**Implementation considerations**:
- Tool call detection: Models signal tool calls differently — some use special tokens (`<|tool_call|>`), others use JSON patterns. The `IChatTemplate` must handle both formatting and parsing of tool calls.
- Parallel tool calls: Some models support generating multiple tool calls in a single response. The parser must handle arrays of tool calls.
- The `IToolCallParser` interface extracts structured tool call data from the model's raw text output.

### Continuous Batching

The scheduler operates at **iteration granularity**, not request granularity. Key design:

- **Request queue** — Incoming requests are buffered with their prompt tokens and generation parameters.
- **Active batch** — Each scheduler iteration selects which sequences to include based on available memory (KV-cache capacity) and priority.
- **Iteration-level scheduling** — On every decode step:
  1. Check for completed sequences (EOS token, max length, stop conditions). Evict them and free their KV-cache blocks.
  2. Check for new requests that can be admitted given freed capacity.
  3. Run prefill for newly admitted requests (can batch prefill tokens).
  4. Run decode for all active sequences (one token per sequence per step).
- **Preemption** — If memory pressure is high, the scheduler can preempt (pause and swap out) lower-priority sequences, reclaiming their KV-cache blocks for higher-priority requests. Swapped sequences resume when capacity is available.
- **Prefill/decode separation** — Prefill (processing the full prompt) is compute-bound; decode (generating one token at a time) is memory-bandwidth-bound. The scheduler can handle them in separate micro-batches within a single iteration for optimal hardware utilization.
- **Request priority** — Each request carries a priority level (from the API key or explicit parameter). The scheduler uses priority for admission ordering and preemption decisions. Higher-priority requests preempt lower-priority ones when memory is scarce.

The scheduler is decoupled from the inference engine via an `IScheduler` interface, allowing experimentation with different scheduling policies.

### Speculative Decoding

Speculative decoding uses a small **draft model** to propose candidate tokens, which the larger **target model** verifies in parallel:

- **Draft phase** — The draft model (e.g., a smaller Llama 1B or a dedicated draft head) autoregressively generates K candidate tokens (typically K=3–5).
- **Verification phase** — The target model processes all K candidates in a single forward pass (batched), producing logits for each position.
- **Acceptance** — Tokens are accepted left-to-right using a **modified rejection sampling** scheme:
  - For each position i, compare draft probability `q(x_i)` with target probability `p(x_i)`.
  - Accept with probability `min(1, p(x_i)/q(x_i))`.
  - On first rejection at position j, sample a corrected token from `norm(max(0, p(x) - q(x)))` and discard positions j+1..K.
  - This guarantees the output distribution is **exactly equal** to the target model's distribution.
- **Speedup** — Typically **2–3× tokens/second** improvement, because the target model verifies K tokens in roughly the same time as generating 1.

Design considerations:
- Draft and target models share vocabulary and tokenizer (required for the acceptance scheme).
- Draft model can be: a separate smaller model, a shallow subset of the target model's layers, or a dedicated speculative head trained alongside the target.
- The `ISpeculativeDecoder` interface encapsulates the draft-verify-accept loop, allowing experimentation with different draft strategies.
- KV-cache management must handle speculative rollback — when tokens are rejected, their KV-cache entries must be invalidated.
- When constrained decoding is active, the draft model must also respect constraints during speculation. On rejection/rollback, the constraint automaton state must also roll back (via `IDecodingConstraint.Clone`).

### Paged KV-Cache (PagedAttention)

Inspired by OS virtual memory paging, applied to KV-cache management:

- KV-cache is divided into fixed-size **blocks** (e.g., 16 or 32 tokens per block).
- A **block table** (per sequence) maps logical token positions to physical blocks — identical to a page table.
- Blocks are allocated on demand from a **free pool**. No pre-allocation of max-sequence-length buffers.
- **Memory waste drops from ~60% to <4%** compared to static pre-allocation.
- Enables **copy-on-write** for beam search — beams sharing a common prefix reference the same physical blocks until they diverge.
- Enables **fork/share** semantics for prompt caching — multiple requests with the same system prompt can share prefix KV blocks.

### KV-Cache Quantization

Storing the KV-cache in FP16 is standard, but for long contexts the cache becomes a dominant memory consumer. KV-cache quantization compresses cached key/value tensors to reduce memory:

- **FP8 (E4M3/E5M2)** — 2× compression vs FP16 with minimal quality loss. Becoming the standard for production deployments on Hopper+ GPUs with native FP8 support.
- **INT8** — 2× compression. Requires per-head or per-block quantization scales. Slightly more quality impact than FP8.
- **INT4** — 4× compression. Higher quality impact, but useful for older tokens in very long contexts where attention weights are small.
- **Mixed precision** — Recent tokens in FP16 (high attention weight), older tokens quantized to INT8/INT4. A sliding window of full-precision cache with quantized tail.

KV-cache quantization is configured per-model via `KvCacheConfig` (dtype, mixed precision window size). Orthogonal to weight quantization — a Q4_K_M model can use FP8 KV-cache.

### Prompt Caching / Automatic Prefix Sharing

When many requests share the same system prompt (extremely common in production), the KV-cache for the shared prefix can be computed once and reused:

- **Automatic prefix caching** — The engine maintains a trie of recently computed prompt prefixes, keyed by token sequences. When a new request's prompt matches an existing prefix, the cached KV blocks are shared (read-only) and only the new suffix requires prefill computation.
- **Integration with PagedAttention** — Shared prefix blocks use reference counting. Blocks are freed only when all referencing sequences complete. Copy-on-write if a sequence needs to modify shared blocks (rare — KV-cache is append-only during normal generation).
- **Cache eviction** — LRU eviction when memory is scarce. Frequently reused prefixes (e.g., the system prompt) are effectively pinned.
- **Explicit prefix registration** — The server API can accept a `prefix_id` to explicitly register and reuse named prefixes, enabling deterministic caching for known prompt patterns.

### Structured Output — Constrained Decoding

The engine supports **constrained decoding** to guarantee that generated output conforms to a specified structure (JSON schema, regex, context-free grammar). This is critical for production use cases: tool calling, function outputs, API responses, structured data extraction.

**How it works** — Constrained decoding is a **logit masking step** in the sampler pipeline, executed *before* temperature/top-k/top-p:

1. A constraint specification (JSON schema, regex, or grammar) is compiled into a **finite state machine (FSM)** or **pushdown automaton (PDA)** at request time.
2. At each decode step, the automaton's current state determines which tokens are valid continuations.
3. A **token mask** is generated: valid token logits pass through, invalid token logits are set to `-∞`.
4. Normal sampling proceeds on the masked logits — the model's probability distribution is preserved over the valid token set.
5. After sampling, the automaton transitions to its next state based on the chosen token.

The output is **mathematically guaranteed** to conform to the constraint while preserving the model's distribution over valid continuations.

**Constraint types** (implementation priority order):

1. **JSON mode** — Guarantees syntactically valid JSON output. Compile a JSON grammar into an FSM that tracks parser state (in object, in array, in string, expecting key, expecting value, etc.). Relatively simple — the JSON grammar is context-free but the subset needed for well-formed output can be handled with a modest state machine.

2. **JSON Schema** — Guarantees output matches a specific JSON schema (required fields, types, enum values, nested objects, arrays with item types). Compile the schema into a more detailed automaton that enforces structural constraints beyond syntax. This is the highest-value feature for tool calling and structured APIs.

3. **Regex** — Compile a regular expression into a DFA. At each step, compute which tokens can extend the current partial match. Useful for constrained formats like dates, phone numbers, enums, identifiers.

4. **Context-free grammar (CFG)** — Support for arbitrary grammars in a BNF/GBNF-like notation (similar to llama.cpp's GBNF support). Uses a pushdown automaton. Most general but most complex. Enables constraining output to programming languages, custom DSLs, XML, etc.

**Key implementation considerations**:

- **Token-level vs byte-level masking** — Tokens often span multiple characters. The automaton must check whether any valid continuation *through* a multi-character token exists, not just whether the first character is valid. This requires pre-computing, for each FSM state, which vocabulary tokens can be accepted — the **token mask cache**.
- **Token mask precomputation** — For each FSM/DFA state, pre-compute the set of allowed token IDs. Cache these masks (indexed by state) to avoid re-scanning the full vocabulary at every step. For regex/JSON mode with modest state counts, this is feasible. For complex JSON schemas with many states, lazy computation with LRU caching.
- **Interaction with speculative decoding** — The draft model must also respect constraints during speculation. Each speculated token advances the automaton state. On rejection/rollback, the automaton state must also roll back.
- **Interaction with continuous batching** — Each sequence in a batch may have a different constraint (or none). The logit masking step applies per-sequence masks, which can be batched as a single masked operation on the logit tensor.

**Interface design**:

The constraint is expressed as an `IDecodingConstraint` that plugs into the sampler pipeline:

```
IDecodingConstraint:
  Advance(tokenId) → void           // Update automaton state after token is sampled
  GetAllowedTokens() → TokenMask    // Bit mask over vocabulary for current state
  IsComplete() → bool               // Whether the constraint is fully satisfied
  Clone() → IDecodingConstraint      // For speculative decoding rollback (snapshot state)
  Reset() → void                     // Return to initial state
```

`TokenMask` is a compact bit vector over the vocabulary (e.g., 128K vocab = 16KB mask). Applied to logits via vectorized AND/masked-fill operation.

Concrete implementations: `JsonConstraint`, `JsonSchemaConstraint`, `RegexConstraint`, `GrammarConstraint`.

**Reference implementations to study**: llama.cpp's GBNF grammar support, Outlines (Python, FSM-based structured generation), guidance (Microsoft, interleaved generation and control), XGrammar (optimized grammar-based constrained decoding used by vLLM/MLC-LLM).

### Sampling Pipeline — Composable `ISamplerStep`

The sampler pipeline is a composable chain of `ISamplerStep` operations applied to raw logits before the final token selection. Each step implements `ISamplerStep` with `Apply(Span<float> logits, SamplerContext ctx) → void`. Steps are ordered and can be reordered or extended by the user.

**Default pipeline order**:
1. **Logit bias** (`LogitBiasStep`) — Apply per-token additive biases from the request's `logit_bias` map. OpenAI API compatible: `{token_id: bias_value}`.
2. **Constrained decoding** (`ConstraintMaskStep`) — Apply `IDecodingConstraint` token mask if structured output is active. Invalid tokens → `-∞`.
3. **Repetition penalties** (`RepetitionPenaltyStep`) — Configurable penalty types:
   - **Repetition penalty** (multiplicative): logits for previously generated tokens are divided by the penalty factor. Used by most open models.
   - **Frequency penalty** (additive, proportional to count): penalize tokens based on how many times they've appeared. OpenAI API compatible.
   - **Presence penalty** (additive, binary): penalize tokens that have appeared at all. OpenAI API compatible.
4. **Temperature** (`TemperatureStep`) — Divide logits by temperature. `T=0` → greedy (argmax), `T=1` → unmodified, `T>1` → more random.
5. **Top-K** (`TopKStep`) — Keep only the K highest-probability tokens.
6. **Top-P / Nucleus** (`TopPStep`) — Keep the smallest set of tokens whose cumulative probability ≥ P.
7. **Min-P** (`MinPStep`) — Keep tokens with probability ≥ `min_p × max_probability`. More stable than top-p across different distributions.
8. **Categorical sampling** (`CategoricalSampleStep`) — Sample from the resulting distribution. Or argmax if temperature was 0.

**Custom logit processors** — Users can inject arbitrary `ILogitProcessor` steps at any point in the pipeline:
```
ILogitProcessor:
  Process(Span<float> logits, IReadOnlyList<int> previousTokens, ProcessorContext ctx) → void
```
This is the user-facing equivalent of the diagnostics hook system but for the sampling stage. Use cases: classifier-free guidance, contrastive decoding, custom penalty schemes.

### Beam Search

In addition to sampling-based decoding, the engine supports **beam search** for tasks where output quality/probability matters more than diversity:

- Maintain N candidate sequences (beams), expanding each by one token per step.
- At each step, score all `N × vocab_size` candidates, keep top N by cumulative log-probability.
- Apply length normalization to avoid bias toward shorter sequences.
- Stop when all beams have generated EOS or hit max length.
- Return the top-K completed sequences ranked by normalized score.

**Integration with KV-cache**: Beams sharing a common prefix reuse the same KV-cache blocks (copy-on-write via PagedAttention). Only the diverging suffix tokens require new KV-cache allocations.

**Integration with constraints**: Each beam maintains its own `IDecodingConstraint` state (cloned from the original at branch points).

### Stop Conditions — `IStopCondition`

Generation stops when any registered `IStopCondition` fires. Multiple conditions can be active simultaneously (first match wins):

- **EOS token** — The model's end-of-sequence token. Always active.
- **Max tokens** — Hard limit on generated tokens. Always active.
- **Stop strings** — One or more text strings that terminate generation when produced (e.g., `"\n\nHuman:"`, `"END"`, `"```"`). OpenAI API compatible (`stop: ["...", "..."]`). Implementation: maintain a rolling buffer of recent decoded text, check for suffix matches after each token. The stop string is not included in the output.
- **Stop token sequences** — Like stop strings but specified as token ID sequences rather than text. Avoids tokenization ambiguity.
- **Custom predicates** — `IStopCondition` interface: `ShouldStop(int tokenId, ReadOnlySpan<int> generatedTokens, string decodedText) → StopResult`. Enables arbitrary stopping logic (e.g., stop after N sentences, stop when a regex matches).

`StopResult` indicates: `Continue`, `Stop` (exclude trigger), or `StopInclude` (include trigger in output).

### LoRA Adapter Support

Load and apply **LoRA (Low-Rank Adaptation)** adapters at runtime without merging into base weights. Enables multi-tenant serving where different users/requests use different fine-tuned behaviors on the same base model.

**How LoRA works at inference**: For each adapted linear layer, the output is `y = x @ W + α(x @ B) @ A`, where `W` is the frozen base weight, `B` (down-projection, `d × r`) and `A` (up-projection, `r × d'`) are the low-rank adapter matrices, `α` is a scaling factor, and `r` is the rank (typically 8–64). The adapter adds negligible compute compared to the base weight matmul.

**Design**:
- **Adapter loading** — Parse adapter weights from SafeTensors or GGUF. An adapter is a collection of `(layer_name, A_matrix, B_matrix, alpha, rank)` entries. Adapters are small (typically 10–100MB for a 7B model).
- **Multi-adapter management** — Multiple named adapters can be loaded simultaneously. Each request specifies which adapter (if any) to apply via a `lora_adapter` parameter.
- **Runtime application** — In the forward pass, adapted layers check whether an adapter is active for the current request. If so, they compute the LoRA delta and add it. If not, they execute the base weight path only.
- **Batching with mixed adapters** — In a continuous batch, different sequences may use different adapters (or none). The matmul for base weights is batched normally. The LoRA delta is computed per-adapter-group and added. This requires grouping sequences by adapter in the batch for efficient execution.
- **No weight merging** — Adapters are never merged into base weights. This allows instant adapter switching and multiple concurrent adapters. The trade-off is a small per-layer overhead for the LoRA matmul, but with typical ranks (r=16–32) this is <5% of the base matmul cost.

**Interface**: `IAdapterManager` handles loading, caching, and per-request adapter selection. Adapted layers query it via a `RequestContext` that carries the active adapter ID.

### Multi-GPU / Tensor Parallelism

For models that exceed single-GPU memory (70B+), the engine supports distributing compute across multiple GPUs:

**Tensor parallelism (TP)** — The primary strategy. Splits individual tensor operations across GPUs:
- Attention: Split Q/K/V heads across GPUs. Each GPU computes attention for its head subset. Outputs are all-gathered.
- FFN: Split the up/gate projection columns across GPUs (each GPU computes a slice of the intermediate), and split the down projection rows. Requires an all-reduce after the down projection.
- All communication uses **NCCL** (via native library P/Invoke).

**Pipeline parallelism (PP)** — Split layers across GPUs. GPU 0 runs layers 0–15, GPU 1 runs layers 16–31. Simpler communication (just forward the hidden state between stages) but harder to keep all GPUs busy. Useful combined with TP.

**Architectural requirements** (must be satisfied even before multi-GPU is implemented):
- Every tensor must carry explicit **device placement** (`DeviceId` in tensor metadata). No implicit "current device" state.
- The `IBackend` interface must support multi-device operations: `AllReduce`, `AllGather`, `Send`/`Recv`.
- The model forward pass must be parameterizable by a `ParallelismConfig` that specifies TP degree, PP degree, and device mapping.
- KV-cache blocks must be device-local — each GPU manages its own KV-cache pool for its assigned heads.

Initial implementation is single-device. The abstractions exist from day one so multi-GPU doesn't require architectural surgery later.

## Diagnostics & Interpretability

### Hook System

The inference pipeline exposes **hook points** at well-defined locations, allowing external code to intercept and inspect (or modify) activations during inference. This is modeled after PyTorch's `register_forward_hook` and HuggingFace's approach.

**Hook points** (each fires with the tensor at that stage):
- `PostEmbedding` — After token embedding lookup, before first layer
- `PreAttention(layer)` — Input to attention mechanism at layer N (after pre-attention norm)
- `PostAttention(layer)` — Output of attention mechanism at layer N (before residual add)
- `PreFfn(layer)` — Input to FFN at layer N (after post-attention norm)
- `PostFfn(layer)` — Output of FFN at layer N (before residual add)
- `PostLayer(layer)` — After residual add at end of layer N (the residual stream)
- `PreLmHead` — Final hidden state before the LM head projection
- `PostLmHead` — Raw logits before sampling

**Hook interface**:
```
IInferenceHook:
  HookPoint      — Which point to attach to
  OnActivation(ReadOnlySpan<float> activation, HookContext ctx) → HookResult
```

`HookResult` can be:
- `Continue` — Proceed normally (read-only inspection)
- `Replace(Span<float>)` — Replace the activation with a modified version (for interventions, steering, ablation studies)

**HookContext** carries metadata: layer index, token position, sequence ID, current step.

Hooks are registered on the `InferenceEngine` and are **disabled by default** (zero overhead when no hooks are registered). When enabled, the engine materializes activations at hook points — this has a performance cost (memory + copy) and is intended for research/debugging, not production serving.

### Built-in Diagnostic Tools

**Activation capture** — `CaptureHook` collects activations at specified layers/positions into a buffer for offline analysis. Configurable to capture all tokens or only specific positions.

**Logit lens** — Projects intermediate hidden states (at any `PostLayer` hook point) through the LM head to produce token probability distributions at each layer. Reveals how the model's "belief" about the next token evolves through layers.

**Attention pattern capture** — When enabled, the attention mechanism exports the full attention weight matrix (softmax output) for visualization or analysis. Expensive for long sequences (O(n²) per head per layer) — use selectively.

**Sparse Autoencoder (SAE) integration** — The hook system enables attaching trained SAEs to any residual stream position:
1. Register a `Replace` hook at `PostLayer(layer)`.
2. The hook encodes the activation through the SAE encoder, producing sparse feature activations.
3. Optionally: log/analyze the sparse features, modify them (feature steering/ablation), then decode back.
4. Return the (possibly modified) activation to the inference pipeline.

This enables mechanistic interpretability workflows entirely within .NET — no need to export activations to Python for analysis. A sample project (`DotLLM.Sample.Interpretability`) demonstrates loading a pre-trained SAE, hooking it into inference, and analyzing feature activations.

## Observability

### Metrics — `IInferenceMetrics`

The engine exposes structured metrics for monitoring and performance analysis, compatible with **OpenTelemetry** and **Prometheus** exporters:

**Throughput metrics**:
- `dotllm_tokens_per_second_prefill` — Tokens/sec during prompt processing (compute-bound)
- `dotllm_tokens_per_second_decode` — Tokens/sec during generation (memory-bandwidth-bound)
- `dotllm_requests_completed_total` — Counter of completed requests

**Latency metrics** (histograms):
- `dotllm_time_to_first_token_seconds` — Time from request receipt to first generated token (TTFT)
- `dotllm_inter_token_latency_seconds` — Time between consecutive generated tokens (ITL)
- `dotllm_request_duration_seconds` — Total request latency including queue wait

**Resource utilization**:
- `dotllm_kv_cache_utilization_ratio` — Fraction of KV-cache blocks in use
- `dotllm_kv_cache_blocks_allocated` / `_free` / `_total`
- `dotllm_gpu_memory_used_bytes` / `_total_bytes`
- `dotllm_batch_size_current` — Number of active sequences in the current batch
- `dotllm_queue_depth` — Number of requests waiting for admission

**Scheduler metrics**:
- `dotllm_preemptions_total` — Number of sequence preemptions
- `dotllm_prefix_cache_hit_ratio` — Hit rate for automatic prefix caching

All metrics are collected via `System.Diagnostics.Metrics` (the .NET standard metrics API), enabling integration with any OpenTelemetry-compatible collector. Zero-cost when no listener is attached.

### Request Tracing — `IRequestTracer`

End-to-end distributed tracing of individual requests through the pipeline, using `System.Diagnostics.Activity` (OpenTelemetry-compatible):

**Trace spans**:
- `dotllm.request` — Root span covering the full request lifecycle
  - `dotllm.queue_wait` — Time spent in the scheduler queue
  - `dotllm.tokenize` — Prompt tokenization
  - `dotllm.template` — Chat template application
  - `dotllm.prefix_lookup` — Prefix cache lookup
  - `dotllm.prefill` — Prompt KV-cache computation
    - `dotllm.layer.{n}` — Per-layer forward pass (optional, high verbosity)
  - `dotllm.decode` — Token generation loop
    - `dotllm.sample` — Sampling pipeline (including constraint evaluation)
  - `dotllm.detokenize` — Token-to-text conversion

Each span carries attributes: token counts, model name, adapter ID, constraint type, GPU device, batch position. This enables identifying bottlenecks (e.g., slow prefill, constraint overhead, queue contention) in production.

## Server Features

### OpenAI-Compatible API Endpoints

The ASP.NET Minimal API server exposes these endpoints:

- **`POST /v1/chat/completions`** — Primary chat endpoint. Supports: `messages`, `model`, `temperature`, `top_p`, `max_tokens`, `stop`, `stream`, `tools`, `tool_choice`, `response_format` (JSON mode/schema), `logit_bias`, `n` (beam count), `frequency_penalty`, `presence_penalty`, `lora_adapter` (extension).
- **`POST /v1/completions`** — Raw completion endpoint (no chat template). Same sampling parameters.
- **`POST /v1/embeddings`** — Extract embedding vectors. Uses the final hidden state (configurable: last layer, mean pool, CLS token). Returns OpenAI-compatible embedding response. Low implementation effort given the hook system — register a `PostLayer(last)` or `PreLmHead` hook, pool the hidden state, normalize.
- **`GET /v1/models`** — List loaded models and their metadata.
- **`POST /v1/tokenize`** (extension) — Tokenize text and return token IDs, token strings, and count. Useful for prompt engineering, billing estimation, and debugging. Not part of the OpenAI spec but widely expected.
- **`POST /v1/detokenize`** (extension) — Convert token IDs back to text.

### Rate Limiting and Request Priority

Production servers need per-user/per-API-key controls:

- **API key authentication** — Configurable API key validation. Keys carry metadata: priority level, rate limits, allowed models, allowed adapters.
- **Rate limiting** — Per-key token-bucket rate limiter. Configurable limits on: requests per minute, tokens per minute (prompt + completion), concurrent requests. Uses `System.Threading.RateLimiting` (.NET 7+).
- **Request priority** — Requests inherit priority from their API key (or explicit parameter). The scheduler uses priority for admission and preemption. Priority levels: `low`, `normal`, `high`, `critical`.
- **Graceful degradation** — When overloaded, low-priority requests are queued or rejected (HTTP 429) before high-priority requests are affected.

### Warm-up

A pre-inference warm-up pass that runs at server startup:

- Triggers JIT compilation (with Dynamic PGO) of all hot inference paths by running a dummy forward pass.
- Pre-loads CUDA kernels and cuBLAS handles.
- Pre-computes RoPE cos/sin tables, tokenizer trie structures, and other one-time initialization.
- Ensures the first real request doesn't pay startup penalties.

Configurable via `WarmupOptions`: enabled/disabled, dummy prompt length, number of warm-up iterations.

## Technical Standards & Conventions

### C# Code Style

- Use file-scoped namespaces.
- Use `readonly record struct` for small value types (TensorShape, DType, TokenId).
- Prefer `Span<T>` and `ReadOnlySpan<T>` over arrays in method signatures.
- Use `[MethodImpl(MethodImplOptions.AggressiveInlining)]` on small, hot-path methods.
- Use `[SkipLocalsInit]` on performance-critical methods to avoid zero-initialization overhead.
- All public APIs must have XML documentation comments.
- Use nullable reference types (`<Nullable>enable</Nullable>`) project-wide.
- Name GPU memory handles as `devicePtr` or `dPtr`, CPU memory as `hostPtr` or `hPtr`.
- Async methods that return `IAsyncEnumerable<T>` for streaming token generation.

### Memory Management Rules

- **NEVER** allocate managed arrays for tensor data. Use `NativeMemory.AlignedAlloc`.
- Use `ArrayPool<T>.Shared` for temporary scratch buffers (return promptly).
- Model weights must be memory-mapped via `MemoryMappedFile` — no copying into managed heap. The OS page cache naturally provides **weight caching across process restarts** — as long as the file remains in the page cache, subsequent loads are near-instant. For explicit multi-process weight sharing, multiple processes can mmap the same file.
- All unmanaged memory must be wrapped in types implementing `IDisposable` with deterministic cleanup.
- GC configuration: Server GC, `SustainedLowLatency` mode during inference.
- Tensor metadata objects (shape, stride, pointer) should be structs, not classes, to avoid heap allocations.

### SIMD & Vectorization

- Use `System.Numerics.Tensors.TensorPrimitives` as the foundation for CPU tensor ops (Dot, Softmax, Exp, Add, Multiply, CosineSimilarity).
- For hot inner loops (quantized matmul, RoPE), drop to `System.Runtime.Intrinsics`:
  - Prefer cross-platform `Vector128<T>`/`Vector256<T>` APIs where possible.
  - Use platform-specific intrinsics (`Fma.MultiplyAdd`, `Avx2.MultiplyAddAdjacent`) only when the cross-platform path leaves measurable performance on the table.
- Always provide a scalar fallback path for platforms without SIMD support.
- Align data to 64 bytes for AVX-512, 32 bytes for AVX2.

### GPU Interop Rules

- The native C/CUDA library exposes a **flat C API** (no C++ classes across the boundary).
- Use `[LibraryImport]` (source-generated P/Invoke, .NET 7+), not `[DllImport]`.
- Use `[SuppressGCTransition]` for trivially short native calls (status queries, handle validation).
- GPU memory handles are `nint` (`IntPtr`) in C# — the native side owns the pointer semantics.
- Native API is coarse-grained: `LoadTensors`, `Forward`, `Attention`, `Sample` — each call takes milliseconds, making P/Invoke overhead invisible.
- Ship native binaries under `runtimes/{rid}/native/` in NuGet packages per .NET RID conventions.

### Quantization

Implementation priority order:
1. **FP16** — baseline, no quantization
2. **Q8_0** — simplest quantized format (block of 32, single scale, `val = scale × q`)
3. **Q4_0** — 4-bit quantization (block of 32, `val = scale × (q - 8)`)
4. **Q4_K_M** — K-quant with super-blocks of 256, sub-blocks with double quantization. Most popular GGUF format.
5. **Q5_K_M, Q6_K** — higher quality K-quants
6. **GPTQ/AWQ** — GPU-native formats (future)

Each quantization type needs: a dequantization kernel (CPU SIMD + CUDA), a `vec_dot` kernel for fused quantized dot product, and a block structure definition.

**Mixed quantization** — GGUF files can contain different quantization types per tensor (e.g., attention layers at Q6_K, FFN at Q4_K_M). The engine must handle heterogeneous quantization naturally by dispatching to the correct dequantization/vec_dot kernel per tensor based on its metadata. Do not assume uniform quantization across the model.

### Model Architecture Pattern

All supported transformer architectures follow this general pattern — parameterize, do not duplicate:

```
Token Embedding
  → (optional) Position Encoding (if absolute/learned)
→ N × [
    Norm → Attention(Q, K, V, position_encoding, kv_cache, mask) → Residual Add
    → Norm → FFN (gate × up, activation, down) → Residual Add
  ]
→ Final Norm → LM Head (linear projection to vocab)
```

Where:
- **Norm** = RMSNorm or LayerNorm (per ModelConfig)
- **Attention** = any IAttentionMechanism (GQA, MLA, etc.), using the best available IAttentionStrategy (flash, tiled, naive)
- **Position encoding** = applied to Q/K inside attention (RoPE), or as bias to scores (ALiBi), or to embeddings (absolute), or not at all
- **FFN activation** = SiLU, GELU, GELUTanh (per ModelConfig)
- **Residual** = always additive
- **LoRA** = if adapter is active for the current request, adapted layers compute and add the low-rank delta

Hook points fire between each stage when diagnostics are enabled.

### Testing Strategy

- Unit tests for every kernel (matmul, softmax, RoPE, quantization) comparing against known-good reference outputs.
- Numerical accuracy tests: dequantize → compute → compare against FP32 reference within epsilon.
- BenchmarkDotNet benchmarks for regression detection on all CPU kernels.
- Integration tests: load a small model (e.g., TinyLlama 1.1B), generate tokens, validate coherent output.
- Hook system tests: verify hooks fire at correct points with correct shapes, verify Replace hooks modify the activation stream.
- Constrained decoding tests: generate with JSON schema constraints, validate all outputs parse correctly.
- Chat template tests: verify template rendering against HuggingFace reference outputs for each supported model family.
- CI must run on both Linux and Windows.

## Implementation Roadmap (Build Order)

This is the recommended sequence — each step builds on the previous:

**Phase 1 — End-to-end single-token generation**
1. GGUF loader — Parse header, metadata, tensor descriptors. Memory-map tensor data.
2. FP16/Q8_0 dequantization — Simplest quantized ops to validate the tensor pipeline.
3. Basic CPU tensor ops — MatMul (GEMV), RMSNorm, SiLU, Softmax using TensorPrimitives + intrinsics.
4. BPE tokenizer — Parse vocabulary from GGUF metadata. Encode/decode text.
5. GQA attention + RoPE — The most common attention/position encoding combination.
6. Llama forward pass — First end-to-end inference. Single-token generation.
7. Simple KV-cache — Pre-allocated, fixed-size. Enables multi-token generation.
8. Sampling pipeline — Composable ISamplerStep chain: repetition penalties → temperature → top-k → top-p → min-p → categorical.
9. Stop conditions — EOS, max tokens, stop strings.

**Phase 2 — Practical local inference**
10. Q4_K_M dequantization — Unlocks the most popular quantization format.
11. Mixed quantization support — Handle heterogeneous per-tensor quantization types.
12. Chat template engine — Jinja2-subset interpreter, parse from GGUF/tokenizer_config.json.
13. Streaming generation — `IAsyncEnumerable<string>` token-by-token output.
14. Hook system — IInferenceHook interface, hook point firing, CaptureHook implementation.
15. Logit lens — Built on hook system. Project intermediate hidden states through LM head.
16. Additional architectures — Mistral (sliding window), Phi, Qwen via ModelConfig parameterization.
17. Logit bias — Per-request logit bias map (OpenAI API compatible).

**Phase 3 — GPU acceleration**
18. CUDA backend — cuBLAS GEMM + custom flash attention + quantized matmul kernels.
19. CPU/GPU hybrid — Layer offloading (some layers on GPU, rest on CPU).
20. KV-cache quantization — FP8/INT8 cache compression for longer context.

**Phase 4 — Production serving**
21. ASP.NET server — OpenAI-compatible endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/v1/tokenize`.
22. Continuous batching — Iteration-level scheduler with preemption and request priority.
23. Paged KV-cache — PagedAttention for production-grade memory management.
24. Prompt caching — Automatic prefix sharing with trie-based lookup.
25. Rate limiting — Per-API-key token-bucket rate limiting and priority levels.
26. Structured output: JSON mode — FSM-based constrained decoding guaranteeing valid JSON.
27. Structured output: JSON Schema — Schema-compiled automaton for typed, structured responses.
28. Structured output: Regex + CFG — Regex DFA and GBNF-style grammar constraints.
29. Tool calling — Function calling protocol with IChatTemplate integration and structured output for arguments.
30. Speculative decoding — Draft-verify-accept loop with KV-cache and constraint state rollback.
31. Beam search — N-best decoding with length normalization and COW KV-cache.
32. Metrics & tracing — OpenTelemetry-compatible metrics and per-request distributed tracing.
33. Warm-up — JIT pre-compilation and CUDA kernel pre-loading at server startup.

**Phase 5 — Expand**
34. LoRA adapters — Runtime adapter loading, multi-adapter batching, per-request adapter selection.
35. MLA attention — DeepSeek-V2/V3 support. Latent KV-cache.
36. ALiBi position encoding — For models that use it.
37. SAE integration — Sparse autoencoder hooks with sample project.
38. Multi-GPU tensor parallelism — NCCL-based TP with head/FFN sharding.
39. ROCm backend — HIP conditional compilation of CUDA kernels. Separate NuGet package.

**Future Considerations** (not in current roadmap, but architecture should not preclude):
- Runtime quantization — Load FP16 model and quantize at load time.
- Vision / multimodal input — Image encoders (CLIP ViT) for LLaVA, Phi-3-Vision, Qwen-VL. Requires `IInputEncoder` abstraction mapping raw inputs to embeddings.
- Guided / interactive generation — Pause generation, inject external tokens (e.g., tool results), resume from arbitrary KV-cache state.
- Model merging — SLERP/TIES/DARE weight arithmetic as a utility feature.
- Pipeline warm-up profiling — Profile warm-up runs to auto-tune batch sizes and memory allocation.

## NuGet Packaging Strategy

| Package | Contents |
|---------|----------|
| `DotLLM` | Core library + CPU backend + models + tokenizers + templates + engine + diagnostics + telemetry. Pure .NET, no native deps. |
| `DotLLM.Backend.Cuda12` | CUDA 12 backend + native CUDA shared library binaries. |
| `DotLLM.Backend.ROCm` | ROCm/HIP backend (future). |
| `DotLLM.Server` | ASP.NET OpenAI-compatible API server with rate limiting. |

## Key Design Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| License | GPLv3 | Strong copyleft ensures all derivative works remain open source. |
| GPU interop | P/Invoke to custom C/CUDA lib | ManagedCUDA is GPLv3 + single maintainer; ILGPU can't access cuBLAS; Silk.NET has no CUDA bindings. P/Invoke overhead is <0.001% of kernel time. |
| Tensor memory | `NativeMemory.AlignedAlloc`, unmanaged | Eliminates GC pressure entirely for tensor data. Enables memory-mapped model loading. |
| Model loading | Memory-mapped GGUF | OS handles demand paging. 7GB model "loads" in milliseconds. No managed heap copies. OS page cache provides cross-process weight sharing. |
| SIMD foundation | `TensorPrimitives` + `System.Runtime.Intrinsics` | TensorPrimitives for standard ops; hand-tuned intrinsics only for quantized inner loops. |
| Default runtime | JIT with Dynamic PGO | Better steady-state throughput than NativeAOT. NativeAOT as opt-in for edge deployment. |
| Primary model format | GGUF | Self-contained (includes tokenizer), supports quantization, massive ecosystem (HuggingFace). |
| Attention abstraction | `IAttentionMechanism` + `IAttentionStrategy` | Mechanism (GQA, MLA) is separate from kernel strategy (naive, flash, paged-flash). Both are pluggable. |
| Position encoding | `IPositionEncoding`, optional | RoPE is most common but not universal. ALiBi, absolute, and none must be supported. Runtime context extension supported for RoPE. |
| Sampler pipeline | Composable `ISamplerStep` chain | Each step (logit bias, constraints, penalties, temperature, top-k/p, min-p) is independent, reorderable, extensible. Custom `ILogitProcessor` injection supported. |
| Structured output | FSM/PDA logit masking in sampler pipeline | Guarantees valid JSON/schema/regex output. Operates as a composable sampler step, interacts cleanly with speculative decoding via state snapshot/rollback. |
| Chat templates | Jinja2-subset interpreter | Covers all known model templates without per-model hardcoding. Parsed once, compiled to efficient callable. |
| Multi-GPU | Tensor parallelism via NCCL | Explicit device placement from day one. TP splits heads and FFN columns. Abstractions ready even before implementation. |
| LoRA | Runtime application, no weight merging | Enables instant adapter switching and concurrent multi-adapter serving. Small overhead vs. large flexibility gain. |
| ROCm strategy | HIP conditional compilation of CUDA source | HIP is API-compatible with CUDA. Same pattern as llama.cpp. |
| API compatibility | OpenAI `/v1/chat/completions` + extensions | De facto standard. Extensions for tokenization, adapter selection. |
| Diagnostics | Hook-based, zero-cost when disabled | Brings HuggingFace-level interpretability tooling to .NET. Hooks are opt-in — no overhead in production. |
| Observability | `System.Diagnostics.Metrics` + `Activity` | Native .NET OpenTelemetry integration. Zero-cost when no listener attached. |

## What Claude Should Know When Working on This Project

- Performance is the #1 priority after correctness. Every allocation matters. Every branch in a hot loop matters. Benchmark before and after changes to kernels.
- When implementing SIMD kernels, always verify numerical accuracy against a scalar reference implementation first, then optimize.
- The GGUF format specification is the source of truth for model loading. When in doubt, check how llama.cpp implements it.
- For CUDA kernels, reference llama.cpp's `ggml-cuda/` directory for proven implementations of attention, quantized matmul, and fused kernels.
- When creating new model architectures, verify against HuggingFace's transformers reference implementations for numerical correctness.
- Prefer composability over inheritance. Use interfaces and records, not deep class hierarchies.
- Keep the native C/CUDA API surface minimal and stable. Changes to the native API require updating P/Invoke declarations and rebuilding native binaries across platforms.
- The diagnostics/hook system must have **zero overhead when no hooks are registered**. Use a simple null check or flag, not an event invocation pattern that allocates.
- The telemetry system must have **zero overhead when no metric listener/tracer is attached**. Use `System.Diagnostics.Metrics` and `Activity` which are designed for this.
- All device placement must be explicit. Never rely on implicit "current device" state. This is critical for future multi-GPU support.
- When in doubt about an architectural decision, document it in the "Key Design Decisions Log" section above with rationale.
