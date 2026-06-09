# Server — dotLLM

## Overview

ASP.NET Minimal API server providing OpenAI-compatible endpoints. Wires together the inference engine, tokenizer, chat templates, scheduler, and telemetry.

## Endpoints

### `POST /v1/chat/completions`
Primary chat endpoint. Accepts OpenAI-compatible request format.

**Request body**:
```json
{
  "model": "llama-3-8b-q4_k_m",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 256,
  "stream": true,
  "stop": ["\n\n"],
  "tools": [...],
  "tool_choice": "auto",
  "response_format": {"type": "json_schema", "json_schema": {...}},
  "logit_bias": {"1234": -100},
  "frequency_penalty": 0.5,
  "presence_penalty": 0.3,
  "n": 1,
  "lora_adapter": "customer-support"
}
```

**Response** (non-streaming):
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "llama-3-8b-q4_k_m",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help?"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23}
}
```

**Streaming**: Server-Sent Events (SSE). Each chunk:
```
data: {"id":"...","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: [DONE]
```

### `POST /v1/completions`
Raw completion (no chat template). Same sampling parameters. Input is `prompt` (string) instead of `messages`.

### `POST /v1/embeddings`
Extract embedding vectors from text.

**Request**: `{"input": "text to embed", "model": "..."}`
**Response**: `{"data": [{"embedding": [0.1, -0.2, ...], "index": 0}]}`

Implementation: Run input through the model, capture hidden state at `PreLmHead` hook point, apply pooling (mean pool over tokens by default, configurable), L2 normalize. Minimal additional code given the hook system.

### `GET /v1/models`
List loaded models: `{"data": [{"id": "llama-3-8b-q4_k_m", "object": "model"}]}`

### `POST /v1/tokenize` (extension)
**Request**: `{"text": "Hello world", "model": "..."}`
**Response**: `{"tokens": [9906, 1917], "token_strings": ["Hello", " world"], "count": 2}`

Not in OpenAI spec but widely expected for prompt engineering and billing estimation.

### `POST /v1/detokenize` (extension)
**Request**: `{"tokens": [9906, 1917], "model": "..."}`
**Response**: `{"text": "Hello world"}`

## response_format Processing

The `response_format` field maps to constrained decoding:

| `response_format.type` | Action |
|------------------------|--------|
| `"text"` | No constraint |
| `"json_object"` | `JsonConstraint` — guarantees valid JSON |
| `"json_schema"` | `JsonSchemaConstraint` compiled from `response_format.json_schema` |

The constraint is passed to the sampler pipeline and applied at every decode step.

## Tool Calling Flow

When `tools` are provided in the request:

1. **Prompt formatting**: `IChatTemplate.Apply(messages, options: { Tools = tools })` includes tool definitions in the prompt using the model's expected format.
2. **Generation**: Model generates response. If structured output is configured for tool calls, the JSON arguments are constrained to match the tool's parameter schema.
3. **Detection**: `IToolCallParser.TryParse(output)` checks if the output contains tool calls.
4. **Response**: If tool calls detected, return with `finish_reason: "tool_calls"` and structured `tool_calls` array.
5. **Continuation**: Client sends tool results as `tool` role messages. Server applies chat template again and generates final response.

## Prompt Caching

Multi-turn conversations benefit from prompt caching — reusing KV-cache state from previous turns to skip redundant prefill.

### How It Works

1. After each generation, `TextGenerator` stores the KV-cache and its full token sequence (prompt + generated) in a `PrefixCache`.
2. On the next request, the new prompt's token IDs are compared element-wise against cached entries to find the longest common prefix.
3. On cache hit: the cached KV-cache is reused, `CurrentLength` is truncated to the matched prefix, and only the new suffix tokens are prefilled.
4. On cache miss: a fresh KV-cache is allocated as usual.

This dramatically reduces time-to-first-token (TTFT) for multi-turn chat, where each turn's prompt shares a long prefix with the previous turn.

### Configuration

Prompt caching is **enabled by default** in both `chat` and `serve` commands.

| Flag | Default | Description |
|------|---------|-------------|
| `--no-prompt-cache` | `false` | Disable prompt caching |
| `--prompt-cache-size` | 1 (chat) / 4 (serve) | Maximum number of cached sessions (LRU eviction) |

### API

Cached token statistics are included in the `timings` field of streaming SSE responses:

```json
{
  "timings": {
    "prefill_time_ms": 2.1,
    "cached_tokens": 847,
    "prompt_tokens": 892
  }
}
```

### `POST /v1/cache/clear`

Clears all cached KV-cache sessions. Called automatically by the Chat UI when the conversation is cleared. Useful for freeing memory or resetting state.

**Response**: `{"status": "cleared"}`

### Scope

- CPU `SimpleKvCache` only. QuantizedKvCache and GPU caches fall back to no caching.
- Cache is cleared on model swap/reload.
- No session-based routing — single global LRU cache, serialized by the request gate.

## Rate Limiting

Per-API-key controls using `System.Threading.RateLimiting`:

### Configuration
```json
{
  "RateLimiting": {
    "DefaultPolicy": {
      "RequestsPerMinute": 60,
      "TokensPerMinute": 100000,
      "ConcurrentRequests": 5
    },
    "ApiKeys": {
      "key-premium": {
        "Priority": "High",
        "RequestsPerMinute": 600,
        "TokensPerMinute": 1000000,
        "ConcurrentRequests": 50
      }
    }
  }
}
```

### Token Counting
Rate limiting by tokens requires counting both prompt tokens (known at request time) and completion tokens (known only after generation). Strategy:
- Deduct estimated completion tokens (using `max_tokens`) from the token budget at request admission.
- After completion, adjust the actual count. Refund unused tokens.

### Response on Limit
HTTP 429 Too Many Requests with `Retry-After` header.

## Request Priority

API keys have priority levels: `Low`, `Normal`, `High`, `Critical`. Priority flows from API key config → request → scheduler.

Higher-priority requests:
- Bypass lower-priority requests in the scheduler queue
- Can trigger preemption of lower-priority active sequences
- Are never rate-limited by token budgets allocated to lower tiers

## Warm-up

At server startup, before accepting requests:

```csharp
if (options.Warmup.Enabled)
{
    // Trigger JIT compilation of hot paths
    var dummyTokens = tokenizer.Encode("The quick brown fox");
    for (int i = 0; i < options.Warmup.Iterations; i++)
        await engine.GenerateAsync(dummyTokens, maxTokens: 16);

    // Pre-load CUDA kernels, cuBLAS handles
    // Pre-compute RoPE tables, tokenizer trie
}
```

Configuration: `WarmupOptions { Enabled, DummyPromptLength, Iterations }`.

Ensures first real request doesn't pay JIT compilation or CUDA kernel loading penalties.

## Health & Readiness

- `GET /health` — Returns 200 when server is running.
- `GET /ready` — Returns 200 only after warm-up completes and model is loaded. Used by load balancers.

## Security

**dotLLM's server is a development/local tool.** It has no authentication, no TLS, and permissive CORS. Do not expose it to the internet without a reverse proxy.

### Binding

The server binds to `localhost` by default. To expose externally, pass `--host 0.0.0.0` — but only behind a reverse proxy (nginx, Caddy, Traefik) that provides TLS and authentication.

### Authentication

No built-in auth. For network-exposed deployments, configure your reverse proxy to require `Authorization: Bearer <key>` headers.

### CORS

Default policy is permissive (`AllowAnyOrigin`) for local Chat UI development. For production, restrict origins via your reverse proxy.

### Dangerous Endpoints

- `POST /v1/models/load` — loads arbitrary GGUF files from disk
- `POST /v1/config` — changes sampling parameters

These are designed for the local Chat UI workflow and must not be internet-exposed.

## Concurrency

The server has two execution paths and picks per-request:

1. **Continuous-batch scheduler path (default for paged-KV serving)**. When `--paged` is on (the default for `serve`) and no speculative-decoding draft model is loaded, `ServerStartup` constructs a `ContinuousBatchSchedulerService` per loaded model and starts its `RunLoopAsync` on a background task tied to `IHostApplicationLifetime.ApplicationStopping`. `/v1/chat/completions` and `/v1/completions` route non-streaming requests through `EnqueueAsync` — multiple concurrent requests pipeline through a single `IModel.ForwardBatch` dispatch per scheduler iteration. The startup log prints `Continuous-batch scheduler active` when this path is engaged.
2. **Single-request gate path (fallback)**. Streaming requests, LoRA-adapter requests, logprob-capturing requests, and any backend without a paged KV-cache factory (CUDA, hybrid GPU, quantized KV) keep using the original `SemaphoreSlim(1, 1)` gate via `ServerState.ExecuteAsync`. Requests serialize FIFO. The startup log prints `Single-request mode — requests processed sequentially` when this is the only path.

### Scheduler tuning

`ContinuousBatchSchedulerOptions`:

| Option | Default | Meaning |
|--------|---------|---------|
| `MaxActiveSequences` | 64 | Slot cap. KV-cache pressure is the hard limit; this is a soft upper bound for batch-formation cost. |
| `MaxPrefillTokensPerStep` | 0 (disabled) | Chunked-prefill cap. When non-zero, no single Step iteration prefills more than this many tokens, even if a long prompt has more to feed. Decode tokens of already-decoding sequences keep running every step regardless — prevents head-of-line blocking. |
| `ReserveBlocksPerSequence` | 0 (disabled) | Admission KV-pressure gate: skip admission when `pagedPool.FreeBlocks < ReserveBlocksPerSequence`. |

### Engine telemetry providers

Once a `ContinuousBatchSchedulerService` is constructed, it wires the observable gauges that
`EngineTelemetry` exposes:

- `dotllm.engine.request.queue_depth` → `Inner.QueueDepth + Inner.ActiveCount` (so saturation is visible — pure queue depth would underreport when sequences are already admitted).
- `dotllm.engine.kvcache.utilization` → `1.0 - FreeBlocks / TotalBlocks` of the underlying paged pool (when present).

Both providers are cleared back to `null` on `Service.Dispose` / model swap so the gauges return to their `-1` sentinel.

## Request Validation

Both `/v1/chat/completions` and `/v1/completions` validate inputs before inference:

| Check | Limit | Response |
|-------|-------|----------|
| Empty messages array | 0 | 400 `"messages array must not be empty"` |
| Messages count | > 1024 | 400 `"messages array exceeds maximum of 1024"` |
| Empty prompt (completions) | empty/null | 400 `"prompt must not be empty"` |
| `max_tokens` | &le; 0 | 400 `"max_tokens must be a positive integer"` |
| Prompt token count | &ge; `MaxSequenceLength` | 400 `"prompt (N tokens) exceeds model context length (M)"` |
| `prompt_tokens + max_tokens` | > `MaxSequenceLength` | `max_tokens` silently clamped to remaining context |
