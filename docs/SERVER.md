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
