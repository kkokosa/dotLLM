# Anthropic Messages API — dotLLM

dotLLM's server exposes an **Anthropic-compatible Messages API** alongside the
OpenAI-compatible surface ([SERVER.md](SERVER.md)). Clients and SDKs written for
the Anthropic Messages API (`anthropic` Python/TypeScript SDKs, anything that
targets `POST /v1/messages`) can point at a running dotLLM server unchanged.

The engine, tokenizer, chat-template, sampler and tool-calling pipeline are
shared verbatim with the OpenAI endpoints — this layer only reshapes the wire
format. Implementation: `MessagesEndpoint`, `AnthropicConverter`, and the
`Anthropic*` DTOs in `DotLLM.Server`.

Reference: <https://docs.anthropic.com/en/api/messages>

## Endpoints

### `POST /v1/messages`

Primary endpoint. Accepts the Anthropic Messages request format; supports both
non-streaming (JSON) and streaming (named SSE events).

**Request body**:
```json
{
  "model": "llama-3-8b-q4_k_m",
  "max_tokens": 256,
  "system": "You are helpful.",
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
    {"role": "user", "content": "What's the weather?"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "stop_sequences": ["\n\nHuman:"],
  "tools": [
    {"name": "get_weather", "description": "Get weather",
     "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}
  ],
  "tool_choice": {"type": "auto"},
  "stream": false
}
```

- `max_tokens` is **required** (per the Anthropic spec). Missing/`<= 0` → `400`.
- `system` is a top-level string **or** an array of `{"type":"text","text":"..."}`
  blocks; it becomes a leading `system` message in the chat template.
- Each message `content` is a string **or** an array of content blocks
  (`text`, `tool_use`, `tool_result`).
- `tool_choice`: `{"type":"auto"}`, `{"type":"any"}` (→ required),
  `{"type":"none"}`, or `{"type":"tool","name":"..."}`.
- `messages[].role` must be `user` or `assistant`; any other role → `400`.
  (The top-level `system` field is the only way to set a system prompt.)
- `image` content blocks are not yet supported (no multimodal pipeline).
- `lora_adapter` is **not** honoured by this endpoint — per-request adapter
  selection is not implemented on the server yet (the OpenAI surface does not
  honour it either). Unknown fields are ignored, not rejected.

**Response** (non-streaming):
```json
{
  "id": "msg_...",
  "type": "message",
  "role": "assistant",
  "model": "llama-3-8b-q4_k_m",
  "content": [{"type": "text", "text": "It's sunny."}],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {"input_tokens": 15, "output_tokens": 8}
}
```

When tool calls are detected, `content` contains `tool_use` blocks and
`stop_reason` is `"tool_use"`:
```json
{
  "content": [
    {"type": "tool_use", "id": "toolu_...", "name": "get_weather",
     "input": {"city": "Paris"}}
  ],
  "stop_reason": "tool_use"
}
```

### `POST /v1/messages/count_tokens`

Returns the prompt token count for a would-be request. Same body as
`/v1/messages` (without requiring `max_tokens`).

**Response**: `{"input_tokens": 42}`

## Streaming

With `"stream": true`, the response is a sequence of **named** SSE events
(`event: <type>\ndata: <json>\n\n`):

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","type":"message","role":"assistant","model":"...","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":15,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"It's"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":15,"output_tokens":8}}

event: message_stop
data: {"type":"message_stop"}
```

Tool calls detected during streaming are emitted after the text block closes, as
additional `tool_use` content blocks (`content_block_start` →
`content_block_delta` with `input_json_delta` → `content_block_stop`) at index
`1+`, and `stop_reason` becomes `"tool_use"`.

## Mapping reference

| Anthropic field | dotLLM engine |
|-----------------|---------------|
| `system` (string/array) | leading `system` `ChatMessage` |
| message `content` string | `ChatMessage.Content` |
| `text` block | concatenated into `ChatMessage.Content` |
| `tool_use` block (assistant) | `ChatMessage.ToolCalls` (`ToolCall`) |
| `tool_result` block (user) | separate `tool`-role `ChatMessage` keyed by `tool_use_id` |
| `tools[].input_schema` | `ToolDefinition.ParametersSchema` |
| `tool_choice` `auto`/`any`/`none`/`tool` | `ToolChoice.Auto`/`Required`/`None`/`Function` |
| `stop_sequences` | `InferenceOptions.StopSequences` |

| dotLLM `FinishReason` | Anthropic `stop_reason` |
|-----------------------|-------------------------|
| `Stop` (EOS / template stop) | `end_turn` |
| `Stop` (caller `stop_sequences` matched) | `stop_sequence` (+ `stop_sequence` field) |
| `Length` | `max_tokens` |
| `ToolCalls` | `tool_use` |

## Errors

Errors use the Anthropic envelope:
```json
{"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens: field required"}}
```

| Condition | HTTP | `error.type` |
|-----------|------|--------------|
| No model loaded | 503 | `api_error` |
| Empty `messages`, missing/invalid `max_tokens`, bad LoRA name | 400 | `invalid_request_error` |
| Prompt exceeds context window | 400 | `invalid_request_error` |

## Limitations

- **Streaming tool calls** are detected post-generation (the engine parses tool
  calls from the full output), so `tool_use` blocks are emitted at the end of the
  stream rather than incrementally — matching the OpenAI streaming endpoint's
  post-hoc detection.
- **`image` / multimodal content blocks** are not supported.
- The same single-request serialization, prompt caching, and validation rules as
  the OpenAI endpoints apply (see [SERVER.md](SERVER.md)).
