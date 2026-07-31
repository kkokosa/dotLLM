# Microsoft.Extensions.AI integration — dotLLM

`DotLLM.Extensions.AI` makes dotLLM a native, **in-process** backend for the
**Microsoft Agent Framework** (MAF) and the broader `Microsoft.Extensions.AI`
(MEAI) ecosystem by implementing **`IChatClient`** — the single interface that
MAF agents, the function-invocation middleware, response caching, OpenTelemetry,
and DI all consume.

Implementation: `DotLLMChatClient` (the adapter) and `ChatClientMapping` (the
pure type translation). The package depends only on `DotLLM.Engine` and
`Microsoft.Extensions.AI.Abstractions`; it targets .NET 10 and adds no
reflection-based serialization (AOT-clean).

## Usage

```csharp
using DotLLM.Extensions.AI;
using Microsoft.Extensions.AI;

// `generator` (TextGenerator) and `chatTemplate` (IChatTemplate) come from
// loading a model with the dotLLM engine; `toolCallParser` is optional.
IChatClient client = new DotLLMChatClient(generator, chatTemplate, modelId: "phi-4-q4_k_m", toolCallParser);

// Use directly as a Microsoft.Extensions.AI chat client...
ChatResponse response = await client.GetResponseAsync("Tell me a joke.");

// ...or as a Microsoft Agent Framework agent (zero glue):
AIAgent agent = client.CreateAIAgent(instructions: "You are concise.", name: "Assistant");
var reply = await agent.RunAsync("Summarize the plot of Hamlet.");

await foreach (var update in client.GetStreamingResponseAsync("Stream this."))
    Console.Write(update.Text);
```

## Integration paths

| Path | How | When |
|------|-----|------|
| **Native `IChatClient`** (this package) | Wrap `TextGenerator` in-process — no HTTP, shares mmap'd weights, surfaces usage/timings directly | Embedded/edge .NET agent apps; highest fidelity & lowest latency |
| **OpenAI-compatible HTTP** | Point `Microsoft.Extensions.AI.OpenAI` at dotLLM's `/v1/` base URL (`OpenAIClient ... .AsIChatClient()`) | Out-of-process / language-agnostic consumers (the server already exposes this) |

Both expose the same `IChatClient` surface to MAF, so agent code is identical.

## Mapping reference

**Request (Microsoft.Extensions.AI → engine)**

| MEAI | dotLLM |
|------|--------|
| `IEnumerable<ChatMessage>` (System/User/Assistant/Tool) | `ChatMessage[]` (chat-template rendered to a prompt) |
| `TextContent` | concatenated message content |
| `FunctionCallContent` | assistant `ToolCall` |
| `FunctionResultContent` | `tool`-role message keyed by `CallId` |
| `ChatOptions.{Temperature,TopP,TopK,MaxOutputTokens,StopSequences,Seed}` | `InferenceOptions` |
| `ChatOptions.ResponseFormat` (Json / JSON-schema) | `ResponseFormat.JsonObject` / `JsonSchema` |
| `ChatOptions.Tools` (`AIFunction`) | `ToolDefinition[]` (via the chat template) |
| `AdditionalProperties["min_p" / "repetition_penalty"]` | `InferenceOptions.MinP` / `RepetitionPenalty` (dotLLM-specific knobs) |

**Response (engine → Microsoft.Extensions.AI)**

| dotLLM | MEAI |
|--------|------|
| generated text | `TextContent` |
| detected `ToolCall`s | `FunctionCallContent` (text dropped, `FinishReason = ToolCalls`) |
| `FinishReason` (Stop/Length/ToolCalls) | `ChatFinishReason` (Stop/Length/ToolCalls) |
| prompt/generated token counts | `UsageDetails` (Input/Output/Total) |
| streaming `GenerationToken` | `ChatResponseUpdate` (`text_delta`) |

## Notes / limitations

- The underlying `TextGenerator` is stateful and single-request; calls are
  serialized through an internal gate (streaming holds it for the enumeration),
  matching the dotLLM server's single-request model.
- Streaming tool calls are detected post-generation (like the server endpoints),
  so `FunctionCallContent` is emitted in the final update rather than incrementally.
  Consequently, **when a tool-call parser is configured and the request carries tools,
  text is buffered instead of streamed** and delivered in that same final update. Streaming
  it would leak the model's raw tool-call syntax to the caller, and would make the coalesced
  stream (`ToChatResponse()`) disagree with `GetResponseAsync`, which drops text when tool
  calls are present. Requests without tools (or without a parser) stream token-by-token as
  usual.
- `GetResponseAsync` offloads the synchronous `TextGenerator.Generate` to the thread pool so
  it does not block the caller. The `CancellationToken` therefore only gates scheduling; it
  cannot abort a generation already in flight. Use `GetStreamingResponseAsync`, which
  cancels cooperatively between decode steps, when mid-generation cancellation matters.
- `ChatOptions.FrequencyPenalty`/`PresencePenalty` have no direct engine
  equivalent and are not applied; use `RepetitionPenalty` via `AdditionalProperties`.
- For automatic tool execution, wrap the client with MEAI's
  `FunctionInvokingChatClient` — no custom loop needed.
