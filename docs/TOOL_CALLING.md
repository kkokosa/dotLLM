# Tool Calling — dotLLM

## Overview

Tool calling (function calling) enables models to invoke external tools by generating structured JSON that the caller can execute and feed back as context. This is the foundation for agentic workflows — the model decides *which* function to call and *what arguments* to pass, the runtime executes it, and the model incorporates the result.

dotLLM's tool calling integrates three subsystems: **chat templates** (formatting tool definitions into the prompt), **constrained decoding** (guaranteeing valid JSON for tool arguments), and **model-specific parsers** (extracting structured tool calls from model output).

## End-to-End Flow

```
┌─────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Client  │────►│ Chat Template│────►│  Text Gen +  │────►│  Tool Call   │
│ (tools,  │     │  (formats    │     │  Constrained │     │  Parser      │
│ messages)│     │  tools into  │     │  Decoding     │     │  (extracts   │
│          │     │  prompt)     │     │  (optional)  │     │  ToolCall[]) │
└─────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                  │
     ┌────────────────────────────────────────────────────────────┘
     │  finish_reason: "tool_calls"
     ▼
┌─────────┐     ┌──────────────┐     ┌──────────────┐
│  Client  │────►│   Execute    │────►│ Add tool     │──── (loop back to
│ receives │     │   Tools      │     │ results to   │      Chat Template)
│ tool calls│    │   Locally    │     │ messages     │
└─────────┘     └──────────────┘     └──────────────┘
```

**Step by step:**

1. Request includes `tools` definitions (name, description, parameter JSON schema) and `tool_choice`.
2. `IChatTemplate.Apply(messages, { Tools = tools })` injects tool definitions into the prompt using the model's Jinja2 template.
3. If `tool_choice` is `required` or a specific function, `ToolCallSchemaBuilder` generates a JSON Schema and `JsonSchemaConstraint` guarantees valid output via constrained decoding.
4. Model generates text. `IToolCallParser.TryParse(output)` extracts structured `ToolCall[]`.
5. `ToolCallDetector` enriches `InferenceResponse` with parsed calls, sets `FinishReason.ToolCalls`.
6. Client executes tools, adds results as `tool` role messages with `ToolCallId`.
7. Template formats results; model generates the final response incorporating tool output.

## Core Types

### ToolDefinition

```
ToolDefinition:
  Name: string              // Function name (e.g., "get_weather")
  Description: string       // Human-readable description
  ParametersSchema: string  // JSON Schema for function parameters
```

### ToolCall

```
ToolCall:
  Id: string                // Unique call ID (e.g., "call_0")
  FunctionName: string      // Which function was called
  Arguments: string         // JSON string of arguments
```

### ToolChoice

Discriminated union controlling how the model selects tool calls. Follows the OpenAI API convention.

```
ToolChoice:
  Auto        // Model decides freely (default when tools present)
  None        // Don't call tools — text only
  Required    // Must call at least one tool
  Function    // Must call a specific function by name
```

| `tool_choice` | Constrained decoding | Parser runs | Use case |
|---------------|---------------------|-------------|----------|
| `Auto` | No | Yes | Model freely decides; detect tool calls post-hoc |
| `None` | No | No | Tools in context for reference only |
| `Required` | Yes — `anyOf` schema | Yes | Force a tool call; guaranteed valid JSON |
| `Function(name)` | Yes — single-tool schema | Yes | Force a specific function |

### ChatMessage (tool-related fields)

```
ChatMessage:
  Role: string           // "system" | "user" | "assistant" | "tool"
  Content: string        // Text content
  ToolCalls: ToolCall[]? // Assistant messages: tool invocations
  ToolCallId: string?    // Tool result messages: which call this answers
```

## IToolCallParser Interface

```
IToolCallParser:
  TryParse(generatedText) → ToolCall[]?   // Extract tool calls from output
  IsToolCallStart(text) → bool            // Detect partial tool call (streaming)
```

Models signal tool calls in different formats. Each parser handles one convention:

### Parser Implementations

| Parser | Marker | JSON Key | Models | Example Output |
|--------|--------|----------|--------|----------------|
| `LlamaToolCallParser` | `<\|python_tag\|>` | `name` + `parameters` | Llama 3.1+ Instruct | `<\|python_tag\|>{"name":"f","parameters":{...}}` |
| `HermesToolCallParser` | `<tool_call>`...`</tool_call>` | `name` + `arguments` | Hermes, Qwen tool-calling | `<tool_call>{"name":"f","arguments":{...}}</tool_call>` |
| `MistralToolCallParser` | `[TOOL_CALLS]` | `name` + `arguments` | Mistral Instruct | `[TOOL_CALLS][{"name":"f","arguments":{...}}]` |
| `GenericToolCallParser` | None (bare JSON) | `name` + `arguments`/`parameters` | Fallback | `{"name":"f","arguments":{...}}` |

### Key Normalization: `parameters` vs `arguments`

Llama models use `"parameters"` for function arguments; the OpenAI API and most other models use `"arguments"`. The shared `ToolCallJsonHelper` normalizes both to the `ToolCall.Arguments` field:

1. Try `"arguments"` key first.
2. Fall back to `"parameters"`.
3. If the value is a string (double-serialized JSON), detect and unwrap it.
4. Generate sequential call IDs (`call_0`, `call_1`, ...) when none provided.

### Parallel Tool Calls

All parsers support parallel tool calls:

- **Llama**: JSON array after `<|python_tag|>` — `[{call1}, {call2}]`
- **Hermes**: Multiple `<tool_call>` blocks
- **Mistral**: JSON array after `[TOOL_CALLS]`
- **Generic**: JSON array with multiple objects

### Graceful Failure

All parsers return `null` on malformed input — they never throw. This is critical because tool call detection runs on every generation when tools are present. Invalid JSON, missing `name` key, unbalanced brackets — all produce `null`, and the response is treated as normal text.

## Parser Auto-Detection

`ToolCallParserFactory.Create(Architecture, chatTemplate?)` selects the appropriate parser via a two-tier heuristic:

**Tier 1 — Template content (highest priority):**
```
Template contains "<tool_call>"       → HermesToolCallParser
Template contains "python_tag"        → LlamaToolCallParser
Template contains "[TOOL_CALLS]"      → MistralToolCallParser
```

**Tier 2 — Architecture fallback:**
```
Architecture.Llama    → LlamaToolCallParser
Architecture.Mistral  → MistralToolCallParser
Architecture.Qwen     → HermesToolCallParser
*                     → GenericToolCallParser
```

Template content takes priority because the template is the source of truth for the model's tool calling convention — the same architecture may have different fine-tunes with different conventions.

## Constrained Decoding for Tool Arguments

When `tool_choice` is `Required` or `Function(name)`, the model output is constrained to valid tool call JSON via the existing `JsonSchemaConstraint` infrastructure (Step 40).

### Schema Generation

`ToolCallSchemaBuilder` synthesizes a JSON Schema from `ToolDefinition[]`:

**Single function** (`BuildForFunction`):
```json
{
  "type": "object",
  "properties": {
    "name": {"const": "get_weather"},
    "arguments": {<tool's parameter schema>}
  },
  "required": ["name", "arguments"],
  "additionalProperties": false
}
```

**Multiple functions** (`BuildForRequired`):
```json
{
  "anyOf": [
    {<schema for tool 1>},
    {<schema for tool 2>}
  ]
}
```

**Parallel calls** (`BuildForParallelCalls`):
```json
{
  "type": "array",
  "items": {"anyOf": [<per-tool schemas>]}
}
```

The `argumentsKey` parameter handles the Llama `"parameters"` vs standard `"arguments"` difference — it's set based on the parser type (Llama parsers use `"parameters"`, all others use `"arguments"`).

### Constraint Integration

The generated schema feeds directly into the existing `ResponseFormat.JsonSchema` → `JsonSchemaConstraint` pipeline:

```
ToolDefinition[] → ToolCallSchemaBuilder.BuildForRequired()
                 → ResponseFormat.JsonSchema { Schema = generatedSchema }
                 → JsonSchemaConstraint (SchemaCompiler, SchemaTracker)
                 → TokenMask per decode step
```

The `SchemaCompiler` already supports `anyOf`, `const`, nested objects, `enum` — no modifications needed. This means tool call arguments are **mathematically guaranteed** to conform to the tool's parameter schema.

### `tool_choice=auto` — No Constraint

When `tool_choice` is `auto`, no schema constraint is applied. The model generates freely and the parser detects tool calls post-hoc. This is because the model may choose to produce text instead of calling a tool — constraining the output format would prevent text-only responses.

## Post-Generation Detection

### ToolCallDetector

Static utility for non-streaming use:

```csharp
var response = generator.Generate(prompt, options);
response = ToolCallDetector.DetectToolCalls(response, parser);

if (response.FinishReason == FinishReason.ToolCalls)
{
    // response.ToolCalls contains parsed tool calls
}
```

Returns the original response unchanged if no tool calls are found (reference equality — no allocation).

### StreamingToolCallAccumulator

For streaming use, accumulates token text and detects tool call boundaries:

```csharp
var accumulator = new StreamingToolCallAccumulator(parser);

await foreach (var token in generator.GenerateStreamingTokensAsync(prompt, options))
{
    bool suppress = accumulator.Append(token.Text);
    if (!suppress)
        Console.Write(token.Text);  // show text to user
    // else: this text is part of a tool call, buffer it
}

// After generation completes:
var toolCalls = accumulator.TryParseCompleted();
```

**Behavior:**
- Before a tool call marker is detected: `Append()` returns `false` — text flows to user.
- Once `IsToolCallStart()` triggers: `Append()` returns `true` for all subsequent text — caller suppresses output.
- After generation: `TryParseCompleted()` extracts the full tool call(s).

## Chat Template Integration

The Jinja2 chat template engine (Step 16) handles tool definitions natively:

### Template Context

`JinjaChatTemplate.BuildContext()` exposes tools in the standard HuggingFace format:

```python
# Available in Jinja template:
tools = [
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather",
      "parameters": {<parsed JSON schema as dict>}
    }
  }
]
```

Tool calls in assistant messages:
```python
message.tool_calls = [
  {
    "id": "call_0",
    "type": "function",
    "function": {"name": "get_weather", "arguments": "{...}"}
  }
]
```

Tool result messages:
```python
message.role = "tool"
message.content = '{"temperature": 22}'
message.tool_call_id = "call_0"
```

### Template Examples

**ChatML with tools** (Qwen, Hermes):
```jinja
{% for message in messages %}
<|im_start|>{{ message.role }}
{% if message.tool_calls %}
{% for tc in message.tool_calls %}
<tool_call>{{ tc | tojson }}</tool_call>
{% endfor %}
{% else %}
{{ message.content }}
{% endif %}
<|im_end|>
{% endfor %}
{% if tools %}
Available tools: {{ tools | tojson }}
{% endif %}
```

**Llama 3.1** uses `<|python_tag|>` and formats tools as a structured system prompt section. The `tojson` filter serializes tool definitions for embedding.

## Multi-Turn Tool Use

A complete multi-turn conversation with tool calling:

```
messages = [
  {role: "system", content: "You are helpful."},
  {role: "user", content: "What's the weather in Paris?"},
]

// Turn 1: Model calls a tool
→ assistant: {tool_calls: [{id: "call_0", name: "get_weather", args: {"location":"Paris"}}]}
→ finish_reason: "tool_calls"

messages += [
  {role: "assistant", content: "", tool_calls: [...]},
  {role: "tool", content: '{"temp":22,"condition":"sunny"}', tool_call_id: "call_0"},
]

// Turn 2: Model uses tool result
→ assistant: "The weather in Paris is 22°C and sunny."
→ finish_reason: "stop"
```

Each turn re-applies the full chat template to the accumulated message history, including tool call and tool result messages.

## CLI Usage

```bash
# Interactive chat with tools
dotllm chat model.gguf --tools @tools.json

# Force tool calling (constrained decoding)
dotllm chat model.gguf --tools @tools.json --tool-choice required

# Force a specific function
dotllm chat model.gguf --tools @tools.json --tool-choice get_weather
```

### Tools JSON Format

`--tools` accepts a JSON array (inline or `@file`). Both flat and OpenAI-style formats supported:

**Flat format:**
```json
[
  {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      },
      "required": ["location"]
    }
  }
]
```

**OpenAI format:**
```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather for a location",
      "parameters": { ... }
    }
  }
]
```

### REPL Commands

| Command | Action |
|---------|--------|
| `/tools` | Display available tool definitions |
| `/clear` | Reset conversation (preserves system prompt) |
| `/exit` | Quit |

When tool calls are detected in the REPL:

```
>>> What's the weather in Paris?

Tool calls detected:
  [call_0] get_weather({"location": "Paris"})
Result for get_weather (Enter to skip):
[tool]>>> {"temperature": 22, "condition": "sunny"}

Generating response with tool results...
The weather in Paris is 22°C and sunny.
```

## Key Files

| File | Purpose |
|------|---------|
| `Core/Configuration/ToolChoice.cs` | `ToolChoice` discriminated union |
| `Tokenizers/ToolCall.cs` | `ToolCall` record |
| `Tokenizers/ToolDefinition.cs` | `ToolDefinition` record |
| `Tokenizers/ChatMessage.cs` | `ChatMessage` with `ToolCalls`, `ToolCallId` |
| `Tokenizers/IToolCallParser.cs` | Parser interface |
| `Tokenizers/ToolCallParsers/LlamaToolCallParser.cs` | Llama 3.1+ parser |
| `Tokenizers/ToolCallParsers/HermesToolCallParser.cs` | Hermes/Qwen parser |
| `Tokenizers/ToolCallParsers/MistralToolCallParser.cs` | Mistral parser |
| `Tokenizers/ToolCallParsers/GenericToolCallParser.cs` | Fallback parser |
| `Tokenizers/ToolCallParsers/ToolCallJsonHelper.cs` | Shared JSON extraction + normalization |
| `Tokenizers/ToolCallParsers/ToolCallParserFactory.cs` | Auto-detection factory |
| `Engine/Constraints/ToolCallSchemaBuilder.cs` | Schema generation from tool definitions |
| `Engine/ToolCallDetector.cs` | Post-generation detection |
| `Engine/StreamingToolCallAccumulator.cs` | Streaming boundary detection |
| `Engine/InferenceResponse.cs` | `ToolCalls` property, `FinishReason.ToolCalls` |
| `Cli/Commands/ChatCommand.cs` | `--tools`, `--tool-choice`, REPL integration |

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Parser per model family | 4 implementations + factory | Models use fundamentally different formats — no single parser handles all |
| Template heuristic over config | Scan template for markers | Template is source of truth; same architecture can have different tool conventions |
| Schema constraint only for required/function | No constraint for auto | Model must be free to produce text instead of tool calls with `auto` |
| Post-generation detection (not in TextGenerator) | `ToolCallDetector` is caller responsibility | Keeps engine minimal; CLI and Server both use it differently |
| `arguments` normalization | `ToolCallJsonHelper` handles both keys | Llama uses `parameters`, everyone else uses `arguments` — normalize once |
| Sequential call IDs | `call_0`, `call_1`, ... | Simple, deterministic; models rarely provide their own IDs |
| Graceful failure | Return `null`, never throw | Parser runs on every generation — exceptions would be disruptive |

## Reference Implementations

- **llama.cpp** — Tool calling via chat template Jinja rendering, `<|python_tag|>` detection
- **vLLM** — `ToolCallParser` with model-specific handlers, guided decoding for tool args
- **Ollama** — Tool support via chat template integration, JSON extraction
- **OpenAI API** — `tool_choice`, `tools`, `finish_reason: "tool_calls"` conventions
