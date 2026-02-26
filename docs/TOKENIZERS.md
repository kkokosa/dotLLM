# Tokenizers & Chat Templates — dotLLM

## Tokenizer Types

### tiktoken-style BPE (Llama 3, GPT-4)

1. **Regex pre-tokenization**: Split input using a compiled regex pattern that separates words, numbers, whitespace, punctuation. Pattern is model-specific (e.g., Llama 3 uses a complex pattern handling Unicode categories).
2. **Byte-level BPE**: Each pre-token is converted to bytes, then BPE merges are applied using a **priority queue** (merge with lowest rank first). This is 3-6× faster than the iterative pair-counting approach.
3. **Vocabulary**: Direct byte-pair → token ID mapping.

Implementation: Trie for prefix matching, compiled regex. Vocabulary loaded from GGUF `tokenizer.ggml.tokens` + `tokenizer.ggml.merges`.

### SentencePiece BPE (Llama 2)

- Unicode code-point level (not byte-level).
- Space represented as `▁` (U+2581) prepended to words.
- Token scores (float) determine merge priority.
- Protobuf `.model` files (but GGUF embeds the vocabulary directly).

Vocabulary from GGUF: `tokenizer.ggml.tokens` + `tokenizer.ggml.scores`.

### HuggingFace tokenizer.json

JSON format containing: model type, vocabulary, merges, pre-tokenizer config, normalizer, post-processor, added tokens. Full specification of the tokenization pipeline.

Used when loading models from SafeTensors (which don't embed tokenizer in the weight file).

## ITokenizer Interface

```
ITokenizer:
  Encode(text) → int[]
  Decode(tokenIds) → string
  DecodeToken(tokenId) → string
  VocabSize → int
  BosTokenId → int
  EosTokenId → int
  CountTokens(text) → int   // Fast count without full encode
```

## Chat Template Engine

### Purpose

Models require specific prompt formatting. The OpenAI API sends `messages[]` — the engine must format them correctly.

### Template Format

Templates use **Jinja2 syntax** (HuggingFace standard), stored in:
- GGUF: `tokenizer.chat_template` metadata key
- HuggingFace: `tokenizer_config.json` → `chat_template` field

### Required Jinja2 Subset

Full Jinja2 is not needed. Required features:
- Variable interpolation: `{{ message.content }}`
- For loops: `{% for message in messages %}`
- Conditionals: `{% if message.role == "system" %}`
- String filters: `{{ text | trim }}`, `{{ text | strip }}`
- `raise_exception("error message")`
- Basic expressions and comparisons

### IChatTemplate Interface

```
IChatTemplate:
  Apply(messages: IReadOnlyList<ChatMessage>, options: ChatTemplateOptions) → string
```

```
ChatMessage:
  Role: string ("system" | "user" | "assistant" | "tool")
  Content: string
  ToolCalls: ToolCall[]?     (for assistant messages with tool calls)
  ToolCallId: string?        (for tool result messages)

ChatTemplateOptions:
  AddGenerationPrompt: bool  (append assistant turn prefix)
  Tools: ToolDefinition[]?   (for tool-calling models)
```

### Known Template Formats

**Llama 3**:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_message}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

**ChatML** (Qwen, many others):
```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
```

**Mistral**:
```
[INST] {user_message} [/INST]
```

### Fallback

If no template found in model metadata, use configurable default (ChatML). Log a warning.

## Tool Calling Protocol

### Flow

1. Request includes `tools` definitions (name, description, parameter JSON schema).
2. Chat template formats tool definitions into prompt.
3. Model generates tool call JSON: `{"name": "func", "arguments": {...}}`.
4. **Constrained decoding** ensures valid JSON matching the tool schema.
5. Server detects tool call, returns `finish_reason: "tool_calls"`.
6. Client executes tool, sends result as `tool` role message.
7. Template formats result; model generates final response.

### IToolCallParser

```
IToolCallParser:
  TryParse(generatedText) → ToolCall[]?
  IsToolCallStart(text) → bool
```

Models signal tool calls differently:
- Special tokens (`<|tool_call|>`, `<|python_tag|>`)
- JSON patterns in output
- Model-specific formats

The parser is associated with the chat template — each template knows its model's tool calling convention.