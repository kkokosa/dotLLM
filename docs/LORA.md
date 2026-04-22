# LoRA Adapter Support — dotLLM

## Overview

LoRA (Low-Rank Adaptation) enables fine-tuned model behaviors without modifying base weights. Multiple adapters can coexist on the same base model, with per-request adapter selection.

### Use Cases

| Scenario | What LoRA enables |
|----------|-------------------|
| Domain specialization (legal, medical, finance) | One base model in memory, many domain adapters at ~10–100 MB each |
| Function / tool calling on a base without native support | Tool-calling adapters bolt capability onto vanilla instruct models |
| Per-tenant customization in a multi-tenant server | Each tenant gets its own adapter; server selects per request |
| A/B testing fine-tunes | Deploy N adapters, route via `lora_adapter` parameter — no redeployment |
| Instruction / chat style variants | One base, many personalities |
| Long-context extensions | Drop-in adapters (e.g. Gradient's 1M-token Llama-3 adapter) extend context via rank updates |

## How LoRA Works at Inference

For each adapted linear layer:
```
y = x @ W + α × (x @ B) @ A
```
- `W`: frozen base weight [d_in × d_out]
- `B`: down-projection [d_in × r] (r = rank, typically 8-64)
- `A`: up-projection [r × d_out]
- `α`: scaling factor (usually `alpha / rank`)

The LoRA delta `α(xB)A` adds <5% compute overhead for typical ranks.

## Adapter Loading

### Format Support
- **SafeTensors (HuggingFace PEFT layout)**: Primary format. An adapter is a directory with two files:
  - `adapter_config.json` — `r` (rank), `lora_alpha`, `target_modules[]`, `base_model_name_or_path`, `task_type`.
  - `adapter_model.safetensors` — weights keyed as `base_model.model.{layer_path}.lora_A.weight` / `lora_B.weight` (PEFT convention). Loader strips the `base_model.model.` prefix when mapping to dotLLM layer names.
- **GGUF**: Possible future support for quantized adapters.

### Compatibility Rules

Validated at `LoadAdapter` time; mismatches are rejected early with a clear error:

- **Architecture must match the base** — a Llama-2 LoRA cannot be applied to Llama-3 (different head dims, rope theta, and tokenizer). The loader compares `adapter_config.json:base_model_name_or_path` against `ModelConfig.ArchitectureFamily`.
- **Hidden size and layer count must align** — `A` shape `[r, d_out]` and `B` shape `[d_in, r]` are checked against the target layer's dimensions per layer.
- **Target modules must exist** — every entry in `target_modules[]` (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, etc.) must resolve to a known linear in the loaded architecture.
- **Rank `r` and `alpha`** are adapter-specific, read from `adapter_config.json`, and applied without per-request tuning.

### Adapter Metadata
```
LoraAdapter:
  Name: string
  Rank: int
  Alpha: float
  TargetModules: string[]   (e.g., ["q_proj", "v_proj", "k_proj", "o_proj"])
  Layers: Dictionary<string, (A_tensor, B_tensor)>
```

## IAdapterManager Interface

```
IAdapterManager:
  LoadAdapter(name, path) → void
  UnloadAdapter(name) → void
  GetAdapter(name) → LoraAdapter?
  ListAdapters() → IReadOnlyList<string>
```

## Runtime Application

### Per-Request Adapter Selection
Each request specifies `lora_adapter: "adapter_name"` (or null for base model). The `RequestContext` carries the active adapter ID through the inference pipeline.

### Adapted Layer Forward Pass
```csharp
public Tensor Forward(Tensor input, RequestContext ctx)
{
    var output = input.MatMul(baseWeight);  // Always compute base

    if (ctx.AdapterId is not null &&
        adapterManager.GetAdapter(ctx.AdapterId) is { } adapter &&
        adapter.Layers.TryGetValue(layerName, out var lora))
    {
        var delta = input.MatMul(lora.B).MatMul(lora.A);
        output.AddInPlace(delta, scale: lora.Alpha / lora.Rank);
    }

    return output;
}
```

## Multi-Adapter Batching

In continuous batching, different sequences may use different adapters:

1. **Group by adapter**: Partition batch into groups sharing the same adapter (including "no adapter").
2. **Base matmul**: Batched across all sequences (same base weight).
3. **LoRA delta**: Computed per adapter group, added to corresponding outputs.

This is less efficient than uniform batching but the LoRA matmuls are small (low rank) so the overhead is modest.

## Design Decisions

- **No weight merging**: Adapters are never merged into base weights (`W' = W + αBA`). This enables instant switching and concurrent adapters. Trade-off: small per-layer overhead vs. large flexibility gain.
- **Adapter caching**: Loaded adapters kept in memory (GPU or CPU). Small footprint (10-100MB typical for 7B model adapter).
- **Hot loading**: Adapters can be loaded/unloaded at runtime without restarting the server.

## User-Facing Usage

### CLI
```
dotllm chat --model llama-3-8b.gguf \
            --lora /path/to/finance-adapter/ \
            --prompt "Explain EBITDA"
```
Multiple `--lora name=path` flags register adapters under aliases; `--lora-default name` selects the default for the session.

### HTTP (`/v1/chat/completions`)
```json
{
  "model": "llama-3-8b-instruct",
  "lora_adapter": "finance",
  "messages": [{"role": "user", "content": "Explain EBITDA"}]
}
```
Omit `lora_adapter` (or pass `null`) to hit the base model. Unknown adapter names return HTTP 400.

### Admin endpoints
- `POST /v1/loras` — register `{ name, path }` (path resolved server-side; server has no outbound fetch).
- `DELETE /v1/loras/{name}` — unload.
- `GET /v1/loras` — list loaded adapters with rank, alpha, target modules.

## Where to Get Adapters

**Primary source: Hugging Face Hub.** Adapters are standalone repos containing `adapter_config.json` + `adapter_model.safetensors`. Find them via the base-model's "Adapters" tab or by searching `lora` tagged with the base model name.

Download a specific adapter:
```
huggingface-cli download <repo> adapter_config.json adapter_model.safetensors \
    --local-dir ./adapters/<name>/
```

### Known-good examples (compatible with dotLLM's SafeTensors loader)

**Llama 3:**
- `unclecode/llama3-function-call-lora-adapter-240424` — function calling
- `anamikac2708/Llama3-8b-LoftQ-finetuned-investopedia-Lora-Adapters` — finance
- `beratcmn/Llama3-ChatQA-1.5-8B-lora` — conversational QA
- `cognitivecomputations/Llama-3-70B-Gradient-1048k-adapter` — 1M-token context extension

**Mistral / Qwen / others:** search Hub for `{base-model}-lora`. PEFT's default target modules (`q_proj`, `v_proj`) match dotLLM's `TargetModules` field directly — no config translation needed.

### Training your own

Output of these toolchains loads directly into dotLLM with no conversion:
- `huggingface/peft` — the reference PEFT library
- `modelscope/ms-swift` — supports Qwen, Mistral, Llama, DeepSeek, many MLLMs
- `unsloth` — faster single-GPU LoRA training, same output format

All produce an `adapter_config.json` + `adapter_model.safetensors` pair.
