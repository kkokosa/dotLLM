# LoRA Adapter Support — dotLLM

## Overview

LoRA (Low-Rank Adaptation) enables fine-tuned model behaviors without modifying base weights. Multiple adapters can coexist on the same base model, with per-request adapter selection.

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
- **SafeTensors**: Primary format. Adapter weights as `{layer_name}.lora_A.weight` and `{layer_name}.lora_B.weight`.
- **GGUF**: Possible future support for quantized adapters.

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
