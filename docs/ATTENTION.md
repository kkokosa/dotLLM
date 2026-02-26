# Attention Mechanisms — dotLLM

## IAttentionMechanism Interface

```
IAttentionMechanism:
  Forward(Q, K, V, positionEncoding, kvCache, mask, hooks) → output
  CreateKvCache(config) → IKvCache
```

## Grouped-Query Attention (GQA)

Single implementation covers three variants via `num_kv_heads`:

| Config | Variant | Models |
|--------|---------|--------|
| `kv_heads == attn_heads` | MHA | GPT-2, older models |
| `kv_heads == 1` | MQA | Falcon, PaLM |
| `1 < kv_heads < attn_heads` | GQA | Llama 2/3, Mistral, Qwen2 |

### Forward Pass

1. Project: Q = x @ W_q, K = x @ W_k, V = x @ W_v
2. Reshape to heads: Q[batch, num_heads, seq, head_dim], K/V[batch, kv_heads, seq, head_dim]
3. Apply position encoding (RoPE) to Q and K
4. Update KV-cache (append K, V)
5. GQA broadcast: expand KV heads by `group_size = num_heads / kv_heads`
6. Scores = (Q @ K.T) / sqrt(head_dim) + mask
7. Weights = softmax(scores)
8. Output = weights @ V → reshape → output @ W_o

### Sliding Window

Mask modifier, not separate mechanism. Limits attention to `[pos - window_size, pos]`. KV-cache evicts older entries. Configured via `ModelConfig.SlidingWindowSize`.

## Multi-head Latent Attention (MLA)

DeepSeek-V2/V3. Compresses KV into low-rank latent space.

1. Compress: `c_kv = x @ W_dkv` → latent_dim (e.g., 512 vs 4096)
2. Store `c_kv` in cache (not full K, V — 8-16× smaller)
3. Decompress at attention time: `K = c_kv @ W_uk`, `V = c_kv @ W_uv`
4. Separate RoPE handling for rope and non-rope dimensions
5. Standard attention computation

Requires its own `IAttentionMechanism` impl with `LatentKvCache`.

## IAttentionStrategy — Kernel Selection

```
IAttentionStrategy:
  ComputeAttention(Q, K, V, mask, scale) → output
  SupportsPagedKvCache → bool
  RequiredComputeCapability → int?
```

| Strategy | Memory | When |
|----------|--------|------|
| **Naive** | O(N²) | Reference, fallback, short sequences |
| **Flash Attention 2** | O(N) | GPU SM80+ (Ampere). Tiled in SRAM, online softmax. 2-7× speedup |
| **Flash Attention 3** | O(N) | GPU SM90+ (Hopper). Async TMA, FP8 |
| **CPU Tiled** | O(N) | CPU. Tiles fit L2 cache |
| **Paged Flash** | O(N) | Flash + non-contiguous KV blocks (PagedAttention) |

Backend advertises capabilities; engine selects best strategy.
