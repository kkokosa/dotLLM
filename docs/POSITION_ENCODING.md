# Position Encoding — dotLLM

## IPositionEncoding Interface

Optional and pluggable. `ModelConfig.PositionEncodingType` selects variant or `None`.

```
IPositionEncoding:
  Apply(Q, K, positions) → (Q_encoded, K_encoded)
  PrecomputeTables(maxSeqLen, config) → void
  InvalidateCache() → void   // When parameters change at runtime
```

## RoPE — Rotary Position Embeddings

Default for Llama, Mistral, Qwen, Phi-3, DeepSeek.

### Mechanism

Rotates pairs of dimensions in Q and K by position-dependent angles:
```
θ_i = θ_base^(-2i/d)
angle = pos × θ_i
q'[2i]   = q[2i] × cos(angle) - q[2i+1] × sin(angle)
q'[2i+1] = q[2i] × sin(angle) + q[2i+1] × cos(angle)
```

### Implementation

1. Pre-compute cos/sin tables `[max_seq_len × head_dim/2]` at model load.
2. At inference: lookup `cos_table[pos]`, `sin_table[pos]`, apply rotation.
3. SIMD: process dimension pairs with `Vector256<float>` — two muls + add/sub per pair.

### RoPEConfig (from GGUF)

```
Theta: float (10000.0 default, Llama 3 = 500000.0)
DimensionCount: int
ScalingType: None | Linear | YaRN | NTK | DynamicNTK
ScalingFactor: float
OrigMaxSeqLen: int
AttnFactor, BetaFast, BetaSlow: float (YaRN-specific)
```

## Scaling Variants

### Linear (Position Interpolation)
`angle = (pos / factor) × θ_i`. Simplest. 4K trained + factor 4 → 16K.

### NTK-Aware
Modifies base: `θ_new = θ × factor^(d/(d-2))`. Preserves local patterns better.

### YaRN
Combines NTK with attention scaling. Per-dimension: high-freq unchanged, low-freq fully scaled, medium interpolated.

### Dynamic NTK
Scales only when `seq_len > orig_max`. Requires recomputing tables dynamically.

### Runtime Context Extension

RoPE params overridable at inference time. When changed:
1. `InvalidateCache()` — clear cos/sin tables
2. Recompute tables with new params
3. **Invalidate KV-cache** — cached K vectors have stale rotation angles

## ALiBi — Attention with Linear Biases

BLOOM, MPT. Adds bias to attention scores, no Q/K modification:
```
scores[h][i][j] += m_h × (j - i)
m_h = 2^(-8h/H)   (per-head slope)
```
No position limit, no tables. Less common in modern models.

## Absolute Learned Embeddings

GPT-2 style. Lookup table `position_emb[pos]` added to token embedding before first layer. Limited to training context length.

## None

No position encoding applied. Set `PositionEncodingType = None`.
