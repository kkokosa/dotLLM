# RoPE — Implementation Notes

> Implementation file: `RoPE.cs`
> Test file: `tests/DotLLM.Tests.Unit/Cpu/Kernels/RoPETests.cs`
> Used by: every transformer model that uses Rotary Position Embeddings — Llama 1/2/3, Mistral, Phi, Qwen, SmolLM, and most modern open models.

---

## What is RoPE?

Rotary Position Embedding (RoPE), introduced by Su et al. (2021), is the dominant position encoding scheme in modern language models. Unlike absolute embeddings (which add a fixed vector to each token) or ALiBi (which subtracts a bias from attention scores), RoPE encodes position by **rotating** the query and key vectors in the complex plane before the dot-product attention computation.

The key insight: if you rotate Q by angle `m·θ` and K by angle `n·θ`, their dot product `Q·K` naturally depends on `(m − n)·θ` — the **relative** distance between positions. Position information is therefore baked into attention scores without any explicit positional term in the attention formula.

### The rotation formula

For a head vector `q` of dimension `d`, RoPE treats consecutive pairs of dimensions `(q[2i], q[2i+1])` as the real and imaginary parts of a complex number, and rotates each pair by a position- and dimension-dependent angle:

```
q'[2i]   = q[2i]   · cos(m · θᵢ) − q[2i+1] · sin(m · θᵢ)
q'[2i+1] = q[2i]   · sin(m · θᵢ) + q[2i+1] · cos(m · θᵢ)
```

where:
- `m` is the absolute position of the token in the sequence
- `θᵢ = base^(−2i / d)` is the per-dimension angular frequency
- `base` is the `theta` hyperparameter (10000 for Llama 2, 500000 for Llama 3)

Written more compactly for a full head vector of dimension `d`:

```
   [cos(m·θ₀)  −sin(m·θ₀)    0          0       ···]   [q[0]  ]
   [sin(m·θ₀)   cos(m·θ₀)    0          0       ···]   [q[1]  ]
   [   0           0       cos(m·θ₁)  −sin(m·θ₁) ···] · [q[2]  ]
   [   0           0       sin(m·θ₁)   cos(m·θ₁) ···]   [q[3]  ]
   [  ···         ···         ···        ···      ···]   [ ···  ]
```

Each pair `(q[2i], q[2i+1])` is multiplied by a 2×2 rotation matrix parameterized by `m·θᵢ`. Crucially, the pairs are **independent** — the rotation of pair `i` does not affect pair `j`.

### Why the dot product encodes relative position

Consider query `q` at position `m` and key `k` at position `n`, both rotated by RoPE:

```
Rₘq · Rₙk = (Rₘq)ᵀ(Rₙk)
           = qᵀ Rₘᵀ Rₙ k
           = qᵀ R_{n−m} k
```

Since rotation matrices satisfy `Rₘᵀ Rₙ = R_{n−m}`, the dot product depends only on the **relative offset** `n − m`, not the absolute positions. This is what makes RoPE a relative position encoding despite being applied to absolute positions.

### Why lower dimensions rotate faster

The frequencies `θᵢ = base^(−2i / d)` decrease geometrically with `i`:

| Dimension pair `i` | Frequency `θᵢ` (base=10000, d=128) | Period |
|--------------------|--------------------------------------|--------|
| 0 (first pair) | 1.0 | 2π ≈ 6.3 tokens |
| 16 | 0.1 | 63 tokens |
| 32 | 0.01 | 628 tokens |
| 48 | 0.001 | 6283 tokens |
| 63 (last pair) | ≈ 0.00084 | ≈ 7470 tokens |

Low-index pairs complete many full rotations across a typical context window and thus encode **local** positional differences precisely. High-index pairs rotate very slowly and give the model a coarse sense of **global** position. The combination allows attention to capture both nearby and long-range relationships simultaneously.

---

## Two-Phase Execution

The kernel is structured in two phases, separated for performance:

### Phase 1 — Precompute frequency tables (once at model load)

`PrecomputeFrequencyTable(maxSeqLen, headDim, theta, cosTable, sinTable)` fills two flat arrays of shape `[maxSeqLen, halfDim]`, indexed as `table[pos * halfDim + i]`. These tables are computed once after model loading and reused across all inference calls.

**Why flatten into two arrays?** Storing `cos` and `sin` separately (rather than interleaved or as complex numbers) lets `ApplyRotation` load `cosVec` and `sinVec` as contiguous 128-bit loads — zero scatter/gather overhead.

**Why `halfDim = headDim / 2`?** Each pair `(2i, 2i+1)` shares a single angle, so we only need one cos/sin entry per pair, not one per dimension.

### Phase 2 — Apply rotation at inference time

`Execute(q, k, positions, ...)` iterates over the token sequence and calls `ApplyRotation` once per head per token. Each call rotates all `halfDim` pairs of a single head vector in-place using the pre-computed table entries for that token's position.

---

## Implementation: `PrecomputeFrequencyTable`

```csharp
public static void PrecomputeFrequencyTable(int maxSeqLen, int headDim, float theta,
                                             Span<float> cosTable, Span<float> sinTable)
```

### The frequency separation optimization

A naïve implementation calls `MathF.Pow(theta, 2f * i / headDim)` for every `(pos, i)` combination — `maxSeqLen × halfDim` calls total. For `headDim=128` (halfDim=64) and `maxSeqLen=4096`, that is **262,144 calls** to `MathF.Pow`.

The fix: the angular frequency `θᵢ = 1 / theta^(2i/d)` depends **only on `i`**, not on `pos`. We precompute the `halfDim` frequency values once, then for each `(pos, i)` compute just `angle = pos * freqs[i]` followed by `MathF.Cos(angle)` and `MathF.Sin(angle)`.

```
freqs[i] = 1 / theta^(2i / headDim)        ← halfDim calls to MathF.Pow (expensive)

for each pos:
  for each i:
    angle = pos * freqs[i]                  ← cheap multiply
    cosTable[pos * halfDim + i] = cos(angle)  ← sin/cos still O(seqLen × halfDim)
    sinTable[pos * halfDim + i] = sin(angle)
```

Result: `MathF.Pow` calls reduced from **O(seqLen × halfDim)** to **O(halfDim)** — a 4096× reduction for Llama configs.

### The scratch buffer

The `freqs` array is small (`halfDim * 4` bytes — typically 256 bytes for headDim=128). It is stackalloc'd when `halfDim * sizeof(float) <= 8192` (always true in practice — headDim would have to exceed 16,384 to spill to ArrayPool). The split between stackalloc and ArrayPool paths uses the same helper `FillTables` to avoid code duplication.

---

## Implementation: `ApplyRotation`

```csharp
[MethodImpl(MethodImplOptions.AggressiveInlining)]
public static void ApplyRotation(Span<float> vec, ReadOnlySpan<float> cos,
                                  ReadOnlySpan<float> sin, int headDim)
```

This is the hottest method in the kernel — called once per head per token. For Llama 3 8B (32 Q heads + 8 K heads = 40 calls per token, headDim=128), this processes 5120 floats per token step.

### SIMD path (AVX2 + optional FMA)

The SIMD path handles 4 dimension pairs (8 floats) per iteration. The challenge is that the input data is **interleaved**: `[e0, o0, e1, o1, e2, o2, e3, o3]` (even = real parts, odd = imaginary parts), but the rotation formula needs them **separated** (all evens together, all odds together) to apply the cos/sin as a vector multiply.

The three-stage pipeline per iteration:

**Stage 1 — Deinterleave (1 instruction)**

```
Input: [e0, o0, e1, o1, e2, o2, e3, o3]   (256-bit, 8 floats)
                      ↓ PermuteVar8x32 with indices [0,2,4,6,1,3,5,7]
Permuted: [e0, e1, e2, e3, o0, o1, o2, o3]  (lower lane = evens, upper = odds)
```

`Avx2.PermuteVar8x32` reorders 32-bit lanes within a 256-bit register using a runtime index vector. Indices `[0,2,4,6]` gather the even-position floats into the lower 128-bit lane; `[1,3,5,7]` gather the odd-position floats into the upper 128-bit lane. A single instruction replaces what would otherwise require two shuffles and two blends.

After the permute, `.GetLower()` and `.GetUpper()` extract `even` and `odd` as 128-bit `Vector128<float>` values — two zero-cost register aliases of the same 256-bit result, with no additional instructions.

**Stage 2 — Apply 2×2 rotation matrix (2 multiply + 2 FMA or 4 multiply + 2 add/sub)**

```
even' = even * cos − odd * sin
odd'  = even * sin + odd * cos
```

With FMA (`Fma.IsSupported`):

```csharp
// MultiplyAddNegated(a, b, c) = -(a*b) + c = c - a*b
newEven = Fma.MultiplyAddNegated(odd, sinVec, even * cosVec);
//        -(odd * sin) + (even * cos) = even*cos - odd*sin  ✓
newOdd  = Fma.MultiplyAdd(even, sinVec, odd * cosVec);
//        (even * sin) + (odd * cos)                        ✓
```

Each line compiles to one `vmulps` (prereq multiply for the `c` argument) + one `vfnmadd132ps`/`vfmadd132ps`. The fused part absorbs one rounding step, giving slightly higher numerical accuracy than the non-FMA path.

Without FMA: four `vmulps` + one `vsubps` + one `vaddps` — six scalar-vector instructions.

**Stage 3 — Re-interleave (2 instructions)**

```
lo = Sse.UnpackLow(newEven, newOdd)   → [ne0, no0, ne1, no1]
hi = Sse.UnpackHigh(newEven, newOdd)  → [ne2, no2, ne3, no3]
result = Vector256.Create(lo, hi)     → [ne0, no0, ne1, no1, ne2, no2, ne3, no3]
```

`Sse.UnpackLow` interleaves the low halves of two 128-bit registers; `Sse.UnpackHigh` does the same for the high halves. The result is the rotated vector in the original interleaved layout, ready to store back.

**Why not a second `PermuteVar8x32` to re-interleave?** A permute can re-interleave `[ne0,ne1,ne2,ne3,no0,no1,no2,no3]` into `[ne0,no0,ne1,no1,ne2,no2,ne3,no3]`, but it requires knowing the inverse permutation index vector. The `UnpackLow`/`UnpackHigh` pair naturally interleaves two 128-bit registers without an index operand — fewer instructions and no live constant register.

**The deinterleave index is hoisted before the loop.** If declared inside the loop, `Vector256.Create(0, 2, 4, 6, 1, 3, 5, 7)` would be re-materialized every iteration (even if the JIT lifts it, the intent is unclear). Hoisting it outside makes the invariance explicit and prevents any JIT heuristic from missing the optimization.

### Scalar fallback

```csharp
for (; i < halfDim; i++)
{
    float e = vec[2 * i];
    float o = vec[2 * i + 1];
    vec[2 * i]     = e * cos[i] - o * sin[i];
    vec[2 * i + 1] = e * sin[i] + o * cos[i];
}
```

The scalar loop handles:
1. Platforms without AVX2 (all `halfDim` pairs)
2. Remainder pairs when `halfDim % 4 != 0` (e.g., headDim=24 → halfDim=12 → SIMD handles 8 pairs, scalar handles 4)

For typical models (headDim=64, 128, 256 — all multiples of 8), the scalar loop runs zero iterations.

---

## Implementation: `Execute`

```csharp
public static void Execute(Span<float> q, Span<float> k, ReadOnlySpan<int> positions,
                            int numHeads, int numKvHeads, int headDim,
                            ReadOnlySpan<float> cosTable, ReadOnlySpan<float> sinTable)
```

`Execute` is the public entry point for rotating all heads of a token batch. It is responsible for:

1. **Slicing the cos/sin table** to the current position: `cosTable[pos * halfDim .. pos * halfDim + halfDim]`. A table lookup replaces trig function calls on the hot path.

2. **Iterating heads**: Q and K may have different head counts (GQA). All Q heads and all K heads share the same cos/sin slice for their token — RoPE is not head-specific, only position-specific.

3. **Slicing head vectors**: `q[t * qStride + h * headDim .. + headDim]` extracts the `h`-th head of token `t` as a contiguous span. `qStride = numHeads * headDim` is the row stride; `kStride = numKvHeads * headDim` is the KV row stride.

The `positions` parameter is an explicit index array (rather than assuming `positions[t] == t`) to support non-contiguous position assignment — required for chunked prefill and decode steps where new tokens have positions into an existing KV cache.

**Data layout**:
```
Q: [token0_head0 | token0_head1 | ... | token0_headH | token1_head0 | ...]
    ←───── qStride (numHeads × headDim) ──────────────────────────────────→
```

---

## Memory Allocation

| Site | Allocation | Notes |
|------|-----------|-------|
| Frequency scratch | `stackalloc float[halfDim]` | ≤ 256 bytes for any real model; falls back to ArrayPool above 8 KB |
| ArrayPool<float> (rare) | `ArrayPool<float>.Shared.Rent(halfDim)` | Only for exotic headDim > 16384; returned immediately |
| Cos/sin tables | Caller-owned `float[]` or span | `RoPEPositionEncoding` owns these as `float[]` fields |
| All inference work | Zero allocations | Table lookup + in-place rotation |

The cos/sin tables themselves (`maxSeqLen * halfDim * 4 * 2` bytes) are allocated once by `RoPEPositionEncoding` at model load time:
- Llama 2 7B (headDim=128, maxSeqLen=4096): **2 MB** total (2 × 4096 × 64 × 4 bytes)
- Llama 3 8B (headDim=128, maxSeqLen=8192): **4 MB** total

---

## Data Structures

| Name | Type | Purpose |
|------|------|---------|
| `cosTable` | `float[]` flat, `[maxSeqLen × halfDim]` | `cosTable[pos * halfDim + i]` = `cos(pos × θᵢ)` |
| `sinTable` | `float[]` flat, `[maxSeqLen × halfDim]` | `sinTable[pos * halfDim + i]` = `sin(pos × θᵢ)` |
| `freqs` | `Span<float>`, length `halfDim` | Scratch in `FillTables`; `freqs[i] = 1 / theta^(2i/d)` |
| `deinterleaveIdx` | `Vector256<int>`, hoisted | Permute control for AVX2 deinterleave: `[0,2,4,6,1,3,5,7]` |

---

## Scalar Reference Implementations

Both `PrecomputeFrequencyTableScalar` and `ApplyRotationScalar` (`internal`) are exact algorithmic equivalents of their public counterparts, without any SIMD or stackalloc optimizations. They exist purely for test verification: the test suite runs both paths on the same inputs and asserts element-wise equality to `1e-5f` tolerance.

`ExecuteScalar` similarly wraps the scalar variants for full end-to-end comparison.

---

## Tests (`RoPETests.cs`)

The test suite has 11 tests grouped by method under test:

### `PrecomputeFrequencyTable` (3 tests)

**`PrecomputeFrequencyTable_KnownTheta_MatchesHandCalculated`** — Pin exact numerical values for `headDim=4`, `theta=10000`, `pos=1`. The hand-calculated values are:
- `i=0`: `freq = 1/10000^0 = 1.0`, `angle = 1.0`
- `i=1`: `freq = 1/10000^0.5 = 0.01`, `angle = 0.01`

This catches any formula error in the exponent (e.g., off-by-one in `2i/headDim`, wrong base) before running SIMD code.

**`PrecomputeFrequencyTable_PositionZero_CosOnesSinZeros`** — At position 0, every angle is `0 × θᵢ = 0`, so `cos = 1` and `sin = 0` regardless of `theta` or `headDim`. This verifies the table isn't garbage-initialized and that multiplying by position 0 works.

**`PrecomputeFrequencyTable_CustomTheta_DifferentFrequencies`** — Confirms that `theta=10000` and `theta=500000` produce different tables at positions > 0. Guards against accidentally hardcoding `theta`.

**`PrecomputeFrequencyTable_MatchesReference`** — Pins three specific table entries for a Llama-style config (`headDim=128`, `theta=10000`, `pos=5`): the first frequency pair, the middle (i=32), and the last (i=63). This is the integration test that would catch a regression in the frequency precomputation optimization.

### `ApplyRotation` (4 tests)

**`ApplyRotation_PositionZero_NoChange`** — With `cos=1, sin=0` (position 0), the rotation matrix is the identity. Output must equal input exactly. Catches any off-by-one in the pair indexing.

**`ApplyRotation_KnownRotation_HandCalculated`** — For `headDim=4` with manually constructed `cos` and `sin` arrays, each of the 4 output floats is verified against its hand-computed formula. This is the most direct test of the rotation arithmetic — it catches swapped signs or transposed cos/sin arguments.

**`ApplyRotation_ScalarMatchesSIMD`** — Runs the same random `headDim=128` vector through both the SIMD and scalar paths, asserting element-wise equality to `1e-5f`. This is the primary guard against bugs in the AVX2 deinterleave-rotate-reinterleave pipeline. Skipped when AVX2 is not available.

**`ApplyRotation_HeadDim128_PairRotation`** — Sets only one pair (pair #17) to `(1, 0)` with all others at zero, and applies a known rotation only to pair #17 (others get identity). Verifies that the SIMD path does not contaminate neighbouring pairs — critical for catching off-by-one errors in the permute indices or store offsets.

### `Execute` (4 tests)

**`Execute_MultipleHeads_EachRotatedIndependently`** — Runs `Execute` with 2 Q heads and 2 K heads, then verifies each head by replaying the scalar reference rotation individually. Confirms that the stride arithmetic `t * qStride + h * headDim` is correct and that heads don't overwrite each other.

**`Execute_DifferentPositions_DifferentRotations`** — Sends 3 identical tokens at positions `[0, 1, 2]`. Position 0 must produce no change (identity); positions 1 and 2 must produce distinct outputs. Verifies that the `positions[t]` lookup is used correctly and that different absolute positions produce different rotations.

**`Execute_ScalarMatchesSIMD_LargeInput`** — Full end-to-end SIMD vs. scalar comparison: 32 Q heads, 8 K heads, `headDim=128`, 4 tokens. Covers the GQA (different Q/K head counts) code path in `Execute`. Skipped on non-AVX2 platforms.

**`PrecomputeFrequencyTable_MatchesReference`** — (in the table group but also serves as an `Execute` smoke test via the pinned values).

---

## Known Limitations

- **RoPE scaling variants not implemented.** `RoPEScalingType.None` only. Linear, NTK, YaRN, and DynamicNTK scaling (for extending context beyond the trained maximum) will be implemented in a later step as needed for long-context models. The `RoPEConfig.ScalingType` field is parsed from GGUF but currently ignored.
- **`headDim` must be even.** The kernel enforces this with an `ArgumentException`. All real models satisfy this; the restriction could in principle be lifted for the scalar path at the cost of complexity.
- **Single-precision only.** The kernel operates on `float` throughout. Key and query projections in Llama-class models are always FP32 during CPU inference, so no higher-precision path is needed.
