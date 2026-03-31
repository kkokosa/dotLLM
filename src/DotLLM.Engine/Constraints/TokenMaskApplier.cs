using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Constraints;

namespace DotLLM.Engine.Constraints;

/// <summary>
/// Applies a <see cref="TokenMask"/> to logits, setting disallowed tokens to <c>-∞</c>.
/// Uses AVX2-vectorized masking when available, with scalar fallback.
/// </summary>
internal static class TokenMaskApplier
{
    /// <summary>
    /// Applies the constraint mask to logits in-place.
    /// Disallowed tokens (bit = 0 in mask) are set to <see cref="float.NegativeInfinity"/>.
    /// </summary>
    public static void Apply(Span<float> logits, TokenMask mask)
    {
        if (Avx2.IsSupported)
            ApplyAvx2(logits, mask);
        else
            ApplyScalar(logits, mask);
    }

    private static void ApplyAvx2(Span<float> logits, TokenMask mask)
    {
        var negInf = Vector256.Create(float.NegativeInfinity);
        var bits = mask.AsSpan();
        int floatIdx = 0;

        for (int block = 0; block < bits.Length && floatIdx < logits.Length; block++)
        {
            long allowed = bits[block];

            // Fast path: all 64 tokens allowed — skip entire block
            if (allowed == ~0L)
            {
                floatIdx += 64;
                continue;
            }

            // Fast path: all 64 tokens disallowed — blast to -inf
            if (allowed == 0L)
            {
                int count = Math.Min(64, logits.Length - floatIdx);
                logits.Slice(floatIdx, count).Fill(float.NegativeInfinity);
                floatIdx += 64;
                continue;
            }

            // Process 8 floats per byte (8 bytes per long)
            for (int byteIdx = 0; byteIdx < 8 && floatIdx < logits.Length; byteIdx++)
            {
                int b = (int)((allowed >> (byteIdx * 8)) & 0xFF);

                // Fast path: all 8 bits set — skip
                if (b == 0xFF)
                {
                    floatIdx += 8;
                    continue;
                }

                int remaining = logits.Length - floatIdx;
                if (remaining >= 8)
                {
                    // Expand byte to 8x int32 mask, blend with -inf
                    var expandedMask = ExpandByteMask(b);
                    var src = Vector256.LoadUnsafe(ref logits[floatIdx]);
                    // BlendVariable: for each element, if mask sign bit is 1 → pick src (allowed),
                    // else pick negInf (disallowed)
                    var result = Avx.BlendVariable(negInf, src, expandedMask.AsSingle());
                    result.StoreUnsafe(ref logits[floatIdx]);
                }
                else
                {
                    // Tail: scalar for remaining < 8 elements
                    for (int bit = 0; bit < remaining; bit++)
                    {
                        if ((b & (1 << bit)) == 0)
                            logits[floatIdx + bit] = float.NegativeInfinity;
                    }
                }

                floatIdx += 8;
            }
        }
    }

    /// <summary>
    /// Expands 8 bits of a byte mask into 8 × int32 values.
    /// Bit = 1 → all-ones (0xFFFFFFFF), bit = 0 → all-zeros (0x00000000).
    /// The sign bit of each int32 is used by VBLENDVPS for selection.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> ExpandByteMask(int byteMask)
    {
        var bcast = Vector256.Create(byteMask);
        var selectors = Vector256.Create(1, 2, 4, 8, 16, 32, 64, 128);
        var masked = Avx2.And(bcast, selectors);
        return Avx2.CompareEqual(masked, selectors);
    }

    private static void ApplyScalar(Span<float> logits, TokenMask mask)
    {
        var bits = mask.AsSpan();

        for (int block = 0; block < bits.Length; block++)
        {
            long allowed = bits[block];
            if (allowed == ~0L)
                continue; // all allowed in this block

            int baseIdx = block * 64;
            int count = Math.Min(64, logits.Length - baseIdx);

            if (allowed == 0L)
            {
                // All disallowed — fill with -inf
                logits.Slice(baseIdx, count).Fill(float.NegativeInfinity);
                continue;
            }

            for (int bit = 0; bit < count; bit++)
            {
                if ((allowed & (1L << bit)) == 0)
                    logits[baseIdx + bit] = float.NegativeInfinity;
            }
        }
    }
}
