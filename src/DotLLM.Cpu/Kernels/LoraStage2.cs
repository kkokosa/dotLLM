using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// LoRA stage 2 — the up-projection accumulate. Computes
/// <c>y[t, o] += scale × sum_r A[o, r] × tmp[t, r]</c> where
/// <c>tmp</c> is the stage-1 output (<c>[seqLen, rank]</c>) and <c>A</c>
/// is the up-projection (<c>[outputDim, rank]</c>) of the LoRA factor pair.
/// </summary>
/// <remarks>
/// <para>
/// The production (pre-Phase 4d.6) stage-2 path iterates tokens, then calls
/// <see cref="MatMul.GemvF32(float*, float*, float*, int, int)"/> with
/// <c>M = outputDim, K = rank</c> per token. For typical PEFT
/// <c>rank ∈ {8, 16, 32, 64}</c> and <c>outputDim ∈ {512..5632}</c> this
/// produces <c>seqLen × outputDim</c> short (length-rank) dot-product calls
/// — function-entry dominated, even though each individual Dot is only
/// 16 elements at rank=16. Direct kernel profiling on Strix Halo
/// (<c>benchmarks/LoraQ8Stage1Probe</c>) showed stage 2 alone accounts for
/// ~85% of total LoRA-Apply wall time at outputDim=2048 / N=512 / rank=16.
/// </para>
/// <para>
/// <b>Outer-product fusion (Phase 4d.6).</b> When <c>rank = 16</c> the
/// per-token <c>tmp[t, :]</c> is exactly one <see cref="Vector512{T}"/>.
/// We pre-broadcast the 16 scalar lanes once per token, then sweep
/// <c>outputDim</c> in tiles of 16, FMA-accumulating into one
/// <see cref="Vector512{T}"/> tile-acc per 16-output-tile. Each tile
/// requires 16 contiguous 16-float reads from <c>A_transposed</c> —
/// hardware-prefetch-friendly. The <c>delta</c> scratch is gone (no
/// per-token <see cref="System.Buffers.ArrayPool{T}"/> rent) and the
/// per-token <c>2048 × 512 = 1M</c> short-Dot calls collapse into
/// <c>seqLen × outputDim/16 ≈ 65K</c> tile FMAs.
/// </para>
/// <para>
/// <b>Why rank-specialised.</b> RyuJIT (.NET 10) keeps 16 explicit
/// <c>Vector512&lt;float&gt;</c> broadcast locals in ZMM registers across
/// the inner <c>outputDim</c>-tile loop, but
/// <c>stackalloc Vector512&lt;float&gt;[rank]</c> spills to L1 because the
/// register allocator cannot prove array element addresses are loop
/// invariants. We therefore unroll the rank dimension by hand — 16 named
/// locals, one per LoRA-A column. Other ranks (4, 8, 32) get scalar
/// fallbacks rather than copy-paste kernels until the bench shows they
/// matter.
/// </para>
/// <para>
/// <b>A layout.</b> The kernel consumes <c>A_transposed</c> in
/// <c>[rank, outputDim]</c> row-major form — built once at adapter-load
/// from the natural <c>[outputDim, rank]</c> A buffer via
/// <see cref="BuildATransposedF32"/>. The original A buffer is left
/// untouched so existing callers (Vulkan, other dtypes) continue to see
/// the canonical layout. Storage overhead is one extra <c>O · r × 4</c>
/// bytes per <c>(layer, proj)</c> — for Llama-3.2-1B / rank=16 that is
/// roughly 4.8 MB total adapter overhead (cf. ~120 MB base weights), an
/// acceptable trade for the ~3-4× LoRA-Apply speedup it unlocks.
/// </para>
/// </remarks>
public static unsafe class LoraStage2
{
    /// <summary>
    /// Returns <c>true</c> when the current CPU supports the AVX-512 fast
    /// path used by <see cref="ApplyF32_R16"/>. Callers must fall back to
    /// the generic per-token GEMV path otherwise.
    /// </summary>
    public static bool IsAvx512FastPathSupported => Avx512F.IsSupported;


    /// <summary>
    /// Builds the transposed-A buffer used by the rank-specialised fast
    /// paths. Source <paramref name="aRowMajor"/> is <c>[outputDim, rank]</c>
    /// row-major F32; destination is <c>[rank, outputDim]</c> row-major F32,
    /// 64-byte aligned. Caller owns the returned handle and must free with
    /// <see cref="NativeMemory.AlignedFree"/>.
    /// </summary>
    public static nint BuildATransposedF32(float* aRowMajor, int outputDim, int rank)
    {
        if (outputDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputDim), outputDim, "outputDim must be positive.");
        if (rank <= 0)
            throw new ArgumentOutOfRangeException(nameof(rank), rank, "rank must be positive.");

        long elems = (long)rank * outputDim;
        nint dst = (nint)NativeMemory.AlignedAlloc((nuint)(elems * sizeof(float)), 64);
        float* d = (float*)dst;

        // Naive transpose — adapter load is one-shot, no SIMD needed.
        for (int r = 0; r < rank; r++)
        {
            float* dstRow = d + (long)r * outputDim;
            float* srcCol = aRowMajor + r; // A[o, r] = aRowMajor[o*rank + r]
            for (int o = 0; o < outputDim; o++)
                dstRow[o] = srcCol[(long)o * rank];
        }
        return dst;
    }

    /// <summary>
    /// Dtype-aware transposed-A builder. Dequantises <paramref name="aHandle"/>
    /// from <paramref name="aDType"/> into an F32 staging buffer, then builds
    /// the <c>[rank, outputDim]</c> transposed F32 layout. Used at the LoRA
    /// dispatch site for lazy first-use materialisation.
    /// </summary>
    /// <remarks>
    /// Storage overhead: one extra <c>outputDim × rank × 4</c> bytes per
    /// <c>(layer, proj)</c>. For Llama-3.2-1B / rank=16 / 7 projections / 16
    /// layers that totals ~4.8 MB — acceptable vs the 3-4× per-call speedup.
    /// One-time build cost is also ~5 ms per adapter at this scale, amortised
    /// from the first inference call onward.
    /// </remarks>
    public static nint BuildATransposedF32FromDType(
        nint aHandle, DotLLM.Core.Lora.LoraWeightDType aDType, int outputDim, int rank)
    {
        if (aHandle == 0)
            throw new ArgumentException("aHandle must be non-zero.", nameof(aHandle));

        long elems = (long)outputDim * rank;
        if (aDType == DotLLM.Core.Lora.LoraWeightDType.F32)
            return BuildATransposedF32((float*)aHandle, outputDim, rank);

        // Dequant A into a staging F32 buffer first, then transpose.
        float* staging = (float*)NativeMemory.AlignedAlloc((nuint)(elems * sizeof(float)), 64);
        try
        {
            switch (aDType)
            {
                case DotLLM.Core.Lora.LoraWeightDType.F16:
                    {
                        Half* src = (Half*)aHandle;
                        var srcSpan = new ReadOnlySpan<Half>(src, (int)elems);
                        var dstSpan = new Span<float>(staging, (int)elems);
                        System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(srcSpan, dstSpan);
                        break;
                    }
                case DotLLM.Core.Lora.LoraWeightDType.BF16:
                    {
                        byte* src = (byte*)aHandle;
                        for (long i = 0; i < elems; i++)
                        {
                            ushort raw = (ushort)(src[i * 2] | (src[i * 2 + 1] << 8));
                            uint asF32 = (uint)raw << 16;
                            staging[i] = BitConverter.UInt32BitsToSingle(asF32);
                        }
                        break;
                    }
                default:
                    throw new NotSupportedException(
                        $"A dtype {aDType} is not supported by BuildATransposedF32FromDType.");
            }
            return BuildATransposedF32(staging, outputDim, rank);
        }
        finally
        {
            NativeMemory.AlignedFree(staging);
        }
    }

    /// <summary>
    /// Helper for the LoRA dispatch sites — when conditions allow
    /// (<c>rank = 16</c>, AVX-512 available, <c>aHandle</c> populated) and
    /// the adapter hasn't yet cached a transposed-A for
    /// <c>(layer, projection)</c>, builds + installs it. Returns the
    /// current transposed-A handle (which may be <c>0</c> when conditions
    /// don't apply — callers pass it to <see cref="LoraDelta"/> as the
    /// optional fast-path opt-in).
    /// </summary>
    public static nint EnsureATransposedF32(
        DotLLM.Core.Lora.LoraAdapter? cpuAdapter,
        int layerIndex,
        string projection,
        in DotLLM.Core.Lora.LoraLayerWeights w,
        int rank)
    {
        if (cpuAdapter is null) return 0;
        if (rank != 16) return 0;
        if (!IsAvx512FastPathSupported) return 0;
        if (w.AHandle == 0) return 0;
        if (w.ATransposedHandle != 0) return w.ATransposedHandle;

        nint freshlyBuilt = BuildATransposedF32FromDType(
            w.AHandle, w.ResolvedAWeightDType, w.OutputDim, rank);
        return cpuAdapter.InstallATransposedHandle(layerIndex, projection, freshlyBuilt);
    }

    /// <summary>
    /// Eagerly materialises the transposed-A cache for every
    /// <c>(layer, projection)</c> pair declared by <paramref name="cpuAdapter"/>.
    /// Idempotent — safe to call multiple times. Used by the runtime to
    /// move the lazy-build cost out of the first inference call (matters
    /// for low-iteration benchmarks and for first-token latency).
    /// </summary>
    /// <returns>
    /// Number of transposed-A buffers freshly built (vs already cached).
    /// </returns>
    public static int PrewarmAdapter(DotLLM.Core.Lora.LoraAdapter? cpuAdapter)
    {
        if (cpuAdapter is null) return 0;
        if (!IsAvx512FastPathSupported) return 0;
        if (cpuAdapter.Rank != 16) return 0;

        // O(1) early-exit when this adapter has already been prewarmed —
        // critical because Forward(adapter) is called per decode token, so
        // cheap wide-key enumeration here would burn on the hot path.
        if (cpuAdapter.IsStage2FastPathPrewarmed) return 0;

        int built = 0;
        // Snapshot the keys so we don't enumerate while installing (which
        // mutates the underlying dictionary under the adapter's lock).
        var keys = new List<(int Layer, string Proj)>(cpuAdapter.LayerWeights.Keys);
        foreach (var (layer, proj) in keys)
        {
            if (cpuAdapter.GetLayerWeights(layer, proj) is not { } w) continue;
            if (w.ATransposedHandle != 0) continue;
            if (w.AHandle == 0) continue;

            nint freshlyBuilt = BuildATransposedF32FromDType(
                w.AHandle, w.ResolvedAWeightDType, w.OutputDim, cpuAdapter.Rank);
            cpuAdapter.InstallATransposedHandle(layer, proj, freshlyBuilt);
            built++;
        }
        cpuAdapter.MarkStage2FastPathPrewarmed();
        return built;
    }

    /// <summary>
    /// Stage-2 fast path for <c>rank = 16</c>: outer-product accumulate.
    /// Writes <c>y[t, :] += scale × (Aᵀ · tmp[t, :])</c> for all
    /// <c>t ∈ [0, seqLen)</c>, where <paramref name="aTransposed"/> is the
    /// <c>[rank=16, outputDim]</c> row-major view of A.
    /// </summary>
    /// <param name="aTransposed">
    /// <c>[16, outputDim]</c> row-major F32 — built once at adapter load.
    /// </param>
    /// <param name="tmp">Stage-1 output, <c>[seqLen, 16]</c> row-major F32.</param>
    /// <param name="y">Destination, <c>[seqLen, outputDim]</c> row-major F32 (read-modify-write).</param>
    /// <param name="seqLen">Token count.</param>
    /// <param name="outputDim">Up-projection output dimension.</param>
    /// <param name="scale">LoRA scaling factor (<c>alpha / rank</c>).</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ApplyF32_R16(
        float* aTransposed, float* tmp, float* y,
        int seqLen, int outputDim, float scale)
    {
        const int rank = 16;
        if (!Avx512F.IsSupported)
            throw new PlatformNotSupportedException(
                "LoraStage2.ApplyF32_R16 requires AVX-512F. Callers must check IsAvx512FastPathSupported first.");

        Vector512<float> scaleVec = Vector512.Create(scale);

        for (int t = 0; t < seqLen; t++)
        {
            float* tmpRow = tmp + (long)t * rank;

            // Pre-broadcast the 16 stage-1 scalars once per token. RyuJIT
            // keeps these 16 named locals in ZMM registers across the inner
            // o-tile loop. Tested empirically vs a `stackalloc Vector512[16]`
            // alternative which spilled — see Phase 4d.6 closure notes.
            Vector512<float> b0 = Vector512.Create(tmpRow[0]);
            Vector512<float> b1 = Vector512.Create(tmpRow[1]);
            Vector512<float> b2 = Vector512.Create(tmpRow[2]);
            Vector512<float> b3 = Vector512.Create(tmpRow[3]);
            Vector512<float> b4 = Vector512.Create(tmpRow[4]);
            Vector512<float> b5 = Vector512.Create(tmpRow[5]);
            Vector512<float> b6 = Vector512.Create(tmpRow[6]);
            Vector512<float> b7 = Vector512.Create(tmpRow[7]);
            Vector512<float> b8 = Vector512.Create(tmpRow[8]);
            Vector512<float> b9 = Vector512.Create(tmpRow[9]);
            Vector512<float> b10 = Vector512.Create(tmpRow[10]);
            Vector512<float> b11 = Vector512.Create(tmpRow[11]);
            Vector512<float> b12 = Vector512.Create(tmpRow[12]);
            Vector512<float> b13 = Vector512.Create(tmpRow[13]);
            Vector512<float> b14 = Vector512.Create(tmpRow[14]);
            Vector512<float> b15 = Vector512.Create(tmpRow[15]);

            float* yRow = y + (long)t * outputDim;
            int o = 0;
            for (; o + 16 <= outputDim; o += 16)
            {
                // Chain of 16 FMAs into one accumulator. The natural FMA
                // dependency depth is 16 × FMA-latency, but Zen 5's OOO
                // scheduler hides most of it via the next iteration's
                // independent loads. Empirically a 4-way independent-
                // accumulator split (see the LoraQ8Stage1Probe `Path E2`
                // experiment) gave no measurable gain on this geometry —
                // the kernel is load- and decode-bound, not FMA-latency
                // bound. Keeping the simpler form for readability.
                Vector512<float> acc = Avx512F.LoadVector512(aTransposed + 0 * outputDim + o) * b0;
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 1 * outputDim + o), b1, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 2 * outputDim + o), b2, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 3 * outputDim + o), b3, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 4 * outputDim + o), b4, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 5 * outputDim + o), b5, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 6 * outputDim + o), b6, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 7 * outputDim + o), b7, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 8 * outputDim + o), b8, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 9 * outputDim + o), b9, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 10 * outputDim + o), b10, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 11 * outputDim + o), b11, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 12 * outputDim + o), b12, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 13 * outputDim + o), b13, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 14 * outputDim + o), b14, acc);
                acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(aTransposed + 15 * outputDim + o), b15, acc);

                Vector512<float> yVec = Avx512F.LoadVector512(yRow + o);
                yVec = Avx512F.FusedMultiplyAdd(scaleVec, acc, yVec);
                Avx512F.Store(yRow + o, yVec);
            }
            // Tail (outputDim not multiple of 16). Rare on transformer shapes
            // (k/v/o/q/gate/up/down outputDims are all multiples of 16).
            for (; o < outputDim; o++)
            {
                float s = tmpRow[0] * aTransposed[0L * outputDim + o]
                        + tmpRow[1] * aTransposed[1L * outputDim + o]
                        + tmpRow[2] * aTransposed[2L * outputDim + o]
                        + tmpRow[3] * aTransposed[3L * outputDim + o]
                        + tmpRow[4] * aTransposed[4L * outputDim + o]
                        + tmpRow[5] * aTransposed[5L * outputDim + o]
                        + tmpRow[6] * aTransposed[6L * outputDim + o]
                        + tmpRow[7] * aTransposed[7L * outputDim + o]
                        + tmpRow[8] * aTransposed[8L * outputDim + o]
                        + tmpRow[9] * aTransposed[9L * outputDim + o]
                        + tmpRow[10] * aTransposed[10L * outputDim + o]
                        + tmpRow[11] * aTransposed[11L * outputDim + o]
                        + tmpRow[12] * aTransposed[12L * outputDim + o]
                        + tmpRow[13] * aTransposed[13L * outputDim + o]
                        + tmpRow[14] * aTransposed[14L * outputDim + o]
                        + tmpRow[15] * aTransposed[15L * outputDim + o];
                yRow[o] += scale * s;
            }
        }
    }
}
