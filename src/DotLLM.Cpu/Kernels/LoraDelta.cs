using System.Buffers;
using System.Buffers.Binary;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Lora;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// LoRA delta accumulation: <c>y += alpha × (x · B) · A</c>.
/// </summary>
/// <remarks>
/// <para>
/// Tensor layouts (row-major F32, matching <see cref="MatMul.GemmF32(float*, float*, float*, int, int, int)"/>):
/// </para>
/// <list type="bullet">
///   <item><c>x</c>: <c>[seqLen, inputDim]</c> — the same input that fed the base projection.</item>
///   <item><c>B</c>: <c>[rank, inputDim]</c> — LoRA down-projection weight, stored row-major
///     so <c>tmp[t, r] = sum_i x[t, i] · B[r, i]</c> matches the standard
///     "weight matrix as <c>[outputDim, inputDim]</c>" convention dotLLM uses everywhere.</item>
///   <item><c>A</c>: <c>[outputDim, rank]</c> — LoRA up-projection weight (same convention).</item>
///   <item><c>y</c>: <c>[seqLen, outputDim]</c> — base-projection output; LoRA delta added in-place.</item>
/// </list>
/// <para>
/// Implementation: two stacked GEMMs through a small <c>[seqLen, rank]</c>
/// scratch — for typical r∈[8, 64] each is a thin matmul that <see cref="MatMul.GemmF32(float*, float*, float*, int, int, int)"/>
/// already handles efficiently. We rent the scratch from <see cref="ArrayPool{T}"/>
/// so there is no per-call native allocation.
/// </para>
/// </remarks>
public static unsafe class LoraDelta
{
    /// <summary>
    /// Accumulates <c>y += scale × (x · B) · A</c> in-place. Mathematically:
    /// <c>tmp[t, r] = sum_i x[t, i] · B[r, i]</c>; then
    /// <c>y[t, o] += scale × sum_r A[o, r] · tmp[t, r]</c>.
    /// </summary>
    /// <param name="x">Input pointer, row-major <c>[seqLen, inputDim]</c>.</param>
    /// <param name="bWeight">B (down-proj) pointer, row-major <c>[rank, inputDim]</c>.</param>
    /// <param name="aWeight">A (up-proj) pointer, row-major <c>[outputDim, rank]</c>.</param>
    /// <param name="y">Output pointer (read-modify-write), row-major <c>[seqLen, outputDim]</c>.</param>
    /// <param name="seqLen">Number of input tokens in this call.</param>
    /// <param name="inputDim">Projection input dimension.</param>
    /// <param name="outputDim">Projection output dimension.</param>
    /// <param name="rank">LoRA rank (typical 8..64).</param>
    /// <param name="scale">Scaling factor — typically <c>alpha / rank</c>.</param>
    /// <param name="aTransposedHandle">
    /// Phase 4d.6 — optional <c>[rank, outputDim]</c> row-major F32 view of
    /// A. When non-zero AND <paramref name="rank"/> = 16 AND AVX-512 is
    /// available, stage 2 dispatches through the outer-product fast path
    /// (<see cref="LoraStage2.ApplyF32_R16"/>). Pass <c>0</c> for the legacy
    /// per-token GEMV stage-2 path.
    /// </param>
    [SkipLocalsInit]
    public static void Apply(float* x, float* bWeight, float* aWeight, float* y,
                             int seqLen, int inputDim, int outputDim, int rank, float scale,
                             nint aTransposedHandle = 0)
    {
        if (seqLen <= 0 || rank <= 0) return;

        // Stage 1: tmp[t, r] = sum_i x[t, i] · B[r, i].
        // GemmF32 contracts as C[N, M] = B[N, K] × A[M, K]^T, so we pass
        //   a = bWeight   (M=rank, K=inputDim)
        //   b = x         (N=seqLen, K=inputDim)
        //   c = tmp       (N=seqLen, M=rank)
        int tmpElems = seqLen * rank;
        float[] tmpBuf = ArrayPool<float>.Shared.Rent(tmpElems);
        try
        {
            fixed (float* tmp = tmpBuf)
            {
                MatMul.GemmF32(bWeight, x, tmp, rank, inputDim, seqLen);

                // Stage 2: y[t, o] += scale × sum_r A[o, r] · tmp[t, r].
                // Phase 4d.6 — when a pre-built [rank, outputDim] transposed-A
                // is available AND rank=16 AND AVX-512, the outer-product fast
                // path collapses the ~seqLen × outputDim short Dot calls into
                // ~seqLen × outputDim/16 tile FMAs (3-4× faster on Strix
                // Halo). Otherwise we fall back to the legacy per-token GEMV +
                // scaled MultiplyAdd path.
                Stage2(tmp, aWeight, y, seqLen, outputDim, rank, scale, aTransposedHandle);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(tmpBuf);
        }
    }

    /// <summary>
    /// Quantised-weight overload (Phase 4d.1). Dequantises B and A from
    /// <paramref name="bDType"/> / <paramref name="aDType"/> into a small F32
    /// scratch and reuses the standard F32 path. Both buffers are typically
    /// the same dtype (PEFT writes lora_A and lora_B together) but we permit
    /// independent dtypes for completeness.
    /// </summary>
    /// <remarks>
    /// <para>
    /// We dequantise the entire (small) B and A factors once per call rather
    /// than inlining dequant into the inner GEMM loop. For r=16 typical
    /// shapes the factors are kilobytes — staying in L1 — and this avoids
    /// duplicating the GEMM kernel for each dtype combination.
    /// </para>
    /// <para>
    /// Phase 4d.6: when <paramref name="aTransposedHandle"/> is non-zero and
    /// <paramref name="rank"/> is 16, stage 2 dispatches to
    /// <see cref="LoraStage2.ApplyF32_R16"/> — the outer-product fast path
    /// that collapses ~1M short Dot calls (per-token GEMV) into
    /// <c>seqLen × outputDim/16</c> tile FMAs. Empirically 3-4× faster on
    /// Strix Halo at typical Llama outputDims. When the handle is <c>0</c>
    /// stage 2 falls back to the legacy per-token GEMV path. Callers (the
    /// LoRA dispatch site in <c>TransformerModel.ApplyLoraDelta</c>) own the
    /// transposed-A buffer lifetime — typically lazily built and cached in
    /// the <see cref="LoraAdapter"/>.
    /// </para>
    /// </remarks>
    [SkipLocalsInit]
    public static void Apply(float* x, void* bWeight, void* aWeight, float* y,
                             int seqLen, int inputDim, int outputDim, int rank, float scale,
                             LoraWeightDType bDType, LoraWeightDType aDType,
                             nint aTransposedHandle = 0)
    {
        if (seqLen <= 0 || rank <= 0) return;

        // Fast path: both F32 — go straight to the existing kernel without
        // copies. Common when callers haven't migrated to the dtype-aware path.
        if (bDType == LoraWeightDType.F32 && aDType == LoraWeightDType.F32)
        {
            Apply(x, (float*)bWeight, (float*)aWeight, y, seqLen, inputDim, outputDim, rank, scale,
                  aTransposedHandle);
            return;
        }

        // Q8_0 B fast-path (Phase 4d.4) — closes the prefill regression that
        // F32-on-Q8_0-base introduces. We use GemmQ8_0 for stage 1 (B is
        // [rank, inputDim] Q8_0; x is [seqLen, inputDim] F32 — the activation
        // is quantised once and reused across all `rank` weight rows). Stage 2
        // dequants A on read into the standard F32 path. A stays F16 / BF16 /
        // F32 because its contracted axis is `rank` (typical 8-16), too short
        // for a 32-element Q8_0 block.
        if (bDType == LoraWeightDType.Q8_0)
        {
            ApplyQ8_0B(x, (byte*)bWeight, aWeight, y,
                       seqLen, inputDim, outputDim, rank, scale, aDType, aTransposedHandle);
            return;
        }

        // Phase 4d.6 — skip A dequant when the outer-product fast path will
        // pick up the cached transposed-A in Stage2 (rank=16 + AVX-512). The
        // F16/BF16 dequant is otherwise dead work: Stage2 doesn't read aF32
        // on the fast path. B is still dequanted (stage 1 needs it).
        bool fastPathActive = rank == 16
            && aTransposedHandle != 0
            && LoraStage2.IsAvx512FastPathSupported;

        long bElems = (long)rank * inputDim;
        float[] bBuf = ArrayPool<float>.Shared.Rent((int)bElems);
        try
        {
            DequantToF32(bWeight, bDType, bBuf, (int)bElems);

            fixed (float* bF32 = bBuf)
            {
                if (fastPathActive)
                {
                    Apply(x, bF32, null, y, seqLen, inputDim, outputDim, rank, scale, aTransposedHandle);
                    return;
                }

                long aElems = (long)outputDim * rank;
                float[] aBuf = ArrayPool<float>.Shared.Rent((int)aElems);
                try
                {
                    DequantToF32(aWeight, aDType, aBuf, (int)aElems);
                    fixed (float* aF32 = aBuf)
                    {
                        Apply(x, bF32, aF32, y, seqLen, inputDim, outputDim, rank, scale, aTransposedHandle);
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(aBuf);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(bBuf);
        }
    }

    /// <summary>
    /// Phase 4d.4 — Q8_0 B-side LoRA delta path. Stores B as Q8_0 to halve
    /// the weight memory footprint of the adapter; on each call B is
    /// dequantised once into a small F32 scratch and the standard F32 GEMM
    /// stage 1 runs against it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Spike (4d.4) initially used <c>GemmQ8_0</c> for stage 1 to mirror the
    /// base-model Q8_0 GEMV. That path measured ~50% slower than the F32
    /// LoRA on Llama-3.2-1B-Q8_0 + Strix Halo (see
    /// <c>.continue-here-lora-quantised-delta.md</c> for the full data) —
    /// the geometry mismatch is the cause: <c>GemmQ8_0</c> with M=rank=16
    /// quantises the entire <c>(seqLen × inputDim)</c> activation tile per
    /// call, but the rank-tall stage-1 matmul does not amortise that
    /// quantisation cost across enough output rows. The activation-quant
    /// overhead alone exceeds the F32 stage-1 compute.
    /// </para>
    /// <para>
    /// The dequant-then-F32 fallback used here keeps the adapter memory
    /// halved (Q8_0 storage is ~1.06 B/elem vs F32's 4 B/elem) without
    /// paying the activation-quantise cost on every Apply call. Per-call
    /// B dequant is <c>rank × inputDim</c> floats written — typical 128 KiB
    /// for rank=16, inputDim=2048 — which fits in L2 and is dominated by
    /// the subsequent F32 GEMM.
    /// </para>
    /// </remarks>
    [SkipLocalsInit]
    private static void ApplyQ8_0B(float* x, byte* bQ8, void* aWeight, float* y,
                                   int seqLen, int inputDim, int outputDim, int rank, float scale,
                                   LoraWeightDType aDType, nint aTransposedHandle = 0)
    {
        if (inputDim % 32 != 0)
            throw new ArgumentException(
                $"Q8_0 LoRA B requires inputDim multiple of 32, got {inputDim}.",
                nameof(inputDim));

        // Dequant B (rank × inputDim Q8_0 → F32 scratch). Block size is
        // small (~128 KiB at typical shapes), fits in L2.
        long bElems = (long)rank * inputDim;
        float[] bF32Buf = ArrayPool<float>.Shared.Rent((int)bElems);
        try
        {
            DequantizeQ8_0RowsToF32(bQ8, bF32Buf.AsSpan(0, (int)bElems), rank, inputDim);

            fixed (float* bF32 = bF32Buf)
            {
                // Phase 4d.6 — when the outer-product fast path is engaged
                // (rank=16 + AVX-512 + cached transposed-A), the per-call
                // A dequant is dead work: stage 2 reads only from
                // aTransposedHandle. Skip the dequant entirely; pass null
                // for aF32 so the (unused) param has a clear sentinel.
                bool fastPathActive = rank == 16
                    && aTransposedHandle != 0
                    && LoraStage2.IsAvx512FastPathSupported;

                // A handling: F32 → use directly; F16/BF16 → dequant + F32 path.
                if (aDType == LoraWeightDType.F32)
                {
                    Apply(x, bF32, (float*)aWeight, y, seqLen, inputDim, outputDim, rank, scale,
                          aTransposedHandle);
                    return;
                }

                if (fastPathActive)
                {
                    // No need to dequant A — Stage2 uses transposed-A only.
                    // Pass null aWeight; Stage2's GemvF32 path is skipped.
                    Apply(x, bF32, null, y, seqLen, inputDim, outputDim, rank, scale,
                          aTransposedHandle);
                    return;
                }

                long aElems = (long)outputDim * rank;
                float[] aBuf = ArrayPool<float>.Shared.Rent((int)aElems);
                try
                {
                    DequantToF32(aWeight, aDType, aBuf, (int)aElems);
                    fixed (float* aF32 = aBuf)
                    {
                        Apply(x, bF32, aF32, y, seqLen, inputDim, outputDim, rank, scale,
                              aTransposedHandle);
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(aBuf);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(bF32Buf);
        }
    }

    /// <summary>
    /// Dequantises a contiguous Q8_0 block of <paramref name="rows"/> ×
    /// <paramref name="elementsPerRow"/> into <paramref name="dst"/>. Q8_0
    /// is a row-wise format so a contiguous range of rows is also a
    /// contiguous range of elements — we dispatch a single call into
    /// <see cref="Dequantize.ToFloat32(nint, long, DotLLM.Core.Configuration.QuantizationType, Span{float})"/>
    /// to use its AVX2 SIMD inner loop.
    /// </summary>
    private static void DequantizeQ8_0RowsToF32(byte* srcQ8, Span<float> dst, int rows, int elementsPerRow)
    {
        long totalElems = (long)rows * elementsPerRow;
        Dequantize.ToFloat32((nint)srcQ8, totalElems,
            DotLLM.Core.Configuration.QuantizationType.Q8_0, dst);
    }

    /// <summary>
    /// Phase 4d.5 / Gap 2 — Q8_0 LoRA-B with pre-quantised activation. When
    /// the dispatch site has already quantised <c>x</c> to Q8_0 for the base
    /// projection's GEMM (which is the case on a Q8_0 base via
    /// <c>TransformerModel.QuantizeInput</c>), we can re-use that buffer for
    /// LoRA stage 1 and skip both the F32 stage-1 multiply *and* the per-call
    /// B dequant. Stage 1 becomes a thin <c>GemmQ8_0</c> with <c>m=rank</c>,
    /// <c>n=seqLen</c>, <c>k=inputDim</c> and <c>preQuantizedInput=xQ8</c>;
    /// the activation-quantise cost that killed the spike (4d.4) is now
    /// fully amortised across base GEMM + LoRA stage 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Stage 2 is unchanged from the F32 / dequant-once path — A is dequanted
    /// once into F32 scratch (when needed), then the per-token <c>GemvF32</c>
    /// + scaled <c>MultiplyAdd</c> accumulates into <c>y</c>. We deliberately
    /// keep stage 2 identical so the Q8_0-with-preQuant path stays bit-
    /// equivalent (post-stage-1) to the dequant-once path used by
    /// <see cref="ApplyQ8_0B"/>.
    /// </para>
    /// <para>
    /// The numerical-equivalence claim above holds modulo Q8_1 → Q8_0
    /// cross-quantisation rounding in stage 1: <c>GemmQ8_0</c>'s
    /// <c>ComputeRows</c> path multiplies B's int8 weights by the Q8_0-encoded
    /// input scalars, where the dequant-once path multiplied dequantised F32
    /// values. The two stage-1 outputs differ by less than the Q8_0
    /// quantisation step — well within the Q8_0 LoRA tolerance bar (5e-2 abs)
    /// already documented for <see cref="LoraWeightDType.Q8_0"/>.
    /// </para>
    /// </remarks>
    /// <param name="xQ8">Pre-quantised input — same <c>seqLen × (inputDim/32)·34</c> Q8_0 byte buffer the base GEMM consumed.</param>
    /// <param name="bWeight">Q8_0 LoRA-B weight, <c>rank</c> rows of <c>(inputDim/32)·34</c> bytes each.</param>
    /// <param name="aWeight">LoRA-A weight in <paramref name="aDType"/> layout (F32 / F16 / BF16).</param>
    /// <param name="y">Output buffer; LoRA delta is accumulated in place.</param>
    /// <param name="seqLen">Input token count.</param>
    /// <param name="inputDim">Projection input dim; must be a multiple of 32.</param>
    /// <param name="outputDim">Projection output dim.</param>
    /// <param name="rank">LoRA rank (typical 8..64; no Q8_0-block constraint on rank since rank is on A's contracted axis only).</param>
    /// <param name="scale">Scaling factor — typically <c>alpha / rank</c>.</param>
    /// <param name="aDType">A-factor dtype — F32 / F16 / BF16.</param>
    /// <param name="pool">Optional thread pool — used for the Q8_0 stage-1 GEMM when seqLen warrants it.</param>
    /// <param name="aTransposedHandle">
    /// Phase 4d.6 — optional <c>[rank, outputDim]</c> row-major F32 view of A.
    /// Routes stage 2 through <see cref="LoraStage2.ApplyF32_R16"/> when
    /// <paramref name="rank"/> = 16 + AVX-512 + handle is non-zero. See
    /// <see cref="Apply(float*, void*, void*, float*, int, int, int, int, float, LoraWeightDType, LoraWeightDType, nint)"/>.
    /// </param>
    [SkipLocalsInit]
    public static void ApplyQ8_0BWithPreQuantX(
        byte* xQ8, byte* bWeight, void* aWeight, float* y,
        int seqLen, int inputDim, int outputDim, int rank, float scale,
        LoraWeightDType aDType, ComputeThreadPool? pool = null,
        nint aTransposedHandle = 0)
    {
        if (seqLen <= 0 || rank <= 0) return;
        if (inputDim % 32 != 0)
            throw new ArgumentException(
                $"Q8_0 LoRA with pre-quantised x requires inputDim multiple of 32, got {inputDim}.",
                nameof(inputDim));

        // Stage 1: tmp[seqLen, rank] = x · B^T using the integer-dot path.
        // GemmQ8_0(weightsQ8=B, m=rank, k=inputDim, n=seqLen, preQuantizedInput=xQ8)
        // skips the activation-quant step entirely (xQ8 was prepared by the
        // base projection's QuantizeInput call), so stage 1 is now pure
        // integer dot + F16-scale multiply per block — same FLOP/byte ratio
        // as the base Q8_0 GEMM.
        //
        // Threading: rank is small (typical PEFT 8..64), so the row dimension
        // M=rank doesn't tile. The Q8_0 GEMM's 2D N-partition would spread
        // ~16 tokens × 16 weight rows ≈ 256 dot products per thread per call;
        // with 32 threads on Strix Halo the per-thread work is too small to
        // amortise the pool dispatch cost (~100 us round-trip for the
        // ComputeThreadPool.Dispatch wake / wait). Single-threaded stage 1
        // mirrors what the F32 dequant-once path already does (GemmF32 with
        // M=16 also falls back to single-threaded because totalTiles<2), so
        // this isn't pessimising vs Agent 7's Phase 4d.4 dequant-once path.
        // Stage 1 work is bandwidth-bound on the activation read anyway.
        int tmpElems = seqLen * rank;
        float[] tmpBuf = ArrayPool<float>.Shared.Rent(tmpElems);
        try
        {
            fixed (float* tmp = tmpBuf)
            {
                MatMul.GemmQ8_0(bWeight, b: null, c: tmp, m: rank, k: inputDim, n: seqLen,
                                preQuantizedInput: xQ8);

                // Stage 2: y[t, o] += scale × sum_r A[o, r] · tmp[t, r].
                // Same path as the F32 LoRA's stage 2 — share the Stage2
                // helper so the post-stage-1 numerical contract matches.
                // Phase 4d.6 — when a pre-built transposed-A is available and
                // rank=16, the Stage2 dispatcher routes to LoraStage2.ApplyF32_R16
                // and the per-call A dequant is dead work (Stage2 reads only
                // from aTransposedHandle on the fast path).
                bool fastPathActive = rank == 16
                    && aTransposedHandle != 0
                    && LoraStage2.IsAvx512FastPathSupported;

                if (aDType == LoraWeightDType.F32 || fastPathActive)
                {
                    Stage2(tmp, (float*)aWeight, y, seqLen, outputDim, rank, scale, aTransposedHandle);
                    return;
                }

                long aElems = (long)outputDim * rank;
                float[] aBuf = ArrayPool<float>.Shared.Rent((int)aElems);
                try
                {
                    DequantToF32(aWeight, aDType, aBuf, (int)aElems);
                    fixed (float* aF32 = aBuf)
                    {
                        Stage2(tmp, aF32, y, seqLen, outputDim, rank, scale, aTransposedHandle);
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(aBuf);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(tmpBuf);
        }
    }

    // Per-token stage 2 (A · tmp[t]) accumulator — extracted so the Q8_0-B
    // dispatch shares the exact same scalar-equivalent path as the F32 fast
    // path above. Extracting it keeps the two paths bit-equivalent post-stage-1.
    //
    // Phase 4d.6: aTransposedHandle, when non-zero, opts into the outer-product
    // stage-2 fast path. See LoraStage2.ApplyF32_R16 — at rank=16 + AVX-512 it
    // collapses ~seqLen × outputDim short Dot calls into ~seqLen × outputDim/16
    // tile FMAs (3-4× faster on Strix Halo at typical Llama outputDims).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Stage2(float* tmp, float* aF32, float* y,
                               int seqLen, int outputDim, int rank, float scale,
                               nint aTransposedHandle = 0)
    {
        // Phase 4d.6 fast path — rank=16, AVX-512, transposed-A available.
        if (rank == 16 && aTransposedHandle != 0 && LoraStage2.IsAvx512FastPathSupported)
        {
            LoraStage2.ApplyF32_R16(
                (float*)aTransposedHandle, tmp, y, seqLen, outputDim, scale);
            return;
        }

        float[] deltaBuf = ArrayPool<float>.Shared.Rent(outputDim);
        try
        {
            fixed (float* delta = deltaBuf)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    MatMul.GemvF32(aF32, tmp + t * rank, delta, outputDim, rank);

                    var deltaSpan = new ReadOnlySpan<float>(delta, outputDim);
                    var ySpan = new Span<float>(y + t * outputDim, outputDim);
                    TensorPrimitives.MultiplyAdd(deltaSpan, scale, ySpan, ySpan);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(deltaBuf);
        }
    }

    private static void DequantToF32(void* src, LoraWeightDType dtype, float[] dst, int count)
    {
        switch (dtype)
        {
            case LoraWeightDType.F32:
                {
                    var srcSpan = new ReadOnlySpan<float>(src, count);
                    srcSpan.CopyTo(dst.AsSpan(0, count));
                    break;
                }
            case LoraWeightDType.F16:
                {
                    var srcSpan = new ReadOnlySpan<Half>(src, count);
                    TensorPrimitives.ConvertToSingle(srcSpan, dst.AsSpan(0, count));
                    break;
                }
            case LoraWeightDType.BF16:
                {
                    byte* p = (byte*)src;
                    for (int i = 0; i < count; i++)
                    {
                        ushort raw = BinaryPrimitives.ReadUInt16LittleEndian(
                            new ReadOnlySpan<byte>(p + i * 2, 2));
                        uint asF32 = (uint)raw << 16;
                        dst[i] = BitConverter.UInt32BitsToSingle(asF32);
                    }
                    break;
                }
            default:
                throw new NotSupportedException(
                    $"LoRA weight dtype {dtype} is not supported by LoraDelta.Apply.");
        }
    }

    /// <summary>
    /// Convenience overload using <see cref="ReadOnlySpan{T}"/> / <see cref="Span{T}"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Apply(ReadOnlySpan<float> x, ReadOnlySpan<float> bWeight, ReadOnlySpan<float> aWeight,
                             Span<float> y, int seqLen, int inputDim, int outputDim, int rank, float scale)
    {
        if (x.Length < seqLen * inputDim)
            throw new ArgumentException($"x span too small: {x.Length} < {seqLen * inputDim}", nameof(x));
        if (bWeight.Length < rank * inputDim)
            throw new ArgumentException($"bWeight span too small: {bWeight.Length} < {rank * inputDim}", nameof(bWeight));
        if (aWeight.Length < outputDim * rank)
            throw new ArgumentException($"aWeight span too small: {aWeight.Length} < {outputDim * rank}", nameof(aWeight));
        if (y.Length < seqLen * outputDim)
            throw new ArgumentException($"y span too small: {y.Length} < {seqLen * outputDim}", nameof(y));

        fixed (float* xPtr = x)
        fixed (float* bPtr = bWeight)
        fixed (float* aPtr = aWeight)
        fixed (float* yPtr = y)
        {
            Apply(xPtr, bPtr, aPtr, yPtr, seqLen, inputDim, outputDim, rank, scale);
        }
    }

    /// <summary>
    /// One-shot adapter-load Q8_0 quantiser for the LoRA B factor. Quantises
    /// the F32 source row-by-row into the destination buffer, where each row
    /// holds <paramref name="elementsPerRow"/> floats (must be a multiple of
    /// 32). Output layout matches dotLLM's GGUF Q8_0 layout: per row,
    /// <c>(elementsPerRow / 32)</c> blocks of 34 bytes each = 2-byte F16
    /// scale + 32 sbytes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is invoked at adapter load time only — once per
    /// <c>(layer, projection)</c> pair when the loader opts into Q8_0
    /// storage for B. Per-row quantisation reuses
    /// <see cref="MatMul.QuantizeF32ToQ8_0(float*, byte*, int)"/> which
    /// already has the AVX-512 / AVX2 / scalar fallbacks; the per-row loop
    /// is intentionally not parallelised because adapter-load is one-shot.
    /// </para>
    /// <para>
    /// Reverse direction (Q8_0 → F32) is not provided here — the runtime
    /// kernel consumes Q8_0 directly via the integer-dot path. A round-trip
    /// dequant is only needed by parity tests, which call
    /// <see cref="DequantizeRowToF32"/> for that purpose.
    /// </para>
    /// </remarks>
    /// <param name="srcF32">Source F32 buffer, <paramref name="rows"/> × <paramref name="elementsPerRow"/> elements.</param>
    /// <param name="dstQ8">Destination Q8_0 buffer, sized
    /// <c>rows × (elementsPerRow / 32) × 34</c> bytes.</param>
    /// <param name="rows">Number of rows (typically <c>rank</c>).</param>
    /// <param name="elementsPerRow">Elements per row (typically <c>inputDim</c>); must be a multiple of 32.</param>
    public static void Quantize_F32_To_Q8_0(float* srcF32, byte* dstQ8,
                                            int rows, int elementsPerRow)
    {
        if (rows < 0)
            throw new ArgumentOutOfRangeException(nameof(rows), rows, "Row count must be non-negative.");
        if (elementsPerRow <= 0 || elementsPerRow % 32 != 0)
            throw new ArgumentException(
                $"elementsPerRow must be a positive multiple of 32, got {elementsPerRow}.",
                nameof(elementsPerRow));

        int rowBytes = (elementsPerRow / 32) * 34;
        for (int row = 0; row < rows; row++)
        {
            MatMul.QuantizeF32ToQ8_0(srcF32 + (long)row * elementsPerRow,
                                      dstQ8 + (long)row * rowBytes,
                                      elementsPerRow);
        }
    }

    /// <summary>
    /// Round-trip helper used by parity tests — dequantises one Q8_0 row
    /// (<paramref name="elementsPerRow"/> elements split into 32-element
    /// blocks) back to F32. Not used on the inference hot path.
    /// </summary>
    public static void DequantizeRowToF32(byte* srcQ8, float* dstF32, int elementsPerRow)
    {
        if (elementsPerRow <= 0 || elementsPerRow % 32 != 0)
            throw new ArgumentException(
                $"elementsPerRow must be a positive multiple of 32, got {elementsPerRow}.",
                nameof(elementsPerRow));

        int blockCount = elementsPerRow / 32;
        for (int block = 0; block < blockCount; block++)
        {
            byte* blockSrc = srcQ8 + block * 34;
            float scale = (float)Unsafe.ReadUnaligned<Half>(blockSrc);
            sbyte* qs = (sbyte*)(blockSrc + 2);
            float* dstBlock = dstF32 + block * 32;
            for (int i = 0; i < 32; i++)
                dstBlock[i] = qs[i] * scale;
        }
    }
}
