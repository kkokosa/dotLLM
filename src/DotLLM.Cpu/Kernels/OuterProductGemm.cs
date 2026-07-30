using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// F32 outer-product tiled GEMM kernels for prefill (multi-token) workloads.
/// Companion to the Q8_0/Q5_0/K-quant outer-product kernels in <see cref="MatMul"/>.
/// </summary>
/// <remarks>
/// <para>
/// dotLLM convention: <c>C[N,M] = B[N,K] × A[M,K]^T</c>.
/// <list type="bullet">
/// <item><c>A</c> is the weight matrix <c>[M,K]</c> in row-major.</item>
/// <item><c>B</c> is the activation matrix <c>[N,K]</c> in row-major.</item>
/// <item><c>C</c> is the output matrix <c>[N,M]</c> in row-major.</item>
/// </list>
/// </para>
/// <para>
/// The outer-product formulation processes an <c>M_R × N_R</c> register tile per K-step
/// (here 4 weight rows × 3 input tokens) so each loaded activation vector is reused
/// <c>M_R</c> times and each loaded weight vector is reused <c>N_R</c> times. Reduces
/// the per-FMA memory bandwidth requirement vs the standard inner-product GEMM that
/// computes one (row, token) cell to completion before moving on.
/// </para>
/// <para>
/// AVX2 register accounting for the 4×3 tile (within 16-YMM budget):
/// <list type="bullet">
/// <item>12 accumulators (one per <c>(row, token)</c> cell)</item>
/// <item>3 token vectors (one per input row, reloaded each K-step)</item>
/// <item>1 weight vector (reloaded for each of the 4 weight rows)</item>
/// </list>
/// Unlike the Q8_0 variant (PR #61 — blocked at 23 YMM due to the dequant artifacts),
/// the F32 path has no <c>ones</c> mask, no scale extraction, and no <c>Half→float</c>
/// conversion, so the 4×3 tile fits the AVX2 register file naturally.
/// </para>
/// <para>
/// Vectorization is along the K dimension (the contraction axis), 8 floats per
/// AVX2 step. The K-tail (last <c>K % 8</c> elements) is handled scalar.
/// </para>
/// <para>
/// This kernel is a new, independent code path. Callers continue to use
/// <see cref="MatMul.GemmF32(float*, float*, float*, int, int, int)"/> until a
/// benchmark-driven decision is made to switch dispatch — that wiring is a separate PR.
/// </para>
/// </remarks>
public static unsafe class OuterProductGemm
{
    /// <summary>Register-tile rows (weight-row count processed per microkernel call).</summary>
    private const int TileRows = 4;

    /// <summary>Register-tile tokens (input-row count processed per microkernel call).</summary>
    private const int TileTokens = 3;

    /// <summary>SIMD lane width for AVX2 (<see cref="Vector256{Single}"/>).</summary>
    private const int LaneAvx2 = 8;

    /// <summary>
    /// Scalar reference outer-product GEMM. Computes <c>C[N,M] = B[N,K] × A[M,K]^T</c>
    /// by iterating M_R × N_R register tiles and accumulating contributions over K.
    /// Pure scalar — present as a correctness oracle for the vector variant.
    /// </summary>
    /// <param name="a">Weight matrix <c>[M,K]</c> row-major.</param>
    /// <param name="b">Input matrix <c>[N,K]</c> row-major.</param>
    /// <param name="c">Output matrix <c>[N,M]</c> row-major.</param>
    /// <param name="m">Number of weight rows (output dim).</param>
    /// <param name="k">Contraction dim.</param>
    /// <param name="n">Number of input tokens (batch dim).</param>
    [SkipLocalsInit]
    public static void OuterProductGemmF32Scalar(float* a, float* b, float* c, int m, int k, int n)
    {
        // Every C cell is assigned exactly once across the three nested passes
        // (full tiles ∪ row tail ∪ token tail = [0,n) × [0,m), no overlap, no
        // gap) — no pre-zero of C is required.

        // Iterate output tiles of size TileTokens × TileRows. Tail rows/tokens
        // fall through to a 1×1 inner-product cleanup so every shape is handled.
        int mFullEnd = (m / TileRows) * TileRows;
        int nFullEnd = (n / TileTokens) * TileTokens;

        for (int tStart = 0; tStart < nFullEnd; tStart += TileTokens)
        {
            for (int rStart = 0; rStart < mFullEnd; rStart += TileRows)
            {
                TileScalar4x3(a, b, c, rStart, tStart, k, m);
            }

            // Row tail (rStart in [mFullEnd, m)).
            for (int rStart = mFullEnd; rStart < m; rStart++)
            {
                for (int tOff = 0; tOff < TileTokens; tOff++)
                {
                    int t = tStart + tOff;
                    c[t * m + rStart] = DotScalar(a + (long)rStart * k, b + (long)t * k, k);
                }
            }
        }

        // Token tail (tStart in [nFullEnd, n)) — process remaining tokens as 1×TileRows
        // tiles, then per-row scalar for the row tail.
        for (int tStart = nFullEnd; tStart < n; tStart++)
        {
            for (int rStart = 0; rStart < m; rStart++)
            {
                c[tStart * m + rStart] = DotScalar(a + (long)rStart * k, b + (long)tStart * k, k);
            }
        }
    }

    /// <summary>
    /// AVX2 outer-product GEMM. Falls back to <see cref="OuterProductGemmF32Scalar"/>
    /// when AVX2/FMA are unavailable. Bit-exact-modulo-FP-order with the scalar variant
    /// to within typical FMA-vs-mul-then-add rounding (≤ a few ULP for typical inputs).
    /// </summary>
    /// <param name="a">Weight matrix <c>[M,K]</c> row-major.</param>
    /// <param name="b">Input matrix <c>[N,K]</c> row-major.</param>
    /// <param name="c">Output matrix <c>[N,M]</c> row-major.</param>
    /// <param name="m">Number of weight rows (output dim).</param>
    /// <param name="k">Contraction dim.</param>
    /// <param name="n">Number of input tokens (batch dim).</param>
    [SkipLocalsInit]
    public static void OuterProductGemmF32(float* a, float* b, float* c, int m, int k, int n)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported)
        {
            OuterProductGemmF32Scalar(a, b, c, m, k, n);
            return;
        }

        // Every C cell is assigned exactly once across the three nested passes
        // (full tiles ∪ row tail ∪ token tail = [0,n) × [0,m), no overlap, no
        // gap) — no pre-zero of C is required.

        int mFullEnd = (m / TileRows) * TileRows;
        int nFullEnd = (n / TileTokens) * TileTokens;
        int kVec = (k / LaneAvx2) * LaneAvx2;

        for (int tStart = 0; tStart < nFullEnd; tStart += TileTokens)
        {
            for (int rStart = 0; rStart < mFullEnd; rStart += TileRows)
            {
                TileAvx2_4x3(a, b, c, rStart, tStart, k, m, kVec);
            }

            // Row tail — fall back to standard inner-product per cell. Vectorised
            // via Vector256 dot-product, mirroring the K-tail strategy.
            for (int rStart = mFullEnd; rStart < m; rStart++)
            {
                for (int tOff = 0; tOff < TileTokens; tOff++)
                {
                    int t = tStart + tOff;
                    c[t * m + rStart] = DotAvx2(a + (long)rStart * k, b + (long)t * k, k, kVec);
                }
            }
        }

        // Token tail — remaining tokens (n % TileTokens). Same row-tail vectorised
        // dot product path covers full and partial row segments uniformly.
        for (int tStart = nFullEnd; tStart < n; tStart++)
        {
            for (int rStart = 0; rStart < m; rStart++)
            {
                c[tStart * m + rStart] = DotAvx2(a + (long)rStart * k, b + (long)tStart * k, k, kVec);
            }
        }
    }

    // ──────────────────── Scalar microkernel ────────────────────

    /// <summary>
    /// Scalar 4×3 microkernel: accumulates contributions of one weight-tile (4 rows)
    /// × one token-tile (3 tokens) over the full K range.
    /// </summary>
    [SkipLocalsInit]
    private static void TileScalar4x3(
        float* a, float* b, float* c,
        int rStart, int tStart, int k, int m)
    {
        // 12 accumulators (4 rows × 3 tokens).
        float acc00 = 0, acc01 = 0, acc02 = 0;
        float acc10 = 0, acc11 = 0, acc12 = 0;
        float acc20 = 0, acc21 = 0, acc22 = 0;
        float acc30 = 0, acc31 = 0, acc32 = 0;

        float* aRow0 = a + (long)(rStart + 0) * k;
        float* aRow1 = a + (long)(rStart + 1) * k;
        float* aRow2 = a + (long)(rStart + 2) * k;
        float* aRow3 = a + (long)(rStart + 3) * k;

        float* bTok0 = b + (long)(tStart + 0) * k;
        float* bTok1 = b + (long)(tStart + 1) * k;
        float* bTok2 = b + (long)(tStart + 2) * k;

        for (int i = 0; i < k; i++)
        {
            float b0 = bTok0[i];
            float b1 = bTok1[i];
            float b2 = bTok2[i];

            float a0 = aRow0[i];
            acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2;

            float a1 = aRow1[i];
            acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2;

            float a2 = aRow2[i];
            acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2;

            float a3 = aRow3[i];
            acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2;
        }

        // C[t,r] layout: c[t * m + r].
        c[(long)(tStart + 0) * m + rStart + 0] = acc00;
        c[(long)(tStart + 0) * m + rStart + 1] = acc10;
        c[(long)(tStart + 0) * m + rStart + 2] = acc20;
        c[(long)(tStart + 0) * m + rStart + 3] = acc30;

        c[(long)(tStart + 1) * m + rStart + 0] = acc01;
        c[(long)(tStart + 1) * m + rStart + 1] = acc11;
        c[(long)(tStart + 1) * m + rStart + 2] = acc21;
        c[(long)(tStart + 1) * m + rStart + 3] = acc31;

        c[(long)(tStart + 2) * m + rStart + 0] = acc02;
        c[(long)(tStart + 2) * m + rStart + 1] = acc12;
        c[(long)(tStart + 2) * m + rStart + 2] = acc22;
        c[(long)(tStart + 2) * m + rStart + 3] = acc32;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotScalar(float* a, float* b, int k)
    {
        float sum = 0;
        for (int i = 0; i < k; i++)
            sum += a[i] * b[i];
        return sum;
    }

    // ──────────────────── AVX2 microkernel ────────────────────

    /// <summary>
    /// AVX2 4×3 microkernel. Vectorises along K (8 floats/lane) with 12 accumulators
    /// (one per <c>(row, token)</c> cell). At each K-step we load 3 token vectors and
    /// 4 weight-row vectors, sharing each load across the orthogonal dimension via
    /// 12 FMAs.
    /// </summary>
    /// <remarks>
    /// Register inventory (16 YMM):
    /// <list type="bullet">
    /// <item>12 accumulators</item>
    /// <item>3 token vectors (vb0, vb1, vb2)</item>
    /// <item>1 weight vector (reloaded for each of the 4 rows)</item>
    /// </list>
    /// </remarks>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void TileAvx2_4x3(
        float* a, float* b, float* c,
        int rStart, int tStart, int k, int m, int kVec)
    {
        Vector256<float> acc00 = Vector256<float>.Zero, acc01 = Vector256<float>.Zero, acc02 = Vector256<float>.Zero;
        Vector256<float> acc10 = Vector256<float>.Zero, acc11 = Vector256<float>.Zero, acc12 = Vector256<float>.Zero;
        Vector256<float> acc20 = Vector256<float>.Zero, acc21 = Vector256<float>.Zero, acc22 = Vector256<float>.Zero;
        Vector256<float> acc30 = Vector256<float>.Zero, acc31 = Vector256<float>.Zero, acc32 = Vector256<float>.Zero;

        float* aRow0 = a + (long)(rStart + 0) * k;
        float* aRow1 = a + (long)(rStart + 1) * k;
        float* aRow2 = a + (long)(rStart + 2) * k;
        float* aRow3 = a + (long)(rStart + 3) * k;

        float* bTok0 = b + (long)(tStart + 0) * k;
        float* bTok1 = b + (long)(tStart + 1) * k;
        float* bTok2 = b + (long)(tStart + 2) * k;

        for (int i = 0; i < kVec; i += LaneAvx2)
        {
            // Load 3 token vectors (held in registers across the row loop).
            Vector256<float> vb0 = Vector256.Load(bTok0 + i);
            Vector256<float> vb1 = Vector256.Load(bTok1 + i);
            Vector256<float> vb2 = Vector256.Load(bTok2 + i);

            // Row 0: load A row vector, do 3 FMAs sharing it across tokens.
            Vector256<float> va = Vector256.Load(aRow0 + i);
            acc00 = Fma.MultiplyAdd(va, vb0, acc00);
            acc01 = Fma.MultiplyAdd(va, vb1, acc01);
            acc02 = Fma.MultiplyAdd(va, vb2, acc02);

            // Row 1.
            va = Vector256.Load(aRow1 + i);
            acc10 = Fma.MultiplyAdd(va, vb0, acc10);
            acc11 = Fma.MultiplyAdd(va, vb1, acc11);
            acc12 = Fma.MultiplyAdd(va, vb2, acc12);

            // Row 2.
            va = Vector256.Load(aRow2 + i);
            acc20 = Fma.MultiplyAdd(va, vb0, acc20);
            acc21 = Fma.MultiplyAdd(va, vb1, acc21);
            acc22 = Fma.MultiplyAdd(va, vb2, acc22);

            // Row 3.
            va = Vector256.Load(aRow3 + i);
            acc30 = Fma.MultiplyAdd(va, vb0, acc30);
            acc31 = Fma.MultiplyAdd(va, vb1, acc31);
            acc32 = Fma.MultiplyAdd(va, vb2, acc32);
        }

        // Horizontal-reduce each accumulator into the corresponding C cell.
        float s00 = HorizontalSum(acc00), s01 = HorizontalSum(acc01), s02 = HorizontalSum(acc02);
        float s10 = HorizontalSum(acc10), s11 = HorizontalSum(acc11), s12 = HorizontalSum(acc12);
        float s20 = HorizontalSum(acc20), s21 = HorizontalSum(acc21), s22 = HorizontalSum(acc22);
        float s30 = HorizontalSum(acc30), s31 = HorizontalSum(acc31), s32 = HorizontalSum(acc32);

        // K-tail (scalar).
        for (int i = kVec; i < k; i++)
        {
            float b0 = bTok0[i], b1 = bTok1[i], b2 = bTok2[i];
            float a0 = aRow0[i]; s00 += a0 * b0; s01 += a0 * b1; s02 += a0 * b2;
            float a1 = aRow1[i]; s10 += a1 * b0; s11 += a1 * b1; s12 += a1 * b2;
            float a2 = aRow2[i]; s20 += a2 * b0; s21 += a2 * b1; s22 += a2 * b2;
            float a3 = aRow3[i]; s30 += a3 * b0; s31 += a3 * b1; s32 += a3 * b2;
        }

        c[(long)(tStart + 0) * m + rStart + 0] = s00;
        c[(long)(tStart + 0) * m + rStart + 1] = s10;
        c[(long)(tStart + 0) * m + rStart + 2] = s20;
        c[(long)(tStart + 0) * m + rStart + 3] = s30;

        c[(long)(tStart + 1) * m + rStart + 0] = s01;
        c[(long)(tStart + 1) * m + rStart + 1] = s11;
        c[(long)(tStart + 1) * m + rStart + 2] = s21;
        c[(long)(tStart + 1) * m + rStart + 3] = s31;

        c[(long)(tStart + 2) * m + rStart + 0] = s02;
        c[(long)(tStart + 2) * m + rStart + 1] = s12;
        c[(long)(tStart + 2) * m + rStart + 2] = s22;
        c[(long)(tStart + 2) * m + rStart + 3] = s32;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotAvx2(float* a, float* b, int k, int kVec)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        for (int i = 0; i < kVec; i += LaneAvx2)
            acc = Fma.MultiplyAdd(Vector256.Load(a + i), Vector256.Load(b + i), acc);
        float sum = HorizontalSum(acc);
        for (int i = kVec; i < k; i++)
            sum += a[i] * b[i];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalSum(Vector256<float> v)
    {
        // 8 → 4 via cross-lane add, then 4 → 2 via hadd, then 2 → 1 via hadd.
        Vector128<float> lo = v.GetLower();
        Vector128<float> hi = v.GetUpper();
        Vector128<float> sum128 = Sse.Add(lo, hi);
        sum128 = Sse3.HorizontalAdd(sum128, sum128);
        sum128 = Sse3.HorizontalAdd(sum128, sum128);
        return sum128.ToScalar();
    }
}
