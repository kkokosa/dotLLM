using DotLLM.Metal;
using DotLLM.Metal.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

// ── GEMM FP32 ────────────────────────────────────────────────────────────────

public sealed class GemmF32Tests
{
    private const float Tol = 1e-4f;

    /// <summary>
    /// Scalar CPU reference: C = alpha · op(A) · op(B) + beta · C.
    /// Operates directly on the storage arrays; transpose flags determine indexing.
    /// </summary>
    private static float[] CpuGemm(
        float[] a, float[] b, float[]? cInit,
        int m, int n, int k,
        bool transposeA, bool transposeB,
        float alpha, float beta)
    {
        float[] c = cInit is null ? new float[m * n] : (float[])cInit.Clone();

        for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            float acc = 0f;
            for (int p = 0; p < k; p++)
            {
                // op(A)[i, p]
                float aIp = transposeA ? a[p * m + i] : a[i * k + p];
                // op(B)[p, j]
                float bPj = transposeB ? b[j * k + p] : b[p * n + j];
                acc += aIp * bPj;
            }
            c[i * n + j] = alpha * acc + beta * c[i * n + j];
        }
        return c;
    }

    private static void AssertClose(float[] expected, float[] actual, float tol = Tol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], tol);
    }

    // ── Basic non-transposed: C = A · B ──────────────────────────────────────

    [MetalTestFact]
    public void NoTranspose_SmallMatrix_MatchesCpu()
    {
        const int m = 3, k = 4, n = 5;
        var rng = new Random(1);

        float[] a = new float[m * k];
        float[] b = new float[k * n];
        float[] c = new float[m * n];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;

        float[] expected = CpuGemm(a, b, null, m, n, k, false, false, 1f, 0f);

        using var ctx = new MetalContext();
        Gemm.ExecuteF32(ctx, a, b, c, m, n, k, transposeA: false, transposeB: false);

        AssertClose(expected, c);
    }

    // ── LLM projection layout: Y = X · Wᵀ, W stored as [N, K] ────────────────

    [MetalTestFact]
    public void TransposeRight_LlamaProjection_MatchesCpu()
    {
        const int m = 4;     // seqLen
        const int k = 16;    // hiddenDim
        const int n = 32;    // outputDim
        var rng = new Random(2);

        float[] x = new float[m * k];                 // X : [m, k]
        float[] w = new float[n * k];                 // W : [n, k] (stored transposed)
        float[] y = new float[m * n];                 // Y : [m, n]
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < w.Length; i++) w[i] = rng.NextSingle() * 2f - 1f;

        float[] expected = CpuGemm(x, w, null, m, n, k, false, true, 1f, 0f);

        using var ctx = new MetalContext();
        Gemm.ExecuteF32(ctx, x, w, y, m, n, k, transposeA: false, transposeB: true);

        AssertClose(expected, y);
    }

    // ── Both transposed ──────────────────────────────────────────────────────

    [MetalTestFact]
    public void BothTransposed_MatchesCpu()
    {
        const int m = 4, k = 6, n = 5;
        var rng = new Random(3);

        // A stored as [k, m]; B stored as [n, k]
        float[] a = new float[k * m];
        float[] b = new float[n * k];
        float[] c = new float[m * n];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;

        float[] expected = CpuGemm(a, b, null, m, n, k, true, true, 1f, 0f);

        using var ctx = new MetalContext();
        Gemm.ExecuteF32(ctx, a, b, c, m, n, k, transposeA: true, transposeB: true);

        AssertClose(expected, c);
    }

    // ── alpha != 1, beta != 0 (accumulate into existing C) ───────────────────

    [MetalTestFact]
    public void AlphaBeta_AccumulatesIntoC()
    {
        const int m = 3, k = 4, n = 3;
        var rng = new Random(4);

        float[] a = new float[m * k];
        float[] b = new float[k * n];
        float[] c = new float[m * n];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < c.Length; i++) c[i] = rng.NextSingle() * 2f - 1f;

        const float alpha = 0.7f, beta = 0.5f;
        float[] expected = CpuGemm(a, b, c, m, n, k, false, false, alpha, beta);

        using var ctx = new MetalContext();
        Gemm.ExecuteF32(ctx, a, b, c, m, n, k, alpha: alpha, beta: beta);

        AssertClose(expected, c);
    }

    // ── Identity check: I · B = B ────────────────────────────────────────────

    [MetalTestFact]
    public void IdentityTimesMatrix_ReturnsMatrix()
    {
        const int n = 8, k = 8, m = n;
        var rng = new Random(5);

        float[] eye = new float[n * n];
        for (int i = 0; i < n; i++) eye[i * n + i] = 1f;

        float[] b = new float[k * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;

        float[] c = new float[m * n];
        using var ctx = new MetalContext();
        Gemm.ExecuteF32(ctx, eye, b, c, m, n, k);

        AssertClose(b, c);
    }

    // ── Large matrices (Llama-7B FFN dimensions) ─────────────────────────────

    [MetalTestFact]
    public void LargeMatrix_LlamaFfnSize_MatchesCpu()
    {
        // Tiny version of FFN: seqLen × hidden · hidden × inter (transposed)
        const int m = 8;     // seqLen
        const int k = 64;    // hidden
        const int n = 128;   // inter
        var rng = new Random(42);

        float[] x = new float[m * k];
        float[] w = new float[n * k];
        float[] y = new float[m * n];
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < w.Length; i++) w[i] = rng.NextSingle() * 2f - 1f;

        float[] expected = CpuGemm(x, w, null, m, n, k, false, true, 1f, 0f);

        using var ctx = new MetalContext();
        Gemm.ExecuteF32(ctx, x, w, y, m, n, k, transposeB: true);

        AssertClose(expected, y, tol: 1e-3f); // larger k → more accumulation
    }
}

// ── GEMM FP16 ────────────────────────────────────────────────────────────────

public sealed class GemmF16Tests
{
    // FP16 accumulation tolerance — relative because rounding error scales with magnitude.
    private static void AssertCloseRel(float[] expected, Half[] actual, float relTol = 5e-3f, float floor = 1e-3f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = (float)actual[i];
            float tol = MathF.Max(MathF.Abs(e) * relTol, floor);
            Assert.True(MathF.Abs(e - a) <= tol,
                $"index {i}: expected={e}, actual={a}, diff={MathF.Abs(e - a)}, tol={tol}");
        }
    }

    private static float[] CpuGemmF32Reference(
        Half[] a, Half[] b,
        int m, int n, int k,
        bool transposeA, bool transposeB)
    {
        float[] c = new float[m * n];
        for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            float acc = 0f;
            for (int p = 0; p < k; p++)
            {
                float aIp = transposeA ? (float)a[p * m + i] : (float)a[i * k + p];
                float bPj = transposeB ? (float)b[j * k + p] : (float)b[p * n + j];
                acc += aIp * bPj;
            }
            c[i * n + j] = acc;
        }
        return c;
    }

    [MetalTestFact]
    public void NoTranspose_SmallMatrix_MatchesCpu()
    {
        const int m = 3, k = 4, n = 5;
        var rng = new Random(1);

        Half[] a = new Half[m * k];
        Half[] b = new Half[k * n];
        Half[] c = new Half[m * n];
        for (int i = 0; i < a.Length; i++) a[i] = (Half)(rng.NextSingle() * 2f - 1f);
        for (int i = 0; i < b.Length; i++) b[i] = (Half)(rng.NextSingle() * 2f - 1f);

        float[] expected = CpuGemmF32Reference(a, b, m, n, k, false, false);

        using var ctx = new MetalContext();
        Gemm.ExecuteF16(ctx, a, b, c, m, n, k);

        AssertCloseRel(expected, c);
    }

    [MetalTestFact]
    public void TransposeRight_LlamaProjection_MatchesCpu()
    {
        const int m = 4, k = 16, n = 32;
        var rng = new Random(2);

        Half[] x = new Half[m * k];
        Half[] w = new Half[n * k];
        Half[] y = new Half[m * n];
        for (int i = 0; i < x.Length; i++) x[i] = (Half)(rng.NextSingle() * 2f - 1f);
        for (int i = 0; i < w.Length; i++) w[i] = (Half)(rng.NextSingle() * 2f - 1f);

        float[] expected = CpuGemmF32Reference(x, w, m, n, k, false, true);

        using var ctx = new MetalContext();
        Gemm.ExecuteF16(ctx, x, w, y, m, n, k, transposeB: true);

        AssertCloseRel(expected, y);
    }

    [MetalTestFact]
    public void LargeMatrix_LlamaFfnSize_MatchesCpu()
    {
        const int m = 8, k = 64, n = 128;
        var rng = new Random(42);

        Half[] x = new Half[m * k];
        Half[] w = new Half[n * k];
        Half[] y = new Half[m * n];
        // Smaller magnitudes to keep FP16 accumulation in range
        for (int i = 0; i < x.Length; i++) x[i] = (Half)((rng.NextSingle() * 2f - 1f) * 0.3f);
        for (int i = 0; i < w.Length; i++) w[i] = (Half)((rng.NextSingle() * 2f - 1f) * 0.3f);

        float[] expected = CpuGemmF32Reference(x, w, m, n, k, false, true);

        using var ctx = new MetalContext();
        Gemm.ExecuteF16(ctx, x, w, y, m, n, k, transposeB: true);

        // FP16 accumulation over k=64 → up to ~1% relative error.
        AssertCloseRel(expected, y, relTol: 1e-2f, floor: 5e-3f);
    }
}
