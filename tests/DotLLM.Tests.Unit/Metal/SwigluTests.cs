using DotLLM.Cpu.Kernels;
using DotLLM.Metal;
using DotLLM.Metal.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

// ── SwigluF32 ─────────────────────────────────────────────────────────────────

public sealed class SwigluF32Tests
{
    private const float Tol = 1e-5f;

    /// <summary>CPU reference via <see cref="FusedOps.SwiGLUScalar"/>.</summary>
    private static float[] CpuRef(float[] gate, float[] up)
    {
        float[] result = new float[gate.Length];
        FusedOps.SwiGLUScalar(gate, up, result);
        return result;
    }

    [MetalTestFact]
    public void KnownValues_MatchExpected()
    {
        // SiLU(0) = 0 * sigmoid(0) = 0 → result = 0 * up
        // SiLU(1) = 1 * sigmoid(1) ≈ 0.7311 → result ≈ 0.7311 * 2 ≈ 1.4623
        float[] gate   = [0f, 1f, -1f, 2f];
        float[] up     = [1f, 2f,  1f, 1f];
        float[] result = new float[4];

        using var ctx = new MetalContext();
        SwigluF32.Execute(ctx, gate, up, result);

        float[] expected = CpuRef(gate, up);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], result[i], Tol);
    }

    [MetalTestFact]
    public void ZeroGate_ProducesZero()
    {
        float[] gate   = [0f, 0f, 0f, 0f];
        float[] up     = [1f, 2f, 3f, 4f];
        float[] result = new float[4];

        using var ctx = new MetalContext();
        SwigluF32.Execute(ctx, gate, up, result);

        Assert.All(result, v => Assert.Equal(0f, v, Tol));
    }

    [MetalTestFact]
    public void ZeroUp_ProducesZero()
    {
        float[] gate   = [1f, 2f, 3f, 4f];
        float[] up     = [0f, 0f, 0f, 0f];
        float[] result = new float[4];

        using var ctx = new MetalContext();
        SwigluF32.Execute(ctx, gate, up, result);

        Assert.All(result, v => Assert.Equal(0f, v, Tol));
    }

    [MetalTestFact]
    public void ScalarReference_MatchesCpu()
    {
        var rng = new Random(42);
        const int n = 1024;
        float[] gate   = new float[n];
        float[] up     = new float[n];
        float[] result = new float[n];

        for (int i = 0; i < n; i++) gate[i] = rng.NextSingle() * 4f - 2f;
        for (int i = 0; i < n; i++) up[i]   = rng.NextSingle() * 4f - 2f;

        float[] expected = CpuRef(gate, up);

        using var ctx = new MetalContext();
        SwigluF32.Execute(ctx, gate, up, result);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], result[i], Tol);
    }

    [MetalTestFact]
    public void EmptySpans_ReturnsSilently()
    {
        using var ctx = new MetalContext();
        SwigluF32.Execute(ctx, [], [], []);
    }

    [MetalTestFact]
    public void LengthMismatch_ThrowsArgumentException()
    {
        using var ctx = new MetalContext();
        Assert.Throws<ArgumentException>(() =>
            SwigluF32.Execute(ctx, [1f, 2f], [1f], new float[2]));
    }

    [MetalTestFact]
    public void ResultTooSmall_ThrowsArgumentException()
    {
        using var ctx = new MetalContext();
        Assert.Throws<ArgumentException>(() =>
            SwigluF32.Execute(ctx, [1f, 2f], [1f, 2f], new float[1]));
    }
}

// ── SwigluF16 ─────────────────────────────────────────────────────────────────

public sealed class SwigluF16Tests
{
    // Tolerance: 1e-2f — FP16 truncation on gate/up (~0.1% relative) + exp precision
    private const float Tol = 1e-2f;

    private static Half[] H(params float[] values) =>
        Array.ConvertAll(values, v => (Half)v);

    /// <summary>
    /// CPU reference: Half→float (lossless), <see cref="FusedOps.SwiGLUScalar"/>, compare vs Half output.
    /// </summary>
    private static float[] CpuRef(Half[] gate, Half[] up)
    {
        float[] gF = Array.ConvertAll(gate, h => (float)h);
        float[] uF = Array.ConvertAll(up,   h => (float)h);
        float[] r  = new float[gate.Length];
        FusedOps.SwiGLUScalar(gF, uF, r);
        return r;
    }

    [MetalTestFact]
    public void KnownValues_MatchCpu()
    {
        Half[] gate   = H(0f, 1f, -1f, 2f);
        Half[] up     = H(1f, 2f,  1f, 1f);
        Half[] result = new Half[4];

        using var ctx = new MetalContext();
        SwigluF16.Execute(ctx, gate, up, result);

        float[] expected = CpuRef(gate, up);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], (float)result[i], Tol);
    }

    [MetalTestFact]
    public void OddLength_HandlesTrailingElement()
    {
        // length=5: half2 covers [0,1] and [2,3]; element 4 is the odd tail
        Half[] gate   = H(1f, -1f, 0.5f, -0.5f, 2f);
        Half[] up     = H(2f,  2f, 2f,    2f,   1f);
        Half[] result = new Half[5];

        using var ctx = new MetalContext();
        SwigluF16.Execute(ctx, gate, up, result);

        float[] expected = CpuRef(gate, up);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], (float)result[i], Tol);
    }

    [MetalTestFact]
    public void ZeroGate_ProducesZero()
    {
        Half[] gate   = H(0f, 0f, 0f, 0f);
        Half[] up     = H(1f, 2f, 3f, 4f);
        Half[] result = new Half[4];

        using var ctx = new MetalContext();
        SwigluF16.Execute(ctx, gate, up, result);

        Assert.All(result, v => Assert.Equal(0f, (float)v, Tol));
    }

    [MetalTestFact]
    public void ScalarReference_MatchesCpu()
    {
        var rng = new Random(7);
        const int n = 1024;
        Half[] gate   = new Half[n];
        Half[] up     = new Half[n];
        Half[] result = new Half[n];

        for (int i = 0; i < n; i++) gate[i] = (Half)(rng.NextSingle() * 4f - 2f);
        for (int i = 0; i < n; i++) up[i]   = (Half)(rng.NextSingle() * 4f - 2f);

        float[] expected = CpuRef(gate, up);

        using var ctx = new MetalContext();
        SwigluF16.Execute(ctx, gate, up, result);

        for (int i = 0; i < n; i++)
            Assert.Equal(expected[i], (float)result[i], Tol);
    }

    [MetalTestFact]
    public void EmptySpans_ReturnsSilently()
    {
        using var ctx = new MetalContext();
        SwigluF16.Execute(ctx, [], [], []);
    }

    [MetalTestFact]
    public void LengthMismatch_ThrowsArgumentException()
    {
        using var ctx = new MetalContext();
        Assert.Throws<ArgumentException>(() =>
            SwigluF16.Execute(ctx, [H(1f)[0], H(2f)[0]], [H(1f)[0]], new Half[2]));
    }
}
