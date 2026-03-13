using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed unsafe class FusedOpsTests
{
    // ──────────────────── SwiGLU Tests ────────────────────

    [Fact]
    public void SwiGLU_MatchesUnfusedSiLuMultiply()
    {
        var rng = new Random(42);
        const int n = 1024;
        float[] gate = new float[n];
        float[] up = new float[n];
        for (int i = 0; i < n; i++)
        {
            gate[i] = rng.NextSingle() * 20f - 10f;
            up[i] = rng.NextSingle() * 20f - 10f;
        }

        // Unfused: SiLu + Multiply
        float[] siluOut = new float[n];
        float[] unfusedResult = new float[n];
        SiLu.Execute(gate, siluOut);
        Multiply.Execute(siluOut, up, unfusedResult);

        // Fused
        float[] fusedResult = new float[n];
        FusedOps.SwiGLU(gate, up, fusedResult);

        for (int i = 0; i < n; i++)
            Assert.Equal(unfusedResult[i], fusedResult[i], 1e-4f);
    }

    [Theory]
    [InlineData(32)]
    [InlineData(128)]
    [InlineData(576)]
    [InlineData(1024)]
    [InlineData(1536)]
    [InlineData(4096)]
    [InlineData(8192)]
    public void SwiGLU_ScalarMatchesFused(int size)
    {
        var rng = new Random(123);
        float[] gate = new float[size];
        float[] up = new float[size];
        for (int i = 0; i < size; i++)
        {
            gate[i] = rng.NextSingle() * 20f - 10f;
            up[i] = rng.NextSingle() * 20f - 10f;
        }

        float[] scalarResult = new float[size];
        float[] fusedResult = new float[size];

        FusedOps.SwiGLUScalar(gate, up, scalarResult);
        FusedOps.SwiGLU(gate, up, fusedResult);

        for (int i = 0; i < size; i++)
            Assert.Equal(scalarResult[i], fusedResult[i], 1e-4f);
    }

    [Fact]
    public void SwiGLU_ZeroGate_ProducesZero()
    {
        float[] gate = [0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f];
        float[] up = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        float[] result = new float[8];

        FusedOps.SwiGLU(gate, up, result);

        // SwiGLU(0, u) = 0 * sigmoid(0) * u = 0 * 0.5 * u = 0
        for (int i = 0; i < 8; i++)
            Assert.Equal(0f, result[i], 1e-6f);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(9)]  // tile boundary + 1
    [InlineData(15)]
    [InlineData(33)]
    [InlineData(255)] // tile size - 1
    [InlineData(257)] // tile size + 1
    public void SwiGLU_AlignedAndUnaligned(int size)
    {
        var rng = new Random(77);
        float[] gate = new float[size];
        float[] up = new float[size];
        for (int i = 0; i < size; i++)
        {
            gate[i] = rng.NextSingle() * 10f - 5f;
            up[i] = rng.NextSingle() * 10f - 5f;
        }

        float[] scalarResult = new float[size];
        float[] fusedResult = new float[size];

        FusedOps.SwiGLUScalar(gate, up, scalarResult);
        FusedOps.SwiGLU(gate, up, fusedResult);

        for (int i = 0; i < size; i++)
            Assert.Equal(scalarResult[i], fusedResult[i], 1e-4f);
    }

    // ──────────────────── RmsNormQuantize Q8_0 Tests ────────────────────

    [Theory]
    [InlineData(576)]
    [InlineData(2048)]
    [InlineData(3072)]
    [InlineData(4096)]
    public void RmsNormQuantizeQ8_0_MatchesUnfused(int dim)
    {
        var rng = new Random(42);
        float[] input = new float[dim];
        float[] weight = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            input[i] = rng.NextSingle() * 2f - 1f;
            weight[i] = rng.NextSingle() * 2f;
        }
        float eps = 1e-5f;

        // Unfused: RmsNorm → normOut → QuantizeQ8_0
        float[] normOut = new float[dim];
        RmsNorm.Execute(input, weight, eps, normOut);

        int blockCount = dim / 32;
        int q8Bytes = blockCount * 34; // Q8_0BlockBytes
        byte[] unfusedQ8 = new byte[q8Bytes];

        fixed (float* normPtr = normOut)
        fixed (byte* unfusedPtr = unfusedQ8)
        {
            MatMul.QuantizeF32ToQ8_0(normPtr, unfusedPtr, dim);
        }

        // Fused
        byte[] fusedQ8 = new byte[q8Bytes];

        fixed (float* inPtr = input)
        fixed (byte* fusedPtr = fusedQ8)
        {
            FusedOps.RmsNormQuantizeQ8_0(inPtr, weight, eps, fusedPtr, dim);
        }

        // Compare byte-for-byte
        for (int i = 0; i < q8Bytes; i++)
            Assert.Equal(unfusedQ8[i], fusedQ8[i]);
    }

    // ──────────────────── RmsNormQuantize Q8_1 Tests ────────────────────

    [Theory]
    [InlineData(576)]
    [InlineData(2048)]
    [InlineData(3072)]
    [InlineData(4096)]
    public void RmsNormQuantizeQ8_1_MatchesUnfused(int dim)
    {
        var rng = new Random(42);
        float[] input = new float[dim];
        float[] weight = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            input[i] = rng.NextSingle() * 2f - 1f;
            weight[i] = rng.NextSingle() * 2f;
        }
        float eps = 1e-5f;

        // Unfused: RmsNorm → normOut → QuantizeQ8_1
        float[] normOut = new float[dim];
        RmsNorm.Execute(input, weight, eps, normOut);

        int blockCount = dim / 32;
        int q8_1Bytes = blockCount * MatMul.Q8_1BlockBytes;
        byte[] unfusedQ8 = new byte[q8_1Bytes];

        fixed (float* normPtr = normOut)
        fixed (byte* unfusedPtr = unfusedQ8)
        {
            MatMul.QuantizeF32ToQ8_1(normPtr, unfusedPtr, dim);
        }

        // Fused
        byte[] fusedQ8 = new byte[q8_1Bytes];

        fixed (float* inPtr = input)
        fixed (byte* fusedPtr = fusedQ8)
        {
            FusedOps.RmsNormQuantizeQ8_1(inPtr, weight, eps, fusedPtr, dim);
        }

        // Compare byte-for-byte
        for (int i = 0; i < q8_1Bytes; i++)
            Assert.Equal(unfusedQ8[i], fusedQ8[i]);
    }

    // ──────────────────── RmsNormQuantize Q8_K Tests ────────────────────

    [Theory]
    [InlineData(2048)]
    [InlineData(3072)]
    [InlineData(4096)]
    public void RmsNormQuantizeQ8_K_MatchesUnfused(int dim)
    {
        // Q8_K requires dim % 256 == 0
        Assert.Equal(0, dim % 256);

        var rng = new Random(42);
        float[] input = new float[dim];
        float[] weight = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            input[i] = rng.NextSingle() * 2f - 1f;
            weight[i] = rng.NextSingle() * 2f;
        }
        float eps = 1e-5f;

        // Unfused: RmsNorm → normOut → QuantizeQ8_K
        float[] normOut = new float[dim];
        RmsNorm.Execute(input, weight, eps, normOut);

        int blockCount = dim / 256;
        int q8kBytes = blockCount * MatMul.Q8_K_BlockBytes;
        byte[] unfusedQ8 = new byte[q8kBytes];

        fixed (float* normPtr = normOut)
        fixed (byte* unfusedPtr = unfusedQ8)
        {
            MatMul.QuantizeF32ToQ8_K(normPtr, unfusedPtr, dim);
        }

        // Fused
        byte[] fusedQ8 = new byte[q8kBytes];

        fixed (float* inPtr = input)
        fixed (byte* fusedPtr = fusedQ8)
        {
            FusedOps.RmsNormQuantizeQ8_K(inPtr, weight, eps, fusedPtr, dim);
        }

        // Compare byte-for-byte
        for (int i = 0; i < q8kBytes; i++)
            Assert.Equal(unfusedQ8[i], fusedQ8[i]);
    }

    // ──────────────────── RmsNormQuantize Dispatch Tests ────────────────────

    [Fact]
    public void RmsNormQuantize_ReturnsNullForF32()
    {
        float[] input = new float[32];
        byte[] dest = new byte[1024];
        fixed (float* inPtr = input)
        fixed (byte* destPtr = dest)
        {
            byte* result = FusedOps.RmsNormQuantize(inPtr, input, 1e-5f, destPtr, 32,
                DotLLM.Core.Configuration.QuantizationType.F32);
            Assert.True(result == null);
        }
    }

    [Fact]
    public void RmsNormQuantize_ReturnsNullForF16()
    {
        float[] input = new float[32];
        byte[] dest = new byte[1024];
        fixed (float* inPtr = input)
        fixed (byte* destPtr = dest)
        {
            byte* result = FusedOps.RmsNormQuantize(inPtr, input, 1e-5f, destPtr, 32,
                DotLLM.Core.Configuration.QuantizationType.F16);
            Assert.True(result == null);
        }
    }

    [Fact]
    public void RmsNormQuantize_DispatchesQ8_0()
    {
        const int dim = 64;
        var rng = new Random(42);
        float[] input = new float[dim];
        float[] weight = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            input[i] = rng.NextSingle() * 2f - 1f;
            weight[i] = rng.NextSingle() * 2f;
        }

        int q8Bytes = (dim / 32) * 34;
        byte[] dest = new byte[q8Bytes];
        fixed (float* inPtr = input)
        fixed (byte* destPtr = dest)
        {
            byte* result = FusedOps.RmsNormQuantize(inPtr, weight, 1e-5f, destPtr, dim,
                DotLLM.Core.Configuration.QuantizationType.Q8_0);
            Assert.True(result != null);
            Assert.True(result == destPtr);
        }
    }
}
