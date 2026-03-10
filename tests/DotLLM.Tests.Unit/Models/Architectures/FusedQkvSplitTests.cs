using DotLLM.Cpu.Kernels;
using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Tests for fused QKV tensor split logic -- verifies that byte offset calculations
/// correctly partition a fused [Q|K|V] weight tensor into separate Q, K, V components.
/// </summary>
public class FusedQkvSplitTests
{
    /// <summary>
    /// Verifies RowByteSize computation for common quantization types (used in fused QKV pointer arithmetic).
    /// </summary>
    [Theory]
    [InlineData(512, QuantizationType.F32, 2048)]
    [InlineData(256, QuantizationType.F16, 512)]
    [InlineData(256, QuantizationType.Q8_0, 256L / 32 * 34)]
    public void RowByteSize_ComputesCorrectly(long elementCount, QuantizationType qt, long expected)
    {
        long actual = Dequantize.RowByteSize(elementCount, qt);
        Assert.Equal(expected, actual);
    }

    /// <summary>
    /// Verifies that fused QKV offset arithmetic produces correct non-overlapping regions.
    /// For a model with 8 heads, 2 KV heads, headDim=64:
    ///   Q dim = 8*64 = 512, K dim = 2*64 = 128, V dim = 2*64 = 128
    ///   Total rows = 512 + 128 + 128 = 768
    /// </summary>
    [Fact]
    public void FusedQkvOffsets_F32_AreCorrect()
    {
        int numHeads = 8, numKvHeads = 2, headDim = 64;
        int inputDim = 512; // hidden_size
        var qt = QuantizationType.F32;
        long rowBytes = Dequantize.RowByteSize(inputDim, qt); // 512 * 4 = 2048

        int qDim = numHeads * headDim;     // 512
        int kvDim = numKvHeads * headDim;   // 128

        // Q starts at offset 0
        long qOffset = 0;
        // K starts after Q rows
        long kOffset = qDim * rowBytes;     // 512 * 2048 = 1,048,576
        // V starts after Q + K rows
        long vOffset = (qDim + kvDim) * rowBytes; // 640 * 2048 = 1,310,720

        Assert.Equal(0, qOffset);
        Assert.Equal(1_048_576, kOffset);
        Assert.Equal(1_310_720, vOffset);

        // Total = (512 + 128 + 128) * 2048 = 768 * 2048 = 1,572,864
        long totalBytes = (qDim + 2 * kvDim) * rowBytes;
        Assert.Equal(1_572_864, totalBytes);

        // No overlaps
        Assert.True(kOffset > qOffset);
        Assert.True(vOffset > kOffset);
        Assert.True(kOffset >= qDim * rowBytes);
        Assert.True(vOffset >= (qDim + kvDim) * rowBytes);
    }

    /// <summary>
    /// Verifies fused QKV bias split: Q bias [qDim], K bias [kvDim], V bias [kvDim]
    /// are contiguous in the fused bias array.
    /// </summary>
    [Fact]
    public void FusedQkvBiasOffsets_AreCorrect()
    {
        int numHeads = 8, numKvHeads = 2, headDim = 64;
        int qDim = numHeads * headDim;     // 512
        int kvDim = numKvHeads * headDim;   // 128

        // Bias is F32: element offsets (in bytes) for sizeof(float)
        long qBiasOffset = 0;
        long kBiasOffset = qDim * sizeof(float);          // 512 * 4 = 2048
        long vBiasOffset = (qDim + kvDim) * sizeof(float); // 640 * 4 = 2560

        Assert.Equal(0, qBiasOffset);
        Assert.Equal(2048, kBiasOffset);
        Assert.Equal(2560, vBiasOffset);

        long totalBiasBytes = (qDim + 2 * kvDim) * sizeof(float); // 768 * 4 = 3072
        Assert.Equal(3072, totalBiasBytes);
    }

    /// <summary>
    /// Verifies fused QKV with Q8_0 quantized weights: row byte size uses block layout.
    /// </summary>
    [Fact]
    public void FusedQkvOffsets_Q8_0_AreCorrect()
    {
        int numHeads = 4, numKvHeads = 2, headDim = 32;
        int inputDim = 128; // hidden_size, must be divisible by 32 for Q8_0
        var qt = QuantizationType.Q8_0;
        long rowBytes = Dequantize.RowByteSize(inputDim, qt); // 128/32 * 34 = 136

        int qDim = numHeads * headDim;     // 128
        int kvDim = numKvHeads * headDim;   // 64

        long kOffset = qDim * rowBytes;             // 128 * 136 = 17,408
        long vOffset = (qDim + kvDim) * rowBytes;   // 192 * 136 = 26,112

        Assert.Equal(17_408, kOffset);
        Assert.Equal(26_112, vOffset);
    }
}
