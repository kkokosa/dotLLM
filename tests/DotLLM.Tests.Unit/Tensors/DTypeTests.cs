using DotLLM.Core.Tensors;
using Xunit;

namespace DotLLM.Tests.Unit.Tensors;

public class DTypeTests
{
    [Fact]
    public void ComputeByteCount_Float32_ReturnsElementTimesSize()
    {
        long result = DType.Float32.ComputeByteCount(32);
        Assert.Equal(32 * 4, result);
    }

    [Fact]
    public void ComputeByteCount_Float16_ReturnsElementTimesSize()
    {
        long result = DType.Float16.ComputeByteCount(100);
        Assert.Equal(100 * 2, result);
    }

    [Fact]
    public void ComputeByteCount_Q4_0_ReturnsBlockCalculation()
    {
        // Q4_0: 32 elements per block, 18 bytes per block
        long result = DType.Q4_0.ComputeByteCount(32);
        Assert.Equal(18, result);

        result = DType.Q4_0.ComputeByteCount(64);
        Assert.Equal(36, result);
    }

    [Fact]
    public void ComputeByteCount_Q8_0_ReturnsBlockCalculation()
    {
        // Q8_0: 32 elements per block, 34 bytes per block
        long result = DType.Q8_0.ComputeByteCount(32);
        Assert.Equal(34, result);

        result = DType.Q8_0.ComputeByteCount(128);
        Assert.Equal(136, result);
    }

    [Fact]
    public void ComputeByteCount_Q4_K_ReturnsBlockCalculation()
    {
        // Q4_K: 256 elements per block, 144 bytes per block
        long result = DType.Q4_K.ComputeByteCount(256);
        Assert.Equal(144, result);

        result = DType.Q4_K.ComputeByteCount(512);
        Assert.Equal(288, result);
    }

    [Fact]
    public void ComputeByteCount_Q6_K_ReturnsBlockCalculation()
    {
        // Q6_K: 256 elements per block, 210 bytes per block
        long result = DType.Q6_K.ComputeByteCount(256);
        Assert.Equal(210, result);
    }

    [Fact]
    public void ComputeByteCount_QuantizedTypes_NonZero()
    {
        // All quantized types must return non-zero for valid element counts
        Assert.True(DType.Q4_0.ComputeByteCount(32) > 0);
        Assert.True(DType.Q4_1.ComputeByteCount(32) > 0);
        Assert.True(DType.Q8_0.ComputeByteCount(32) > 0);
        Assert.True(DType.Q4_K.ComputeByteCount(256) > 0);
        Assert.True(DType.Q5_K.ComputeByteCount(256) > 0);
        Assert.True(DType.Q6_K.ComputeByteCount(256) > 0);
    }
}
