using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.KvCache;

public sealed class SimpleKvCacheValidationTests
{
    [Theory]
    [InlineData(0, 2, 32, 16, "numLayers")]
    [InlineData(-1, 2, 32, 16, "numLayers")]
    [InlineData(1, 0, 32, 16, "numKvHeads")]
    [InlineData(1, -1, 32, 16, "numKvHeads")]
    [InlineData(1, 2, 0, 16, "headDim")]
    [InlineData(1, 2, -1, 16, "headDim")]
    [InlineData(1, 2, 32, 0, "maxSeqLen")]
    [InlineData(1, 2, 32, -1, "maxSeqLen")]
    public void Constructor_RejectsNonPositiveDimensions(
        int layers, int heads, int dimension, int length, string parameter)
    {
        var error = Assert.Throws<ArgumentOutOfRangeException>(
            () => new SimpleKvCache(layers, heads, dimension, length));
        Assert.Equal(parameter, error.ParamName);
    }

    [Fact]
    public void Constructor_RejectsOverflowingStrideBeforeAllocation()
    {
        Assert.Throws<OverflowException>(() => new SimpleKvCache(1, int.MaxValue, 2, 1));
    }

    [Fact]
    public void Constructor_RejectsOverflowingLayerBytesBeforeAllocation()
    {
        Assert.Throws<OverflowException>(
            () => new SimpleKvCache(1, 1, int.MaxValue, int.MaxValue));
    }
}
