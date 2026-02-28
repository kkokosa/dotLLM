using DotLLM.Core.Tensors;
using Xunit;

namespace DotLLM.Tests.Unit.Tensors;

public class TensorShapeTests
{
    [Fact]
    public void Scalar_HasRankZero()
    {
        var shape = new TensorShape();

        Assert.Equal(0, shape.Rank);
    }

    [Fact]
    public void Scalar_ElementCountIsOne()
    {
        var shape = new TensorShape();

        Assert.Equal(1L, shape.ElementCount);
    }

    [Fact]
    public void Rank1_ReportsCorrectRankAndCount()
    {
        var shape = new TensorShape(128);

        Assert.Equal(1, shape.Rank);
        Assert.Equal(128L, shape.ElementCount);
    }

    [Fact]
    public void Rank2_ReportsCorrectRankAndCount()
    {
        var shape = new TensorShape(32, 128);

        Assert.Equal(2, shape.Rank);
        Assert.Equal(4096L, shape.ElementCount);
    }

    [Fact]
    public void Rank4_ReportsCorrectRankAndCount()
    {
        var shape = new TensorShape(2, 32, 128, 4096);

        Assert.Equal(4, shape.Rank);
        Assert.Equal(2L * 32 * 128 * 4096, shape.ElementCount);
    }

    [Fact]
    public void Indexer_ReturnsDimensionAtAxis()
    {
        var shape = new TensorShape(2, 32, 128);

        Assert.Equal(2, shape[0]);
        Assert.Equal(32, shape[1]);
        Assert.Equal(128, shape[2]);
    }

    [Fact]
    public void EqualShapes_AreEqual()
    {
        var a = new TensorShape(32, 128, 4096);
        var b = new TensorShape(32, 128, 4096);

        Assert.Equal(a, b);
        Assert.True(a == b);
    }

    [Fact]
    public void DifferentShapes_AreNotEqual()
    {
        var a = new TensorShape(32, 128);
        var b = new TensorShape(32, 256);

        Assert.NotEqual(a, b);
        Assert.True(a != b);
    }

    [Fact]
    public void DifferentRanks_AreNotEqual()
    {
        var a = new TensorShape(32, 128);
        var b = new TensorShape(32, 128, 1);

        Assert.NotEqual(a, b);
    }

    [Fact]
    public void EqualShapes_HaveSameHashCode()
    {
        var a = new TensorShape(32, 128, 4096);
        var b = new TensorShape(32, 128, 4096);

        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }

    [Fact]
    public void ToString_FormatsAsBracketedList()
    {
        var shape = new TensorShape(32, 128, 4096);

        Assert.Equal("[32, 128, 4096]", shape.ToString());
    }

    [Fact]
    public void Scalar_ToString_IsEmptyBrackets()
    {
        var shape = new TensorShape();

        Assert.Equal("[]", shape.ToString());
    }

    [Fact]
    public void Constructor_CopiesDimensions_NoAliasing()
    {
        int[] dims = [32, 128];
        var shape = new TensorShape(dims);
        dims[0] = 999;

        Assert.Equal(32, shape[0]);
    }
}
