using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using Xunit;

namespace DotLLM.Tests.Unit.PositionEncoding;

public sealed class AlibiPositionEncodingTests
{
    [Fact]
    public void CreateSlopes_PowerOfTwo_MatchesStandardSchedule()
    {
        float[] slopes = AlibiPositionEncoding.CreateSlopes(4);

        Assert.Equal([0.25f, 0.0625f, 0.015625f, 0.00390625f], slopes, FloatComparer.Instance);
    }

    [Fact]
    public void CreateSlopes_NonPowerOfTwo_AppendsEveryOtherExpandedSlope()
    {
        float[] slopes = AlibiPositionEncoding.CreateSlopes(6);

        Assert.Equal(6, slopes.Length);
        Assert.Equal(0.25f, slopes[0], 1e-7f);
        Assert.Equal(0.0625f, slopes[1], 1e-7f);
        Assert.Equal(0.015625f, slopes[2], 1e-7f);
        Assert.Equal(0.00390625f, slopes[3], 1e-7f);
        Assert.Equal(0.5f, slopes[4], 1e-7f);
        Assert.Equal(0.125f, slopes[5], 1e-7f);
    }

    [Fact]
    public void Apply_DoesNotMutateQOrK()
    {
        using var q = UnmanagedTensor.Allocate(new TensorShape(1, 4), DType.Float32);
        using var k = UnmanagedTensor.Allocate(new TensorShape(1, 4), DType.Float32);

        var encoding = new AlibiPositionEncoding();
        encoding.PrecomputeTables(8, new ModelConfig
        {
            Architecture = Architecture.Llama,
            VocabSize = 16,
            HiddenSize = 4,
            IntermediateSize = 16,
            NumLayers = 1,
            NumAttentionHeads = 2,
            NumKvHeads = 2,
            HeadDim = 2,
            MaxSequenceLength = 8,
            PositionEncodingType = PositionEncodingType.ALiBi,
            RoPEConfig = null
        });

        var (qOut, kOut) = encoding.Apply(q, k, [0]);

        Assert.Same(q, qOut);
        Assert.Same(k, kOut);
        Assert.Equal(2, encoding.Slopes.Length);
    }

    private sealed class FloatComparer : IEqualityComparer<float>
    {
        public static readonly FloatComparer Instance = new();

        public bool Equals(float x, float y) => MathF.Abs(x - y) <= 1e-7f;

        public int GetHashCode(float obj) => obj.GetHashCode();
    }
}
