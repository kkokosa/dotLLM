using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

public sealed class SsmStateCacheTests
{
    private static MambaSsmConfig TinySsm() => new(
        DConv: 4,
        DInner: 8,
        DState: 4,
        NGroup: 2,
        NHead: 2);

    [Fact]
    public void NewCache_IsZeroed()
    {
        using var cache = new SsmStateCache(TinySsm(), numSsmLayers: 3);

        Assert.Equal(3, cache.NumSsmLayers);
        for (int i = 0; i < cache.NumSsmLayers; i++)
        {
            foreach (var v in cache.GetConvState(i)) Assert.Equal(0f, v);
            foreach (var v in cache.GetSsmState(i)) Assert.Equal(0f, v);
        }
    }

    [Fact]
    public void LayerSlicesAreIndependent()
    {
        using var cache = new SsmStateCache(TinySsm(), numSsmLayers: 2);

        var convA = cache.GetConvState(0);
        var convB = cache.GetConvState(1);
        convA.Fill(1f);
        convB.Fill(2f);

        foreach (var v in cache.GetConvState(0)) Assert.Equal(1f, v);
        foreach (var v in cache.GetConvState(1)) Assert.Equal(2f, v);

        var ssmA = cache.GetSsmState(0);
        var ssmB = cache.GetSsmState(1);
        ssmA.Fill(3f);
        ssmB.Fill(4f);
        foreach (var v in cache.GetSsmState(0)) Assert.Equal(3f, v);
        foreach (var v in cache.GetSsmState(1)) Assert.Equal(4f, v);
    }

    [Fact]
    public void Reset_ZeroesAllLayers()
    {
        using var cache = new SsmStateCache(TinySsm(), numSsmLayers: 2);
        cache.GetConvState(0).Fill(99f);
        cache.GetSsmState(1).Fill(42f);

        cache.Reset();

        foreach (var v in cache.GetConvState(0)) Assert.Equal(0f, v);
        foreach (var v in cache.GetSsmState(1)) Assert.Equal(0f, v);
    }

    [Fact]
    public void OutOfRangeIndex_Throws()
    {
        using var cache = new SsmStateCache(TinySsm(), numSsmLayers: 2);
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetConvState(2));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetSsmState(-1));
    }

    [Fact]
    public void ZeroLayers_AllocatesNothingButStillWorks()
    {
        using var cache = new SsmStateCache(TinySsm(), numSsmLayers: 0);
        Assert.Equal(0, cache.NumSsmLayers);
        Assert.Equal(0L, cache.AllocatedBytes);
    }

    [Fact]
    public void ElementCounts_MatchConfig()
    {
        var ssm = TinySsm();
        using var cache = new SsmStateCache(ssm, numSsmLayers: 1);

        // conv_state = (d_conv-1) * (d_inner + 2*n_group*d_state) = 3 * (8 + 2*2*4) = 3 * 24 = 72
        Assert.Equal(72, cache.ConvStateElements);
        // ssm_state = d_inner * d_state = 8 * 4 = 32
        Assert.Equal(32, cache.SsmStateElements);
        Assert.Equal(cache.ConvStateElements, cache.GetConvState(0).Length);
        Assert.Equal(cache.SsmStateElements, cache.GetSsmState(0).Length);
    }

    [Fact]
    public void DoubleDispose_IsSafe()
    {
        var cache = new SsmStateCache(TinySsm(), numSsmLayers: 1);
        cache.Dispose();
        cache.Dispose(); // should not throw
    }
}
