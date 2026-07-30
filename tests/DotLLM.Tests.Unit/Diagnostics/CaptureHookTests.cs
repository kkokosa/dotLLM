using DotLLM.Core.Diagnostics;
using DotLLM.Diagnostics;
using Xunit;

namespace DotLLM.Tests.Unit.Diagnostics;

public sealed class CaptureHookTests
{
    [Fact]
    public void Constructor_SetsHookPoint()
    {
        var hook = new CaptureHook(HookPoint.PostLayer);
        Assert.Equal(HookPoint.PostLayer, hook.HookPoint);
        Assert.Empty(hook.Captures);
    }

    [Fact]
    public void OnActivation_NoFilters_CapturesEveryCall()
    {
        var hook = new CaptureHook(HookPoint.PostLayer);
        hook.OnActivation(new float[] { 1, 2, 3 }, new HookContext(0, 0, 0, 0));
        hook.OnActivation(new float[] { 4, 5, 6 }, new HookContext(0, 1, 0, 0));
        hook.OnActivation(new float[] { 7, 8, 9 }, new HookContext(1, 0, 0, 0));

        Assert.Equal(3, hook.Captures.Count);
        Assert.Equal(new float[] { 1, 2, 3 }, hook.Captures[new CaptureHook.CaptureKey(0, 0)]);
        Assert.Equal(new float[] { 4, 5, 6 }, hook.Captures[new CaptureHook.CaptureKey(0, 1)]);
        Assert.Equal(new float[] { 7, 8, 9 }, hook.Captures[new CaptureHook.CaptureKey(1, 0)]);
    }

    [Fact]
    public void OnActivation_LayerFilter_SkipsUnselectedLayers()
    {
        var hook = new CaptureHook(HookPoint.PostLayer, layers: new[] { 0, 2 });
        hook.OnActivation(new float[] { 1 }, new HookContext(0, 0, 0, 0));
        hook.OnActivation(new float[] { 2 }, new HookContext(1, 0, 0, 0)); // filtered
        hook.OnActivation(new float[] { 3 }, new HookContext(2, 0, 0, 0));

        Assert.Equal(2, hook.Captures.Count);
        Assert.True(hook.Captures.ContainsKey(new CaptureHook.CaptureKey(0, 0)));
        Assert.True(hook.Captures.ContainsKey(new CaptureHook.CaptureKey(2, 0)));
        Assert.False(hook.Captures.ContainsKey(new CaptureHook.CaptureKey(1, 0)));
    }

    [Fact]
    public void OnActivation_PositionFilter_SkipsUnselectedPositions()
    {
        var hook = new CaptureHook(HookPoint.PostLayer, tokenPositions: new[] { 0, 2 });
        hook.OnActivation(new float[] { 1 }, new HookContext(0, 0, 0, 0));
        hook.OnActivation(new float[] { 2 }, new HookContext(0, 1, 0, 0)); // filtered
        hook.OnActivation(new float[] { 3 }, new HookContext(0, 2, 0, 0));

        Assert.Equal(2, hook.Captures.Count);
        Assert.True(hook.Captures.ContainsKey(new CaptureHook.CaptureKey(0, 0)));
        Assert.True(hook.Captures.ContainsKey(new CaptureHook.CaptureKey(0, 2)));
    }

    [Fact]
    public void OnActivation_LayerAndPositionFilter_Intersect()
    {
        var hook = new CaptureHook(HookPoint.PostLayer,
            layers: new[] { 0, 1 },
            tokenPositions: new[] { 0 });

        hook.OnActivation(new float[] { 1 }, new HookContext(0, 0, 0, 0)); // kept
        hook.OnActivation(new float[] { 2 }, new HookContext(0, 1, 0, 0)); // filtered (position)
        hook.OnActivation(new float[] { 3 }, new HookContext(2, 0, 0, 0)); // filtered (layer)

        Assert.Single(hook.Captures);
        Assert.True(hook.Captures.ContainsKey(new CaptureHook.CaptureKey(0, 0)));
    }

    [Fact]
    public void OnActivation_AlwaysReturnsContinue()
    {
        var hook = new CaptureHook(HookPoint.PostLayer);
        var result = hook.OnActivation(new float[] { 1, 2 }, new HookContext(0, 0, 0, 0));
        Assert.IsType<HookResult.ContinueResult>(result);
    }

    [Fact]
    public void OnActivation_CapturesIndependentCopy_NotAliasingSource()
    {
        // After capture, mutating the source span must NOT affect the stored snapshot.
        var hook = new CaptureHook(HookPoint.PostLayer);
        var src = new float[] { 1, 2, 3 };
        hook.OnActivation(src, new HookContext(0, 0, 0, 0));

        src[0] = 99;
        src[1] = 99;
        src[2] = 99;

        Assert.Equal(new float[] { 1, 2, 3 }, hook.Captures[new CaptureHook.CaptureKey(0, 0)]);
    }

    [Fact]
    public void Clear_EmptiesCaptures()
    {
        var hook = new CaptureHook(HookPoint.PostLayer);
        hook.OnActivation(new float[] { 1 }, new HookContext(0, 0, 0, 0));
        Assert.NotEmpty(hook.Captures);

        hook.Clear();
        Assert.Empty(hook.Captures);
    }

    [Fact]
    public void OnActivation_NonLayerPoint_KeyedByMinusOneLayer()
    {
        var hook = new CaptureHook(HookPoint.PostEmbedding);
        hook.OnActivation(new float[] { 1, 2 }, new HookContext(-1, 0, 0, 0));

        Assert.True(hook.Captures.ContainsKey(new CaptureHook.CaptureKey(-1, 0)));
    }

    [Fact]
    public void OnActivation_RegisteredInRegistry_RoundTripCapturesActivation()
    {
        var registry = new HookRegistry();
        var hook = new CaptureHook(HookPoint.PostLayer, layers: new[] { 0 });
        registry.Register(hook);

        Span<float> buf = stackalloc float[] { 1.5f, 2.5f, 3.5f };
        registry.Fire(HookPoint.PostLayer, buf, new HookContext(0, 7, 42, 3));

        Assert.True(hook.Captures.TryGetValue(new CaptureHook.CaptureKey(0, 7), out var captured));
        Assert.Equal(new float[] { 1.5f, 2.5f, 3.5f }, captured);
    }
}
