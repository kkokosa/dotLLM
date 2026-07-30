using System;
using System.Collections.Generic;
using DotLLM.Core.Diagnostics;
using Xunit;

namespace DotLLM.Tests.Unit.Diagnostics;

public sealed class HookRegistryTests
{
    [Fact]
    public void HasHookAt_NoRegistration_ReturnsFalseForEveryPoint()
    {
        var registry = new HookRegistry();
        foreach (HookPoint point in Enum.GetValues<HookPoint>())
        {
            Assert.False(registry.HasHookAt(point));
            Assert.False(registry.HasHookAt(point, layer: 5));
            Assert.Equal(0, registry.CountAt(point));
        }
    }

    [Fact]
    public void Register_SingleHook_HasHookAtReturnsTrue()
    {
        var registry = new HookRegistry();
        var hook = new CountingHook(HookPoint.PostLayer);

        registry.Register(hook);

        Assert.True(registry.HasHookAt(HookPoint.PostLayer));
        Assert.Equal(1, registry.CountAt(HookPoint.PostLayer));
        Assert.False(registry.HasHookAt(HookPoint.PostEmbedding));
    }

    [Fact]
    public void Register_Null_Throws()
    {
        var registry = new HookRegistry();
        Assert.Throws<ArgumentNullException>(() => registry.Register(null!));
    }

    [Fact]
    public void Unregister_ThenFire_IsNoOp()
    {
        var registry = new HookRegistry();
        var hook = new CountingHook(HookPoint.PostLayer);
        registry.Register(hook);

        bool removed = registry.Unregister(hook);

        Assert.True(removed);
        Assert.False(registry.HasHookAt(HookPoint.PostLayer));
        Assert.Equal(0, registry.CountAt(HookPoint.PostLayer));

        // Fire should be a no-op after unregister.
        Span<float> buf = stackalloc float[4];
        registry.Fire(HookPoint.PostLayer, buf, new HookContext(0, 0, 0, 0));
        Assert.Equal(0, hook.CallCount);
    }

    [Fact]
    public void Unregister_NotRegistered_ReturnsFalse()
    {
        var registry = new HookRegistry();
        var hook = new CountingHook(HookPoint.PostLayer);
        Assert.False(registry.Unregister(hook));
    }

    [Fact]
    public void Unregister_OneOfMany_KeepsHasHookTrue()
    {
        var registry = new HookRegistry();
        var a = new CountingHook(HookPoint.PostLayer);
        var b = new CountingHook(HookPoint.PostLayer);
        registry.Register(a);
        registry.Register(b);

        Assert.True(registry.Unregister(a));
        Assert.True(registry.HasHookAt(HookPoint.PostLayer));
        Assert.Equal(1, registry.CountAt(HookPoint.PostLayer));
    }

    [Fact]
    public void Fire_NoHookRegistered_DoesNothing()
    {
        var registry = new HookRegistry();
        Span<float> buf = stackalloc float[] { 1, 2, 3, 4 };
        registry.Fire(HookPoint.PostEmbedding, buf, new HookContext(-1, 0, 0, 0));
        Assert.Equal(1f, buf[0]);
        Assert.Equal(4f, buf[3]);
    }

    [Fact]
    public void Fire_ContinueHook_LeavesActivationUnchanged()
    {
        var registry = new HookRegistry();
        var hook = new CountingHook(HookPoint.PostLayer);
        registry.Register(hook);

        Span<float> buf = stackalloc float[] { 1, 2, 3, 4 };
        registry.Fire(HookPoint.PostLayer, buf, new HookContext(0, 0, 0, 0));

        Assert.Equal(1, hook.CallCount);
        Assert.Equal(new float[] { 1, 2, 3, 4 }, buf.ToArray());
    }

    [Fact]
    public void Fire_ReplaceHook_OverwritesActivation()
    {
        var registry = new HookRegistry();
        var hook = new ReplaceHook(HookPoint.PostLayer, new float[] { 10, 20, 30, 40 });
        registry.Register(hook);

        Span<float> buf = stackalloc float[] { 1, 2, 3, 4 };
        registry.Fire(HookPoint.PostLayer, buf, new HookContext(0, 0, 0, 0));

        Assert.Equal(new float[] { 10, 20, 30, 40 }, buf.ToArray());
    }

    [Fact]
    public void Fire_ReplaceLengthMismatch_Throws()
    {
        var registry = new HookRegistry();
        var hook = new ReplaceHook(HookPoint.PostLayer, new float[] { 10, 20 });
        registry.Register(hook);

        Assert.Throws<InvalidOperationException>(() =>
        {
            var buf = new float[4];
            registry.Fire(HookPoint.PostLayer, buf.AsSpan(), new HookContext(0, 0, 0, 0));
        });
    }

    [Fact]
    public void Fire_MultipleHooks_FireInRegistrationOrder()
    {
        var registry = new HookRegistry();
        var order = new List<string>();
        registry.Register(new OrderHook("a", HookPoint.PostLayer, order));
        registry.Register(new OrderHook("b", HookPoint.PostLayer, order));
        registry.Register(new OrderHook("c", HookPoint.PostLayer, order));

        Span<float> buf = stackalloc float[2];
        registry.Fire(HookPoint.PostLayer, buf, new HookContext(0, 0, 0, 0));

        Assert.Equal(new[] { "a", "b", "c" }, order);
    }

    [Fact]
    public void Fire_ReplaceThenInspect_ThreadsReplacementDownstream()
    {
        var registry = new HookRegistry();
        var first = new ReplaceHook(HookPoint.PostLayer, new float[] { 7, 8 });
        var second = new CapturingHook(HookPoint.PostLayer);
        registry.Register(first);
        registry.Register(second);

        Span<float> buf = stackalloc float[] { 1, 2 };
        registry.Fire(HookPoint.PostLayer, buf, new HookContext(0, 0, 0, 0));

        Assert.Equal(new float[] { 7, 8 }, second.LastObserved);
        Assert.Equal(new float[] { 7, 8 }, buf.ToArray());
    }

    [Fact]
    public void HotPathGuard_NoHooksRegistered_ZeroAllocations()
    {
        // Direct verification of the "zero-cost when off" hard constraint.
        var registry = new HookRegistry();
        var hookSlot = (HookRegistry?)registry;
        var ctx = new HookContext(0, 0, 0, 0);
        var buf = new float[64];

        // Warmup to ensure JIT + any first-time allocations are settled.
        for (int i = 0; i < 100; i++)
            GuardedFire(hookSlot, buf, in ctx);

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10_000; i++)
            GuardedFire(hookSlot, buf, in ctx);
        long after = GC.GetAllocatedBytesForCurrentThread();

        // Strict == 0 is intentional, not flaky: "zero-cost when disabled" is a hard project
        // mandate, the warm-up loop above settles JIT/tiered compilation, the measured region has
        // no boxing/closure/enumerator allocation, and GetAllocatedBytesForCurrentThread is
        // per-thread so concurrent test allocations cannot leak in. A threshold would silently
        // permit a regression that introduces hot-path allocation.
        Assert.Equal(0L, after - before);
    }

    [Fact]
    public void HotPathGuard_NullRegistry_ZeroAllocations()
    {
        HookRegistry? registry = null;
        var ctx = new HookContext(0, 0, 0, 0);
        var buf = new float[64];

        for (int i = 0; i < 100; i++)
            GuardedFire(registry, buf, in ctx);

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10_000; i++)
            GuardedFire(registry, buf, in ctx);
        long after = GC.GetAllocatedBytesForCurrentThread();

        Assert.Equal(0L, after - before);
    }

    [Fact]
    public void HotPathGuard_NoHookForPoint_HookOnAnotherPoint_DoesNotFire()
    {
        // Registry has hooks at PostEmbedding only — PostLayer guard must short-circuit
        // and never call the (registered) PostEmbedding hook.
        var registry = new HookRegistry();
        var counter = new CountingHook(HookPoint.PostEmbedding);
        registry.Register(counter);

        var ctx = new HookContext(0, 0, 0, 0);
        var buf = new float[8];
        for (int i = 0; i < 1_000; i++)
        {
            if (registry.HasHookAt(HookPoint.PostLayer))
                registry.Fire(HookPoint.PostLayer, buf, in ctx);
        }

        Assert.Equal(0, counter.CallCount);
    }

    private static void GuardedFire(HookRegistry? hooks, Span<float> buf, in HookContext ctx)
    {
        if (hooks is not null && hooks.HasHookAt(HookPoint.PostLayer))
            hooks.Fire(HookPoint.PostLayer, buf, in ctx);
    }

    private sealed class CountingHook : IInferenceHook
    {
        public CountingHook(HookPoint point) => HookPoint = point;
        public HookPoint HookPoint { get; }
        public int CallCount { get; private set; }
        public HookResult OnActivation(ReadOnlySpan<float> activation, HookContext context)
        {
            CallCount++;
            return HookResult.Continue;
        }
    }

    private sealed class ReplaceHook : IInferenceHook
    {
        private readonly float[] _payload;
        public ReplaceHook(HookPoint point, float[] payload)
        {
            HookPoint = point;
            _payload = payload;
        }
        public HookPoint HookPoint { get; }
        public HookResult OnActivation(ReadOnlySpan<float> activation, HookContext context)
            => HookResult.Replace(_payload);
    }

    private sealed class OrderHook : IInferenceHook
    {
        private readonly string _name;
        private readonly List<string> _order;
        public OrderHook(string name, HookPoint point, List<string> order)
        {
            _name = name;
            HookPoint = point;
            _order = order;
        }
        public HookPoint HookPoint { get; }
        public HookResult OnActivation(ReadOnlySpan<float> activation, HookContext context)
        {
            _order.Add(_name);
            return HookResult.Continue;
        }
    }

    private sealed class CapturingHook : IInferenceHook
    {
        public CapturingHook(HookPoint point) => HookPoint = point;
        public HookPoint HookPoint { get; }
        public float[]? LastObserved { get; private set; }
        public HookResult OnActivation(ReadOnlySpan<float> activation, HookContext context)
        {
            LastObserved = activation.ToArray();
            return HookResult.Continue;
        }
    }
}
