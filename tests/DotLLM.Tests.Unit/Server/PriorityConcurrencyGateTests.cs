using System;
using System.Threading;
using System.Threading.Tasks;
using DotLLM.Server.RateLimiting;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Tests for the per-key priority-aware concurrency gate. The gate is
/// internal — visible to this test project via the
/// <c>InternalsVisibleTo("DotLLM.Tests.Unit")</c> attribute on
/// <c>DotLLM.Server</c>.
/// </summary>
public class PriorityConcurrencyGateTests
{
    [Fact]
    public async Task Acquire_BelowLimit_GrantsImmediately()
    {
        using var gate = new PriorityConcurrencyGate(maxConcurrent: 2);
        var lease1 = await gate.AcquireAsync(RequestPriority.Normal, TimeSpan.FromSeconds(1), default);
        var lease2 = await gate.AcquireAsync(RequestPriority.Normal, TimeSpan.FromSeconds(1), default);

        Assert.NotNull(lease1);
        Assert.NotNull(lease2);
        Assert.Equal(2, gate.ActiveCount);

        lease1!.Dispose();
        Assert.Equal(1, gate.ActiveCount);
        lease2!.Dispose();
        Assert.Equal(0, gate.ActiveCount);
    }

    [Fact]
    public async Task Acquire_AtLimit_QueuesWaiterUntilSlotFrees()
    {
        using var gate = new PriorityConcurrencyGate(maxConcurrent: 1);
        var first = (await gate.AcquireAsync(RequestPriority.Normal, TimeSpan.FromSeconds(1), default))!;

        var pending = gate.AcquireAsync(RequestPriority.Normal, TimeSpan.FromSeconds(5), default).AsTask();
        // Pending — slot is occupied.
        Assert.False(pending.IsCompleted);
        Assert.Equal(1, gate.QueueLength);

        first.Dispose();
        var second = await pending;
        Assert.NotNull(second);
        Assert.Equal(0, gate.QueueLength);
        second!.Dispose();
    }

    [Fact]
    public async Task Acquire_HighPriorityWaiter_JumpsAheadOfLowerPriority()
    {
        using var gate = new PriorityConcurrencyGate(maxConcurrent: 1);
        var holder = (await gate.AcquireAsync(RequestPriority.Normal, TimeSpan.FromSeconds(1), default))!;

        // Enqueue Low first, then High. When the holder releases, High must
        // be served first — even though it arrived later.
        var lowTask = gate.AcquireAsync(RequestPriority.Low, TimeSpan.FromSeconds(5), default).AsTask();
        await WaitForQueueLength(gate, expected: 1);

        var highTask = gate.AcquireAsync(RequestPriority.High, TimeSpan.FromSeconds(5), default).AsTask();
        await WaitForQueueLength(gate, expected: 2);

        holder.Dispose();

        // High should resolve first; Low remains pending.
        var highLease = await highTask;
        Assert.NotNull(highLease);
        Assert.False(lowTask.IsCompleted);

        // Release High → Low gets served.
        highLease!.Dispose();
        var lowLease = await lowTask;
        Assert.NotNull(lowLease);
        lowLease!.Dispose();
    }

    [Fact]
    public async Task Acquire_SamePriority_PreservesFifoOrder()
    {
        using var gate = new PriorityConcurrencyGate(maxConcurrent: 1);
        var holder = (await gate.AcquireAsync(RequestPriority.Normal, TimeSpan.FromSeconds(1), default))!;

        var first = gate.AcquireAsync(RequestPriority.Normal, TimeSpan.FromSeconds(5), default).AsTask();
        await WaitForQueueLength(gate, expected: 1);

        var second = gate.AcquireAsync(RequestPriority.Normal, TimeSpan.FromSeconds(5), default).AsTask();
        await WaitForQueueLength(gate, expected: 2);

        holder.Dispose();
        // First-arrived should resolve first when priorities tie.
        var firstLease = await first;
        Assert.NotNull(firstLease);
        Assert.False(second.IsCompleted);
        firstLease!.Dispose();

        var secondLease = await second;
        Assert.NotNull(secondLease);
        secondLease!.Dispose();
    }

    [Fact]
    public async Task Acquire_QueueTimeout_ReturnsNullLeaseWithoutConsumingSlot()
    {
        using var gate = new PriorityConcurrencyGate(maxConcurrent: 1);
        var holder = (await gate.AcquireAsync(RequestPriority.Normal, TimeSpan.FromSeconds(1), default))!;

        var timedOut = await gate.AcquireAsync(RequestPriority.Normal, TimeSpan.FromMilliseconds(100), default);
        Assert.Null(timedOut);
        Assert.Equal(1, gate.ActiveCount);
        // Note: cancelled waiters remain in the priority heap until the next
        // Release() sweeps them. This is a deliberate trade-off — peeking the
        // heap under the lock to scrub cancelled waiters would be O(n) per
        // timeout. ActiveCount is the load-bearing invariant.

        // Holder still owns the only slot — releasing it should restore the gate.
        holder.Dispose();
        Assert.Equal(0, gate.ActiveCount);
        Assert.Equal(0, gate.QueueLength);
    }

    [Fact]
    public async Task Acquire_ExternalCancellation_ThrowsOperationCanceled()
    {
        using var gate = new PriorityConcurrencyGate(maxConcurrent: 1);
        var holder = (await gate.AcquireAsync(RequestPriority.Normal, TimeSpan.FromSeconds(1), default))!;

        using var cts = new CancellationTokenSource();
        var pending = gate.AcquireAsync(RequestPriority.Normal, TimeSpan.FromSeconds(5), cts.Token).AsTask();
        await WaitForQueueLength(gate, expected: 1);

        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pending);
        holder.Dispose();
    }

    [Fact]
    public async Task Acquire_AfterDispose_DoesNotLeakSlots()
    {
        var gate = new PriorityConcurrencyGate(maxConcurrent: 1);
        var holder = (await gate.AcquireAsync(RequestPriority.Normal, TimeSpan.FromSeconds(1), default))!;

        // Park a waiter so Dispose has something to fault.
        var pending = gate.AcquireAsync(RequestPriority.Normal, TimeSpan.FromSeconds(5), default).AsTask();
        await WaitForQueueLength(gate, expected: 1);

        gate.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => pending);

        // Dropping the holder after dispose must not throw.
        holder.Dispose();
    }

    [Fact]
    public void Constructor_RejectsNonPositiveLimit()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PriorityConcurrencyGate(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PriorityConcurrencyGate(-1));
    }

    /// <summary>
    /// Spin until the gate observes the expected queue length, with a short
    /// timeout — keeps tests deterministic without sleeping on every call.
    /// </summary>
    private static async Task WaitForQueueLength(PriorityConcurrencyGate gate, int expected)
    {
        var deadline = DateTime.UtcNow.AddSeconds(2);
        while (gate.QueueLength != expected)
        {
            if (DateTime.UtcNow >= deadline)
                throw new TimeoutException($"Expected QueueLength={expected}, observed {gate.QueueLength}");
            await Task.Yield();
        }
    }
}
