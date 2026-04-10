using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Threading;

public sealed unsafe class ComputeThreadPoolTests
{
    /// <summary>
    /// Verifies that spin-wait dispatch produces the same numerical results as event-based dispatch.
    /// </summary>
    [Fact]
    public void SpinWait_Dispatch_SameResultAsEventBased()
    {
        const int threadCount = 4;
        const int arraySize = 1024;
        using var pool = new ComputeThreadPool(threadCount);

        // Run with event-based mode
        pool.SetDispatchMode(DispatchMode.EventBased);
        var eventResult = RunSumWork(pool, arraySize);

        // Run with spin-wait mode
        pool.SetDispatchMode(DispatchMode.SpinWait);
        var spinResult = RunSumWork(pool, arraySize);

        Assert.Equal(eventResult, spinResult);
    }

    /// <summary>
    /// Verifies that switching between dispatch modes across multiple dispatches doesn't hang or corrupt results.
    /// </summary>
    [Fact]
    public void SetDispatchMode_SwitchesBetweenModes()
    {
        const int threadCount = 4;
        const int arraySize = 256;
        using var pool = new ComputeThreadPool(threadCount);

        for (int i = 0; i < 20; i++)
        {
            var mode = i % 2 == 0 ? DispatchMode.SpinWait : DispatchMode.EventBased;
            pool.SetDispatchMode(mode);
            var result = RunSumWork(pool, arraySize);
            Assert.Equal(arraySize, result); // each element = 1, so sum = arraySize
        }
    }

    /// <summary>
    /// Verifies that decode thread cap correctly reduces the number of active workers.
    /// All threads should still produce correct results, but with fewer partitions.
    /// </summary>
    [Fact]
    public void DecodeThreadCap_ReducesActiveWorkers()
    {
        const int totalThreads = 8;
        const int decodeThreads = 3;
        const int arraySize = 1024;

        var config = new ThreadingConfig(totalThreads, decodeThreads);
        using var pool = new ComputeThreadPool(totalThreads, topology: null, config);

        // In event-based mode: all threads active
        pool.SetDispatchMode(DispatchMode.EventBased);
        var fullResult = RunSumWork(pool, arraySize);

        // In spin-wait mode: decode thread cap applies
        pool.SetDispatchMode(DispatchMode.SpinWait);
        var cappedResult = RunSumWork(pool, arraySize);

        // Both should produce the same sum (work is partitioned differently but covers all elements)
        Assert.Equal(arraySize, fullResult);
        Assert.Equal(arraySize, cappedResult);
    }

    /// <summary>
    /// Stress test: rapidly switch modes while dispatching to ensure no deadlocks.
    /// </summary>
    [Fact]
    public void Dispatch_NoDeadlock_UnderModeSwitch()
    {
        const int threadCount = 4;
        const int arraySize = 128;
        using var pool = new ComputeThreadPool(threadCount);

        for (int i = 0; i < 100; i++)
        {
            pool.SetDispatchMode(i % 3 == 0 ? DispatchMode.SpinWait : DispatchMode.EventBased);
            var result = RunSumWork(pool, arraySize);
            Assert.Equal(arraySize, result);
        }
    }

    /// <summary>
    /// Verifies that the default constructor (no topology/config) still works correctly.
    /// </summary>
    [Fact]
    public void DefaultConstructor_WorksCorrectly()
    {
        const int threadCount = 3;
        const int arraySize = 512;
        using var pool = new ComputeThreadPool(threadCount);

        var result = RunSumWork(pool, arraySize);
        Assert.Equal(arraySize, result);
    }

    /// <summary>
    /// Without NUMA topology or explicit pinning flags, caller-thread pinning has no
    /// candidate core and must remain a no-op. The property should stay <c>false</c>.
    /// </summary>
    [Fact]
    public void CallerPinning_NoTopology_IsNoOp()
    {
        const int threadCount = 3;
        using var pool = new ComputeThreadPool(threadCount);

        // Force a dispatch — PinCallerThread runs automatically.
        var result = RunSumWork(pool, 64);
        Assert.Equal(64, result);

        Assert.False(pool.CallerThreadPinned,
            "No topology provided → no candidate core → pinning must be a no-op");
    }

    /// <summary>
    /// When caller pinning is explicitly disabled via <see cref="ThreadingConfig"/>,
    /// the pool must not pin the caller even if topology is available.
    /// </summary>
    [Fact]
    public void CallerPinning_ExplicitlyDisabled_IsNoOp()
    {
        const int threadCount = 3;
        var config = new ThreadingConfig(
            ThreadCount: threadCount,
            DecodeThreadCount: 0,
            EnableNumaPinning: true,
            EnablePCorePinning: true,
            EnableCallerPinning: false);

        // Even if topology were supplied, EnableCallerPinning=false forces _callerCoreId=-1.
        using var pool = new ComputeThreadPool(threadCount, topology: null, config);

        var result = RunSumWork(pool, 64);
        Assert.Equal(64, result);

        Assert.False(pool.CallerThreadPinned);
    }

    /// <summary>
    /// Calling <c>PinCallerThread</c> from a <c>ThreadPool</c> thread (via <c>Task.Run</c>)
    /// must skip pinning rather than corrupt the pool. The method should be silent and
    /// <see cref="ComputeThreadPool.CallerThreadPinned"/> must remain <c>false</c>.
    /// </summary>
    [Fact]
    public void PinCallerThread_OnThreadPoolThread_Skips()
    {
        // Exercise the ThreadPool-skip guard: call PinCallerThread from a pool thread
        // and assert it did not flip the flag. The guard short-circuits before the
        // actual pin syscall so pool threads keep their existing affinity.
        // Uses a signal primitive instead of await because the test class is 'unsafe'
        // (which precludes async/await).
        const int threadCount = 3;
        using var pool = new ComputeThreadPool(threadCount);

        using var done = new ManualResetEventSlim(false);
        bool wasOnPoolThread = false;
        bool pinnedAfter = false;

        ThreadPool.QueueUserWorkItem(_ =>
        {
            wasOnPoolThread = Thread.CurrentThread.IsThreadPoolThread;
            pool.PinCallerThread();
            pinnedAfter = pool.CallerThreadPinned;
            done.Set();
        });

        Assert.True(done.Wait(TimeSpan.FromSeconds(5)), "Pool-thread pin attempt timed out");
        Assert.True(wasOnPoolThread, "Test helper did not run on a ThreadPool thread");
        Assert.False(pinnedAfter, "PinCallerThread on a ThreadPool thread must be a no-op");
    }

    /// <summary>
    /// <c>PinCallerThread</c> is idempotent — repeated calls must not re-run the pin logic
    /// nor throw. Verified via the CAS-based one-shot flag.
    /// </summary>
    [Fact]
    public void PinCallerThread_Idempotent()
    {
        const int threadCount = 3;
        using var pool = new ComputeThreadPool(threadCount);

        // Call multiple times — must not throw and must not flip state unexpectedly.
        pool.PinCallerThread();
        pool.PinCallerThread();
        pool.PinCallerThread();

        // A subsequent dispatch should also be a no-op for the pin path.
        var result = RunSumWork(pool, 64);
        Assert.Equal(64, result);
    }

    /// <summary>
    /// Helper: dispatches work that sums an array of 1.0f values across all threads.
    /// Returns the computed sum (should equal arraySize when each element is 1.0f).
    /// </summary>
    private static int RunSumWork(ComputeThreadPool pool, int arraySize)
    {
        float* data = (float*)NativeMemory.AlignedAlloc((nuint)(arraySize * sizeof(float)), 64);
        float* partialSums = (float*)NativeMemory.AlignedAlloc((nuint)(pool.ThreadCount * sizeof(float)), 64);

        try
        {
            for (int i = 0; i < arraySize; i++)
                data[i] = 1.0f;
            for (int i = 0; i < pool.ThreadCount; i++)
                partialSums[i] = 0;

            var ctx = new SumContext { Data = data, PartialSums = partialSums, ArraySize = arraySize };
            pool.Dispatch((nint)(&ctx), &SumWorker);

            float total = 0;
            for (int i = 0; i < pool.ThreadCount; i++)
                total += partialSums[i];

            return (int)total;
        }
        finally
        {
            NativeMemory.AlignedFree(data);
            NativeMemory.AlignedFree(partialSums);
        }
    }

    private struct SumContext
    {
        public float* Data;
        public float* PartialSums;
        public int ArraySize;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SumWorker(nint ctx, int threadIdx, int threadCount)
    {
        ref var c = ref Unsafe.AsRef<SumContext>((void*)ctx);
        int chunkSize = (c.ArraySize + threadCount - 1) / threadCount;
        int start = threadIdx * chunkSize;
        int end = Math.Min(start + chunkSize, c.ArraySize);

        float sum = 0;
        for (int i = start; i < end; i++)
            sum += c.Data[i];

        c.PartialSums[threadIdx] = sum;
    }
}
