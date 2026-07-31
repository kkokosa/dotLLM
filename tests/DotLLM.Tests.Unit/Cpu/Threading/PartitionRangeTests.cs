using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Threading;

/// <summary>
/// Covers <see cref="ComputeThreadPool.PartitionRange"/>, which replaced the ceiling-division
/// split every kernel worker used to repeat.
/// </summary>
public sealed class PartitionRangeTests
{
    private static (int Start, int End)[] PartitionAll(int totalItems, int threadCount)
    {
        var ranges = new (int Start, int End)[threadCount];
        for (int t = 0; t < threadCount; t++)
        {
            ComputeThreadPool.PartitionRange(totalItems, t, threadCount, out int start, out int end);
            ranges[t] = (start, end);
        }
        return ranges;
    }

    [Theory]
    // The cases the old ceiling split got wrong, plus the ones it got right.
    [InlineData(32, 32)]   // exact fit
    [InlineData(33, 32)]   // N = T + 1 — the worst case: ceiling idled ~47% of the pool
    [InlineData(32, 24)]   // ceiling idled 8 of 24
    [InlineData(32, 20)]   // ceiling idled 4 of 20
    [InlineData(64, 48)]
    [InlineData(512, 32)]  // large N, as the matmul workers see
    [InlineData(511, 32)]  // just under a multiple
    [InlineData(96, 32)]   // exact multiple
    [InlineData(1, 1)]
    [InlineData(7, 3)]
    public void EveryThreadGetsWork_WhenItemsAtLeastThreads(int totalItems, int threadCount)
    {
        var ranges = PartitionAll(totalItems, threadCount);

        Assert.All(ranges, r => Assert.True(r.End > r.Start,
            $"a thread received an empty range for N={totalItems}, T={threadCount}"));

        int min = ranges.Min(r => r.End - r.Start);
        int max = ranges.Max(r => r.End - r.Start);
        Assert.True(max - min <= 1,
            $"per-thread counts differ by {max - min} for N={totalItems}, T={threadCount}");
    }

    [Theory]
    [InlineData(32, 32)]
    [InlineData(33, 32)]
    [InlineData(32, 24)]
    [InlineData(5, 32)]     // fewer items than threads
    [InlineData(0, 32)]     // no work at all
    [InlineData(512, 32)]
    [InlineData(1000, 7)]
    public void RangesTileTheItemsExactlyOnce(int totalItems, int threadCount)
    {
        var ranges = PartitionAll(totalItems, threadCount);

        // Contiguous and ascending: thread t ends exactly where thread t+1 begins. This is what
        // keeps results bit-identical — each thread still owns a disjoint, in-order output range.
        Assert.Equal(0, ranges[0].Start);
        for (int t = 1; t < threadCount; t++)
            Assert.Equal(ranges[t - 1].End, ranges[t].Start);
        Assert.Equal(totalItems, ranges[^1].End);

        Assert.Equal(totalItems, ranges.Sum(r => r.End - r.Start));
    }

    [Theory]
    [InlineData(5, 32)]
    [InlineData(1, 8)]
    [InlineData(0, 4)]
    public void FewerItemsThanThreads_GivesAtMostOneItemEach_AndNoOverrun(int totalItems, int threadCount)
    {
        var ranges = PartitionAll(totalItems, threadCount);

        // Granularity limit, not a partitioning flaw: with N < T some threads must idle. What
        // matters is that exactly N threads get exactly one item and none reads past the end.
        Assert.Equal(totalItems, ranges.Count(r => r.End - r.Start == 1));
        Assert.All(ranges, r => Assert.True(r.End - r.Start <= 1));
        Assert.All(ranges, r => Assert.True(r.End <= totalItems));
    }

    [Fact]
    public void FrontThreadsTakeTheRemainder()
    {
        // 33 items over 32 threads: thread 0 takes 2, the rest take 1. Under the old ceiling split
        // every thread claimed 2 and threads 17..31 got nothing.
        var ranges = PartitionAll(33, 32);

        Assert.Equal((0, 2), ranges[0]);
        Assert.Equal((2, 3), ranges[1]);
        Assert.Equal((32, 33), ranges[31]);
        Assert.Equal(32, ranges.Count(r => r.End > r.Start));
    }
}
