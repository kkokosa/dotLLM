using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Threading;

public sealed class NumaTopologyTests
{
    [Fact]
    public void Detect_ReturnsNonNull()
    {
        var topology = NumaTopology.Detect();
        Assert.NotNull(topology);
    }

    [Fact]
    public void Detect_HasAtLeastOneNode()
    {
        var topology = NumaTopology.Detect();
        Assert.True(topology.NumaNodeCount >= 1);
    }

    [Fact]
    public void Detect_ProcessorCount_Positive()
    {
        var topology = NumaTopology.Detect();
        Assert.True(topology.Processors.Count > 0);
    }

    [Fact]
    public void PerformanceCoreIds_NonEmpty()
    {
        var topology = NumaTopology.Detect();
        Assert.NotEmpty(topology.PerformanceCoreIds);
    }

    [Fact]
    public void MemoryChannelEstimate_Positive()
    {
        var topology = NumaTopology.Detect();
        Assert.True(topology.MemoryChannelEstimate > 0);
    }

    [Fact]
    public void ProcessorsByNumaNode_ContainsAllProcessors()
    {
        var topology = NumaTopology.Detect();
        int total = topology.ProcessorsByNumaNode.Values.Sum(list => list.Count);
        Assert.Equal(topology.Processors.Count, total);
    }

    [Fact]
    public void ParseCpuList_SingleValues()
    {
        var result = NumaTopology.ParseCpuList("0,3,5").ToList();
        Assert.Equal(new[] { 0, 3, 5 }, result);
    }

    [Fact]
    public void ParseCpuList_Ranges()
    {
        var result = NumaTopology.ParseCpuList("0-3").ToList();
        Assert.Equal(new[] { 0, 1, 2, 3 }, result);
    }

    [Fact]
    public void ParseCpuList_Mixed()
    {
        var result = NumaTopology.ParseCpuList("0-2,5,7-9").ToList();
        Assert.Equal(new[] { 0, 1, 2, 5, 7, 8, 9 }, result);
    }

    [Fact]
    public void ParseCpuList_Empty()
    {
        var result = NumaTopology.ParseCpuList("").ToList();
        Assert.Empty(result);
    }
}
