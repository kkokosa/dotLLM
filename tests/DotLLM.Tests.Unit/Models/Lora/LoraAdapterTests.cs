using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Lora;

/// <summary>
/// Unit tests for the <see cref="LoraAdapter"/> core type — construction,
/// per-(layer, proj) lookup, IsCompatible shape validation, and the
/// IDisposable native-memory contract.
/// </summary>
public sealed class LoraAdapterTests
{
    private static ModelConfig BuildBaseConfig() => new()
    {
        Architecture = Architecture.Llama,
        VocabSize = 32,
        HiddenSize = 64,
        IntermediateSize = 128,
        NumLayers = 2,
        NumAttentionHeads = 4,
        NumKvHeads = 4,
        HeadDim = 16,
        MaxSequenceLength = 128,
    };

    [Fact]
    public void Constructor_RejectsInvalidRank()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LoraAdapter("test", rank: 0, alpha: 16f, targetModules: ["q_proj"]));
    }

    [Fact]
    public void Constructor_RejectsNullName()
    {
        Assert.Throws<ArgumentException>(() =>
            new LoraAdapter("", rank: 8, alpha: 16f, targetModules: ["q_proj"]));
    }

    [Fact]
    public void GetLayerWeights_ReturnsNullForUnknownEntry()
    {
        using var adapter = new LoraAdapter("a", rank: 8, alpha: 16f, targetModules: ["q_proj"]);
        Assert.Null(adapter.GetLayerWeights(0, "q_proj"));
        Assert.Null(adapter.GetLayerWeights(0, "k_proj"));
    }

    [Fact]
    public void AddLayerWeights_RoundTripsLookup()
    {
        var cfg = BuildBaseConfig();
        using var adapter = new LoraAdapter("a", rank: 8, alpha: 16f, targetModules: ["q_proj"]);

        int rank = 8;
        int qOut = cfg.NumAttentionHeads * cfg.HeadDim; // 64
        nint a = LoraAdapter.AllocAligned((long)qOut * rank);
        nint b = LoraAdapter.AllocAligned((long)rank * cfg.HiddenSize);
        adapter.AddLayerWeights(0, "q_proj",
            new LoraLayerWeights(AHandle: a, BHandle: b, InputDim: cfg.HiddenSize, OutputDim: qOut));

        var got = adapter.GetLayerWeights(0, "q_proj");
        Assert.NotNull(got);
        Assert.Equal(a, got!.Value.AHandle);
        Assert.Equal(b, got.Value.BHandle);
        Assert.Equal(cfg.HiddenSize, got.Value.InputDim);
        Assert.Equal(qOut, got.Value.OutputDim);
    }

    [Fact]
    public void AddLayerWeights_RejectsDuplicate()
    {
        using var adapter = new LoraAdapter("a", rank: 8, alpha: 16f, targetModules: ["q_proj"]);

        int rank = 8;
        int qOut = 64;
        nint a1 = LoraAdapter.AllocAligned((long)qOut * rank);
        nint b1 = LoraAdapter.AllocAligned((long)rank * 64);
        nint a2 = LoraAdapter.AllocAligned((long)qOut * rank);
        nint b2 = LoraAdapter.AllocAligned((long)rank * 64);
        try
        {
            adapter.AddLayerWeights(0, "q_proj",
                new LoraLayerWeights(a1, b1, 64, qOut));
            Assert.Throws<InvalidOperationException>(() =>
                adapter.AddLayerWeights(0, "q_proj",
                    new LoraLayerWeights(a2, b2, 64, qOut)));
        }
        finally
        {
            // a2/b2 leak for this test path — release them explicitly so the
            // process doesn't accumulate.
            unsafe
            {
                System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)a2);
                System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)b2);
            }
        }
    }

    [Fact]
    public void IsCompatible_AcceptsMatchingShapes()
    {
        var cfg = BuildBaseConfig();
        using var adapter = new LoraAdapter("a", rank: 8, alpha: 16f, targetModules: ["q_proj", "k_proj"]);

        int rank = 8;
        int qOut = cfg.NumAttentionHeads * cfg.HeadDim;
        int kvOut = cfg.NumKvHeads * cfg.HeadDim;

        adapter.AddLayerWeights(0, "q_proj",
            new LoraLayerWeights(
                LoraAdapter.AllocAligned((long)qOut * rank),
                LoraAdapter.AllocAligned((long)rank * cfg.HiddenSize),
                cfg.HiddenSize, qOut));
        adapter.AddLayerWeights(1, "k_proj",
            new LoraLayerWeights(
                LoraAdapter.AllocAligned((long)kvOut * rank),
                LoraAdapter.AllocAligned((long)rank * cfg.HiddenSize),
                cfg.HiddenSize, kvOut));

        Assert.True(adapter.IsCompatible(cfg));
    }

    [Fact]
    public void IsCompatible_RejectsLayerOutOfRange()
    {
        var cfg = BuildBaseConfig();
        using var adapter = new LoraAdapter("a", rank: 8, alpha: 16f, targetModules: ["q_proj"]);

        int rank = 8;
        int qOut = cfg.NumAttentionHeads * cfg.HeadDim;

        adapter.AddLayerWeights(99, "q_proj",
            new LoraLayerWeights(
                LoraAdapter.AllocAligned((long)qOut * rank),
                LoraAdapter.AllocAligned((long)rank * cfg.HiddenSize),
                cfg.HiddenSize, qOut));

        Assert.False(adapter.IsCompatible(cfg));
    }

    [Fact]
    public void IsCompatible_RejectsShapeMismatch()
    {
        var cfg = BuildBaseConfig();
        using var adapter = new LoraAdapter("a", rank: 8, alpha: 16f, targetModules: ["q_proj"]);

        int rank = 8;
        int qOut = cfg.NumAttentionHeads * cfg.HeadDim;

        adapter.AddLayerWeights(0, "q_proj",
            new LoraLayerWeights(
                LoraAdapter.AllocAligned((long)qOut * rank),
                LoraAdapter.AllocAligned((long)rank * 999), // wrong inputDim
                999, qOut));

        Assert.False(adapter.IsCompatible(cfg));
    }

    [Fact]
    public void Dispose_FreesNativeBuffers()
    {
        var adapter = new LoraAdapter("a", rank: 8, alpha: 16f, targetModules: ["q_proj"]);
        int rank = 8;
        adapter.AddLayerWeights(0, "q_proj",
            new LoraLayerWeights(
                LoraAdapter.AllocAligned((long)64 * rank),
                LoraAdapter.AllocAligned((long)rank * 64),
                64, 64));
        adapter.Dispose();

        // Idempotent: second dispose is a no-op.
        adapter.Dispose();
    }
}
