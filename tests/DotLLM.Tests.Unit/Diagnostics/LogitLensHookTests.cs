using System.Runtime.InteropServices;
using DotLLM.Core.Diagnostics;
using DotLLM.Diagnostics;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Diagnostics;

public sealed class LogitLensHookTests
{
    private static readonly string SmolLmModelPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");

    // ---------------------------------------------------------------------
    // Pure-math unit tests — no model required.
    // ---------------------------------------------------------------------

    [Fact]
    public void Softmax_ProducesValidDistribution_SumIsOne()
    {
        var logits = new float[] { -1.0f, 0.0f, 2.0f, 1.5f, 0.5f };
        var output = new float[logits.Length];

        LogitLensMath.Softmax(logits, output);

        float sum = 0;
        for (int i = 0; i < output.Length; i++)
        {
            Assert.InRange(output[i], 0f, 1f);
            sum += output[i];
        }
        Assert.Equal(1.0, sum, precision: 5);
    }

    [Fact]
    public void Softmax_NumericallyStable_WithLargeLogits()
    {
        // Without subtract-max this would overflow to inf/NaN.
        var logits = new float[] { 1000.0f, 999.0f, 998.0f };
        var output = new float[logits.Length];

        LogitLensMath.Softmax(logits, output);

        Assert.All(output, p => Assert.False(float.IsNaN(p)));
        Assert.True(output[0] > output[1] && output[1] > output[2]);
    }

    [Fact]
    public void Entropy_UniformDistribution_EqualsLogN()
    {
        const int n = 16;
        var probs = new float[n];
        for (int i = 0; i < n; i++) probs[i] = 1.0f / n;

        float entropy = LogitLensMath.Entropy(probs);
        Assert.Equal(MathF.Log(n), entropy, precision: 4);
    }

    [Fact]
    public void Entropy_PointMass_IsZero()
    {
        var probs = new float[] { 0, 0, 1, 0 };
        Assert.Equal(0f, LogitLensMath.Entropy(probs), precision: 6);
    }

    [Fact]
    public void TopK_ReturnsHighestProbabilitiesInDescendingOrder()
    {
        var probs = new float[] { 0.1f, 0.4f, 0.05f, 0.3f, 0.15f };
        LogitLensMath.TopK(probs, 3, out var indices, out var values);

        Assert.Equal(new[] { 1, 3, 4 }, indices);
        Assert.Equal(new[] { 0.4f, 0.3f, 0.15f }, values);
    }

    [Fact]
    public void TopK_ClampsKToVocabSize()
    {
        var probs = new float[] { 0.2f, 0.8f };
        LogitLensMath.TopK(probs, k: 10, out var indices, out var values);

        Assert.Equal(2, indices.Length);
        Assert.Equal(2, values.Length);
        Assert.Equal(1, indices[0]);
    }

    // ---------------------------------------------------------------------
    // Hook plumbing tests over a stub ILogitsProjector — no model required.
    // ---------------------------------------------------------------------

    [Fact]
    public void HookPoint_IsPostLayer()
    {
        var hook = new LogitLensHook(new IdentityProjector(hidden: 4, vocab: 4));
        Assert.Equal(HookPoint.PostLayer, hook.HookPoint);
    }

    [Fact]
    public void OnActivation_FiltersOutNonAnalyzedLayers()
    {
        var projector = new IdentityProjector(hidden: 4, vocab: 4);
        var config = new LogitLensConfig { Layers = LogitLensLayerSelector.Specific(new[] { 1 }) };
        var hook = new LogitLensHook(projector, config);

        hook.OnActivation(new float[] { 1, 0, 0, 0 }, new HookContext(0, 0, 0, 0));
        hook.OnActivation(new float[] { 0, 1, 0, 0 }, new HookContext(1, 0, 0, 0));
        hook.OnActivation(new float[] { 0, 0, 1, 0 }, new HookContext(2, 0, 0, 0));

        Assert.Equal(1, hook.CaptureCount);
        Assert.Contains((1, 0), hook.CapturedKeys);
    }

    [Fact]
    public void OnActivation_EveryNthSelector_KeepsCorrectLayers()
    {
        var projector = new IdentityProjector(hidden: 4, vocab: 4);
        var config = new LogitLensConfig { Layers = LogitLensLayerSelector.EveryNth(2) };
        var hook = new LogitLensHook(projector, config);

        for (int i = 0; i < 6; i++)
            hook.OnActivation(new float[] { i, 0, 0, 0 }, new HookContext(i, 0, 0, 0));

        Assert.Equal(3, hook.CaptureCount);
        Assert.Contains((0, 0), hook.CapturedKeys);
        Assert.Contains((2, 0), hook.CapturedKeys);
        Assert.Contains((4, 0), hook.CapturedKeys);
    }

    [Fact]
    public void OnActivation_TokenPositionFilter_Restricts()
    {
        var projector = new IdentityProjector(hidden: 4, vocab: 4);
        var config = new LogitLensConfig
        {
            TokenPositions = new[] { 2 },
        };
        var hook = new LogitLensHook(projector, config);

        hook.OnActivation(new float[] { 1, 0, 0, 0 }, new HookContext(0, 0, 0, 0));
        hook.OnActivation(new float[] { 0, 1, 0, 0 }, new HookContext(0, 1, 0, 0));
        hook.OnActivation(new float[] { 0, 0, 1, 0 }, new HookContext(0, 2, 0, 0));

        Assert.Equal(1, hook.CaptureCount);
        Assert.Contains((0, 2), hook.CapturedKeys);
    }

    [Fact]
    public void GetResults_ProducesValidProbabilityDistributions()
    {
        // Identity projector treats hidden state as raw logits — every captured layer's
        // distribution must sum to ~1 and be in [0, 1].
        var projector = new IdentityProjector(hidden: 8, vocab: 8);
        var config = new LogitLensConfig
        {
            TopK = 3,
            StoreFullProbabilities = true,
        };
        var hook = new LogitLensHook(projector, config);

        hook.OnActivation(new float[] { 0.1f, 0.2f, 5.0f, 0.3f, 0.0f, -1.0f, 0.5f, 1.5f },
            new HookContext(0, 0, 0, 0));
        hook.OnActivation(new float[] { 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
            new HookContext(1, 0, 0, 0));

        var results = hook.GetResults();
        Assert.Equal(2, results.Count);
        foreach (var r in results)
        {
            Assert.NotNull(r.FullProbabilities);
            float sum = 0;
            for (int i = 0; i < r.FullProbabilities!.Length; i++)
            {
                Assert.InRange(r.FullProbabilities[i], 0f, 1f);
                sum += r.FullProbabilities[i];
            }
            Assert.Equal(1.0, sum, precision: 4);

            Assert.Equal(3, r.TopKTokens.Length);
            Assert.Equal(3, r.TopKProbabilities.Length);
            // Top-K probs descending
            Assert.True(r.TopKProbabilities[0] >= r.TopKProbabilities[1]);
            Assert.True(r.TopKProbabilities[1] >= r.TopKProbabilities[2]);
        }
    }

    [Fact]
    public void GetResults_ConfigurableTopK_HonouredAndClamped()
    {
        var projector = new IdentityProjector(hidden: 4, vocab: 4);
        var config = new LogitLensConfig { TopK = 99 };
        var hook = new LogitLensHook(projector, config);
        hook.OnActivation(new float[] { 1, 2, 3, 4 }, new HookContext(0, 0, 0, 0));

        var result = Assert.Single(hook.GetResults());
        Assert.Equal(4, result.TopKTokens.Length); // clamped to vocab size
        Assert.Equal(3, result.TopKTokens[0]);     // logit 4 → index 3 wins
    }

    [Fact]
    public void Analysis_ConvergenceLayer_FindsEarliestMatchingLayer()
    {
        // Hand-built results: layer 0 predicts token 7, layer 1 predicts token 3,
        // layer 2 predicts token 5, layer 3 predicts token 5.
        var results = new List<LogitLensResult>
        {
            BuildResult(layer: 0, position: 0, top: 7),
            BuildResult(layer: 1, position: 0, top: 3),
            BuildResult(layer: 2, position: 0, top: 5),
            BuildResult(layer: 3, position: 0, top: 5),
        };

        Assert.Equal(2, LogitLensAnalysis.ConvergenceLayer(results, targetTokenId: 5, tokenPosition: 0));
        Assert.Equal(1, LogitLensAnalysis.ConvergenceLayer(results, targetTokenId: 3, tokenPosition: 0));
        Assert.Null(LogitLensAnalysis.ConvergenceLayer(results, targetTokenId: 99, tokenPosition: 0));
    }

    [Fact]
    public void Analysis_ConfidenceAcrossLayers_ReadsFromFullProbabilitiesWhenAvailable()
    {
        var results = new List<LogitLensResult>
        {
            new()
            {
                LayerIndex = 0,
                TokenPosition = 0,
                TopKTokens = new[] { 0 },
                TopKProbabilities = new[] { 0.9f },
                Entropy = 0.1f,
                FullProbabilities = new[] { 0.9f, 0.05f, 0.05f },
            },
            new()
            {
                LayerIndex = 1,
                TokenPosition = 0,
                TopKTokens = new[] { 1 },
                TopKProbabilities = new[] { 0.6f },
                Entropy = 0.3f,
                FullProbabilities = new[] { 0.1f, 0.6f, 0.3f },
            },
        };

        var trajectory = LogitLensAnalysis.ConfidenceAcrossLayers(results, targetTokenId: 1, tokenPosition: 0);
        Assert.Equal(2, trajectory.Count);
        Assert.Equal(0.05f, trajectory[0].Probability, precision: 4);
        Assert.Equal(0.6f, trajectory[1].Probability, precision: 4);
    }

    [Fact]
    public void Analysis_RankOf_UsesTopKThenFullDistribution()
    {
        var result = new LogitLensResult
        {
            LayerIndex = 0,
            TokenPosition = 0,
            TopKTokens = new[] { 7, 3, 9 },
            TopKProbabilities = new[] { 0.4f, 0.3f, 0.2f },
            Entropy = 1.0f,
            FullProbabilities = new[] { 0.01f, 0.02f, 0.03f, 0.3f, 0.04f, 0.05f, 0.06f, 0.4f, 0.07f, 0.2f },
        };

        Assert.Equal(0, LogitLensAnalysis.RankOf(result, 7));
        Assert.Equal(1, LogitLensAnalysis.RankOf(result, 3));
        Assert.Equal(2, LogitLensAnalysis.RankOf(result, 9));
        // Token 6 has p=0.06; entries strictly greater: 0.4 (idx 7), 0.3 (idx 3),
        // 0.2 (idx 9), 0.07 (idx 8) → rank 4 (0-based, ties handled by strict >).
        Assert.Equal(4, LogitLensAnalysis.RankOf(result, 6));
    }

    // ---------------------------------------------------------------------
    // Discriminating test: final-layer lens output equals the model's logits.
    // Uses a real GGUF model — skipped when not on disk.
    // ---------------------------------------------------------------------

    [SkippableFact]
    public unsafe void FinalLayer_LogitLens_MatchesModelOutput_OnSingleTokenForward()
    {
        Skip.If(!File.Exists(SmolLmModelPath),
            "SmolLM-135M Q8_0 GGUF not found (run: dotllm run QuantFactory/SmolLM-135M-GGUF -q Q8_0)");

        using var gguf = GgufFile.Open(SmolLmModelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);

        var hook = new LogitLensHook(model, new LogitLensConfig
        {
            Layers = LogitLensLayerSelector.AllLayers,
            TopK = 5,
            StoreFullProbabilities = false,
        });
        model.Hooks = new HookRegistry();
        model.Hooks.Register(hook);

        // Single-token forward: the LM head in Forward runs at seqLen == 1, which is the
        // same kernel path ProjectToLogits takes — guarantees bit-exact match.
        int[] tokens = [1]; // any valid token id
        int[] positions = [0];

        using var logitsTensor = model.Forward(tokens, positions, deviceId: -1);

        int vocab = config.VocabSize;
        var modelLogits = new float[vocab];
        new ReadOnlySpan<float>((float*)logitsTensor.DataPointer, vocab).CopyTo(modelLogits);

        // Project the captured final-layer PostLayer hidden state via the lens path.
        int finalLayer = config.NumLayers - 1;
        Assert.Contains((finalLayer, 0), hook.CapturedKeys);

        // Re-run the projection independent of GetResults so we get raw logits, not softmax.
        var capturedHidden = Capture(hook, finalLayer, 0);
        var lensLogits = new float[vocab];
        ((ILogitsProjector)model).ProjectToLogits(capturedHidden, lensLogits);

        // Per-element bit-exact comparison: same kernel path on identical inputs must
        // produce byte-for-byte identical floats.
        for (int i = 0; i < vocab; i++)
        {
            Assert.Equal(
                BitConverter.SingleToInt32Bits(modelLogits[i]),
                BitConverter.SingleToInt32Bits(lensLogits[i]));
        }
    }

    /// <summary>Pull out the captured hidden state for a (layer, position) — uses reflection
    /// to keep the storage private without exposing it for non-test consumers.</summary>
    private static float[] Capture(LogitLensHook hook, int layer, int position)
    {
        var capturesField = typeof(LogitLensHook).GetField("_captures",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var dict = (System.Collections.IDictionary)capturesField.GetValue(hook)!;
        return (float[])dict[(layer, position)]!;
    }

    private static LogitLensResult BuildResult(int layer, int position, int top)
    {
        return new LogitLensResult
        {
            LayerIndex = layer,
            TokenPosition = position,
            TopKTokens = new[] { top },
            TopKProbabilities = new[] { 1.0f },
            Entropy = 0f,
        };
    }

    /// <summary>
    /// Test projector — treats the hidden state as raw logits (identity LM head, no norm).
    /// Lets math/plumbing tests run without loading a real model.
    /// </summary>
    private sealed class IdentityProjector : ILogitsProjector
    {
        public IdentityProjector(int hidden, int vocab)
        {
            HiddenSize = hidden;
            VocabSize = vocab;
            if (hidden != vocab)
                throw new ArgumentException("IdentityProjector requires hidden == vocab.");
        }

        public int HiddenSize { get; }
        public int VocabSize { get; }

        public void ProjectToLogits(ReadOnlySpan<float> hiddenState, Span<float> logits)
        {
            hiddenState.CopyTo(logits);
        }
    }
}
