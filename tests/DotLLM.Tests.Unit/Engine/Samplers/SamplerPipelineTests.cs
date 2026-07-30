using System.Numerics.Tensors;
using System.Reflection;
using DotLLM.Core.Configuration;
using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class SamplerPipelineTests
{
    [Fact]
    public void Sample_Greedy_ReturnsArgMax()
    {
        var options = new InferenceOptions { Temperature = 0f };
        var pipeline = new SamplerPipeline(options);

        float[] logits = [1.0f, 5.0f, 3.0f, 2.0f];

        int result = pipeline.Sample(logits, []);

        Assert.Equal(1, result); // index of max value (5.0)
    }

    [Fact]
    public void Sample_SeededDeterminism()
    {
        var options = new InferenceOptions { Temperature = 1.0f, Seed = 42 };
        var pipeline1 = new SamplerPipeline(options);
        var pipeline2 = new SamplerPipeline(options);

        float[] logits1 = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f];
        float[] logits2 = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f];

        int result1 = pipeline1.Sample(logits1, []);
        int result2 = pipeline2.Sample(logits2, []);

        Assert.Equal(result1, result2);
    }

    [Fact]
    public void Sample_DefaultOptions_ProducesValidIndex()
    {
        var options = new InferenceOptions { Seed = 42 };
        var pipeline = new SamplerPipeline(options);

        float[] logits = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f];

        int result = pipeline.Sample(logits, []);

        Assert.InRange(result, 0, logits.Length - 1);
    }

    [Fact]
    public void Sample_WithRepetitionPenalty_ReducesRepeats()
    {
        var options = new InferenceOptions { Temperature = 0f, RepetitionPenalty = 100.0f };
        var pipeline = new SamplerPipeline(options);

        // Token 2 has the highest logit but was already generated
        float[] logits = [1.0f, 1.0f, 5.0f, 4.9f];
        var previousTokens = new List<int> { 2 };

        int result = pipeline.Sample(logits, previousTokens);

        // With extreme penalty, token 2's logit (5.0/100) = 0.05 < 4.9
        Assert.Equal(3, result);
    }

    [Fact]
    public void Sample_GreedyMultipleCallsAreDeterministic()
    {
        var options = new InferenceOptions { Temperature = 0f };
        var pipeline = new SamplerPipeline(options);

        for (int i = 0; i < 10; i++)
        {
            float[] logits = [1.0f, 3.0f, 2.0f];
            int result = pipeline.Sample(logits, []);
            Assert.Equal(1, result);
        }
    }

    [Fact]
    public void Sample_GreedyWithLogitBias_AppliesBiasBeforeArgMax()
    {
        var options = new InferenceOptions
        {
            Temperature = 0f,
            LogitBias = new Dictionary<int, float> { [1] = 2.0f },
        };
        var pipeline = new SamplerPipeline(options);

        float[] logits = [2.0f, 1.0f];

        int result = pipeline.Sample(logits, []);

        Assert.Equal(1, result);
    }

    [Fact]
    public void Sample_ComposableViaOptions_PrependsLogitBiasStep()
    {
        var options = new InferenceOptions
        {
            Temperature = 0f,
            LogitBias = new Dictionary<int, float> { [2] = 3.0f },
            SamplerSteps = [new TemperatureSampler(0.8f)],
        };
        var pipeline = new SamplerPipeline(options);

        float[] logits = [1.0f, 2.0f, 0.0f];

        int result = pipeline.Sample(logits, []);

        Assert.Equal(2, result);
    }

    [Fact]
    public void Sample_LargeNegativeLogitBias_EffectivelyPreventsSelection()
    {
        var options = new InferenceOptions
        {
            Temperature = 0f,
            LogitBias = new Dictionary<int, float> { [1] = -100.0f },
        };
        var pipeline = new SamplerPipeline(options);

        float[] logits = [0.0f, 50.0f];

        int result = pipeline.Sample(logits, []);

        Assert.Equal(0, result);
    }

    [Fact]
    public void Constructor_InstantiatesLogitBiasStep_OnlyWhenBiasMapNonEmpty()
    {
        var absent = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        var empty = new SamplerPipeline(new InferenceOptions
        {
            Temperature = 0f,
            LogitBias = new Dictionary<int, float>(),
        });
        var nonEmpty = new SamplerPipeline(new InferenceOptions
        {
            Temperature = 0f,
            LogitBias = new Dictionary<int, float> { [0] = 1.0f },
        });

        Assert.DoesNotContain(GetSamplerSteps(absent), s => s is LogitBiasStep);
        Assert.DoesNotContain(GetSamplerSteps(empty), s => s is LogitBiasStep);
        Assert.Contains(GetSamplerSteps(nonEmpty), s => s is LogitBiasStep);
    }

    [Fact]
    public void Sample_NoLogitBiasVsEmptyMap_HasSameBehavior()
    {
        var noBias = new SamplerPipeline(new InferenceOptions
        {
            Temperature = 0f,
        });
        var emptyBias = new SamplerPipeline(new InferenceOptions
        {
            Temperature = 0f,
            LogitBias = new Dictionary<int, float>(),
        });

        float[] logits1 = [1.0f, 2.0f, 1.5f];
        float[] logits2 = [1.0f, 2.0f, 1.5f];

        int result1 = noBias.Sample(logits1, []);
        int result2 = emptyBias.Sample(logits2, []);

        Assert.Equal(result1, result2);
        Assert.Equal(logits1, logits2);
    }

    [Fact]
    public void Sample_ComposableConstructor_ProducesValidIndex()
    {
        var pipeline = new SamplerPipeline(
            new TemperatureSampler(0.8f),
            new TopKSampler(3),
            new TopPSampler(0.95f),
            new MinPSampler(0.05f));

        float[] logits = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f];

        int result = pipeline.Sample(logits, []);

        Assert.InRange(result, 0, logits.Length - 1);
    }

    [Fact]
    public void Sample_ComposableViaOptions_ProducesValidIndex()
    {
        var options = new InferenceOptions
        {
            SamplerSteps =
            [
                new TemperatureSampler(0.8f),
                new TopKSampler(3)
            ],
            Seed = 42,
            MaxTokens = 10
        };
        var pipeline = new SamplerPipeline(options);

        float[] logits = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f];

        int result = pipeline.Sample(logits, []);

        Assert.InRange(result, 0, logits.Length - 1);
    }

    [Fact]
    public void Sample_ComposableSeededDeterminism()
    {
        var pipeline1 = new SamplerPipeline(
            processors: null,
            steps: [new TemperatureSampler(0.8f), new TopKSampler(3)],
            seed: 42);
        var pipeline2 = new SamplerPipeline(
            processors: null,
            steps: [new TemperatureSampler(0.8f), new TopKSampler(3)],
            seed: 42);

        float[] logits1 = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f];
        float[] logits2 = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f];

        int result1 = pipeline1.Sample(logits1, []);
        int result2 = pipeline2.Sample(logits2, []);

        Assert.Equal(result1, result2);
    }

    private static ISamplerStep[] GetSamplerSteps(SamplerPipeline pipeline)
    {
        var field = typeof(SamplerPipeline).GetField("_steps", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(field);
        var value = field!.GetValue(pipeline);
        Assert.NotNull(value);
        return (ISamplerStep[])value!;
    }
}
