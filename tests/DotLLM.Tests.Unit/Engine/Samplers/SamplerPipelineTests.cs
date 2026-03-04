using System.Numerics.Tensors;
using DotLLM.Core.Configuration;
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
}
