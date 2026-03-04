using System.Numerics.Tensors;
using DotLLM.Core.Configuration;
using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Orchestrates the sampling pipeline: logit processors → sampler steps → final token selection.
/// Can be built automatically from <see cref="InferenceOptions"/> or composed explicitly
/// from individual <see cref="ISamplerStep"/> instances.
/// </summary>
public sealed class SamplerPipeline
{
    private readonly ILogitProcessor[] _processors;
    private readonly ISamplerStep[] _steps;
    private readonly ProcessorContext _processorContext;
    private readonly SamplerContext _samplerContext;
    private readonly Random _rng;
    private readonly bool _greedy;

    /// <summary>
    /// Creates a composable sampling pipeline from explicit steps.
    /// Steps are applied in the order provided, followed by categorical sampling.
    /// </summary>
    /// <param name="steps">Sampler steps to apply in order (e.g., temperature → top-K → top-P → min-P).</param>
    public SamplerPipeline(params ISamplerStep[] steps)
        : this(processors: null, steps: steps, seed: null)
    {
    }

    /// <summary>
    /// Creates a composable sampling pipeline from explicit processors and steps.
    /// </summary>
    /// <param name="processors">Logit processors (e.g., repetition penalty). Applied before steps.</param>
    /// <param name="steps">Sampler steps to apply in order.</param>
    /// <param name="seed">Random seed for reproducible sampling. Null = non-deterministic.</param>
    public SamplerPipeline(
        IReadOnlyList<ILogitProcessor>? processors,
        IReadOnlyList<ISamplerStep> steps,
        int? seed = null)
    {
        _greedy = false;
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
        _processors = processors?.ToArray() ?? [];
        _steps = steps.ToArray();
        _processorContext = new ProcessorContext(1.0f, 0, SequenceId: 0);
        _samplerContext = default;
    }

    /// <summary>
    /// Creates a new sampling pipeline from the given inference options.
    /// When <see cref="InferenceOptions.SamplerSteps"/> is set, uses those explicit steps.
    /// Otherwise builds steps automatically from flat properties, skipping disabled ones.
    /// </summary>
    public SamplerPipeline(InferenceOptions options)
    {
        _rng = options.Seed.HasValue ? new Random(options.Seed.Value) : new Random();

        // Explicit steps provided — use composable path
        if (options.SamplerSteps is not null)
        {
            _greedy = false;
            _steps = options.SamplerSteps.ToArray();

            // Build processors: use explicit list if provided, otherwise auto-build from flat properties
            if (options.LogitProcessors is not null)
            {
                _processors = options.LogitProcessors.ToArray();
            }
            else
            {
                var processors = new List<ILogitProcessor>();
                if (options.RepetitionPenalty != 1.0f)
                    processors.Add(new RepetitionPenaltyProcessor());
                _processors = processors.ToArray();
            }

            _processorContext = new ProcessorContext(
                options.RepetitionPenalty,
                options.RepetitionPenaltyWindow,
                SequenceId: 0);
            _samplerContext = new SamplerContext(
                options.Temperature,
                options.TopK,
                options.TopP,
                options.MinP,
                options.Seed);
            return;
        }

        // Auto-build from flat properties
        _greedy = options.Temperature <= 0f;

        // Build processor chain (only add if enabled)
        if (options.LogitProcessors is not null)
        {
            _processors = options.LogitProcessors.ToArray();
        }
        else
        {
            var processors = new List<ILogitProcessor>();
            if (options.RepetitionPenalty != 1.0f)
                processors.Add(new RepetitionPenaltyProcessor());
            _processors = processors.ToArray();
        }

        // Build sampler step chain (only add if enabled)
        var steps = new List<ISamplerStep>();
        if (!_greedy)
        {
            if (options.Temperature != 1.0f)
                steps.Add(new TemperatureSampler());
            if (options.TopK > 0)
                steps.Add(new TopKSampler());
            if (options.TopP < 1.0f)
                steps.Add(new TopPSampler());
            if (options.MinP > 0f)
                steps.Add(new MinPSampler());
        }
        _steps = steps.ToArray();

        _processorContext = new ProcessorContext(
            options.RepetitionPenalty,
            options.RepetitionPenaltyWindow,
            SequenceId: 0);

        _samplerContext = new SamplerContext(
            options.Temperature,
            options.TopK,
            options.TopP,
            options.MinP,
            options.Seed);
    }

    /// <summary>
    /// Samples a token from the given logits, applying all enabled processors and steps.
    /// </summary>
    /// <param name="logits">Logit values to sample from (modified in-place).</param>
    /// <param name="previousTokens">Previously generated token IDs for repetition penalty.</param>
    /// <returns>The sampled token index.</returns>
    public int Sample(Span<float> logits, IReadOnlyList<int> previousTokens)
    {
        // 1. Run logit processors (repetition penalty)
        for (int i = 0; i < _processors.Length; i++)
            _processors[i].Process(logits, previousTokens, _processorContext);

        // 2. Greedy: argmax, skip everything else
        if (_greedy)
            return TensorPrimitives.IndexOfMax(logits);

        // 3. Run sampler steps (temperature → top-k → top-p → min-p)
        for (int i = 0; i < _steps.Length; i++)
            _steps[i].Apply(logits, _samplerContext);

        // 4. Categorical sample
        return CategoricalSampler.Sample(logits, _rng);
    }
}
