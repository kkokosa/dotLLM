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
        ApplyTransforms(logits, previousTokens);
        return SampleFromTransformed(logits);
    }

    /// <summary>
    /// True when this pipeline is effectively greedy (argmax selection, no stochastic sampling).
    /// Speculative decoding's accept/reject scheme uses this to match the pipeline mode for
    /// both the draft proposal distribution <c>q</c> and the target distribution <c>p</c>.
    /// </summary>
    public bool IsGreedy => _greedy;

    /// <summary>
    /// Applies the same chain of logit transforms (processors + sampler steps) that
    /// <see cref="Sample"/> would, but does <b>not</b> draw a token. The resulting span is the
    /// pre-softmax logit distribution the pipeline would sample from.
    /// </summary>
    /// <param name="logits">Logits, mutated in-place.</param>
    /// <param name="previousTokens">Tokens generated so far — used by repetition penalty.
    /// For correct results in speculative decoding, this must include any provisionally
    /// accepted draft tokens preceding the current position.</param>
    /// <remarks>
    /// In greedy configurations <c>_steps</c> is empty, so this is equivalent to running the
    /// processor chain only (repetition penalty if enabled). Callers that want the same token
    /// the pipeline would sample should call <see cref="SampleFromTransformed"/> on the result.
    /// </remarks>
    public void ApplyTransforms(Span<float> logits, IReadOnlyList<int> previousTokens)
    {
        for (int i = 0; i < _processors.Length; i++)
            _processors[i].Process(logits, previousTokens, _processorContext);

        if (_greedy)
            return;

        for (int i = 0; i < _steps.Length; i++)
            _steps[i].Apply(logits, _samplerContext);
    }

    /// <summary>
    /// Samples a token from logits that have already been transformed by
    /// <see cref="ApplyTransforms"/>. Greedy pipelines return <c>IndexOfMax</c>;
    /// otherwise a categorical draw via the pipeline's RNG.
    /// </summary>
    public int SampleFromTransformed(ReadOnlySpan<float> transformedLogits)
    {
        if (_greedy)
            return TensorPrimitives.IndexOfMax(transformedLogits);
        return CategoricalSampler.Sample(transformedLogits, _rng);
    }
}
