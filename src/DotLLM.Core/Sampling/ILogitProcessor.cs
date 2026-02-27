namespace DotLLM.Core.Sampling;

/// <summary>
/// Processes logits before sampling, using token history for context-dependent adjustments.
/// Examples: repetition penalty, frequency penalty, presence penalty.
/// </summary>
public interface ILogitProcessor
{
    /// <summary>
    /// Processes logits in-place using previous token context.
    /// </summary>
    /// <param name="logits">Logits to modify. Length = vocab size.</param>
    /// <param name="previousTokens">Previously generated token IDs in this sequence.</param>
    /// <param name="context">Processor parameters for the current request.</param>
    void Process(Span<float> logits, IReadOnlyList<int> previousTokens, ProcessorContext context);
}
