namespace DotLLM.Core.Sampling;

/// <summary>
/// Context passed to <see cref="ILogitProcessor.Process"/> providing token history and request metadata.
/// </summary>
/// <param name="RepetitionPenalty">Repetition penalty factor. 1.0 = disabled.</param>
/// <param name="RepetitionPenaltyWindow">Number of recent tokens to consider for repetition penalty.</param>
/// <param name="SequenceId">Identifier for the current sequence/request.</param>
public readonly record struct ProcessorContext(
    float RepetitionPenalty,
    int RepetitionPenaltyWindow,
    int SequenceId);
