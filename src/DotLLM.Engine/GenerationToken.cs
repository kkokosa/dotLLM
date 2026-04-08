namespace DotLLM.Engine;

/// <summary>
/// A single token yielded during streaming generation.
/// </summary>
/// <param name="TokenId">The sampled token ID.</param>
/// <param name="Text">Incremental decoded text. May be empty for incomplete UTF-8 sequences, multi-char when a sequence completes.</param>
/// <param name="FinishReason">Null during generation; set on the final yielded token.</param>
/// <param name="Timings">Null except on the final token, which carries prefill/decode/sampling breakdown.</param>
/// <param name="Logprobs">Per-token log-probability info. Null when logprobs not requested.</param>
public readonly record struct GenerationToken(
    int TokenId,
    string Text,
    FinishReason? FinishReason,
    InferenceTimings? Timings = null,
    TokenLogprobInfo? Logprobs = null);
