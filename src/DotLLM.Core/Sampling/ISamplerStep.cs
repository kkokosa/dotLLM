namespace DotLLM.Core.Sampling;

/// <summary>
/// A single step in the composable sampling pipeline.
/// Steps are applied in order to transform logits into a token selection.
/// Examples: temperature scaling, top-K filtering, top-P (nucleus) filtering, min-P filtering.
/// </summary>
public interface ISamplerStep
{
    /// <summary>
    /// Applies this sampling step to the logits in-place.
    /// </summary>
    /// <param name="logits">Logits to modify. Length = vocab size.</param>
    /// <param name="context">Sampling parameters for the current request.</param>
    void Apply(Span<float> logits, SamplerContext context);
}
