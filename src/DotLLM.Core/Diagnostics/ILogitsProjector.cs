namespace DotLLM.Core.Diagnostics;

/// <summary>
/// Projects a residual-stream hidden state through the model's final normalization and
/// language-model head to produce raw logits.
/// </summary>
/// <remarks>
/// <para>
/// Implemented by models that wish to expose their final-norm + LM-head as a stand-alone
/// projection — primarily for diagnostics tools such as the logit lens, which need to map
/// per-layer residuals through the same head the model itself uses, in order to compare
/// per-layer predictions against the final output.
/// </para>
/// <para>
/// Implementations <b>must</b> reuse the same code paths the forward pass uses so that calling
/// <see cref="ProjectToLogits(System.ReadOnlySpan{float}, System.Span{float})"/> with the
/// post-final-layer hidden state produced by a single-token forward returns logits equal
/// (within numerical noise of the same kernel) to the row the model returned from
/// <c>Forward</c>. The discriminating logit-lens test relies on this contract.
/// </para>
/// <para>
/// Implementations are <b>not</b> required to be safe for concurrent invocation with an
/// in-flight <c>Forward</c> on the same instance — they typically reuse the model's scratch
/// buffers and are intended to be called after a forward pass returns.
/// </para>
/// </remarks>
public interface ILogitsProjector
{
    /// <summary>
    /// Length of the input hidden state expected by <see cref="ProjectToLogits"/> — equals the
    /// model's residual-stream width (<c>HiddenSize</c>).
    /// </summary>
    int HiddenSize { get; }

    /// <summary>
    /// Length of the output logits produced by <see cref="ProjectToLogits"/> — equals the
    /// model's vocabulary size.
    /// </summary>
    int VocabSize { get; }

    /// <summary>
    /// Applies the model's final RMSNorm (or equivalent) and language-model head to
    /// <paramref name="hiddenState"/>, writing the resulting raw (un-softmaxed) logits to
    /// <paramref name="logits"/>.
    /// </summary>
    /// <param name="hiddenState">
    /// Residual-stream hidden state of length <see cref="HiddenSize"/>. May come from any layer's
    /// <c>PostLayer</c> hook point — the projector applies the final norm itself.
    /// </param>
    /// <param name="logits">Destination buffer of length <see cref="VocabSize"/>.</param>
    void ProjectToLogits(ReadOnlySpan<float> hiddenState, Span<float> logits);
}
