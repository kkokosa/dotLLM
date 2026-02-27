namespace DotLLM.Core.Diagnostics;

/// <summary>
/// Points in the transformer forward pass where inference hooks can intercept activations.
/// </summary>
public enum HookPoint
{
    /// <summary>After token embedding lookup, before the first layer.</summary>
    PostEmbedding,

    /// <summary>Before attention computation in a layer.</summary>
    PreAttention,

    /// <summary>After attention computation in a layer.</summary>
    PostAttention,

    /// <summary>Before the feed-forward network in a layer.</summary>
    PreFfn,

    /// <summary>After the feed-forward network in a layer.</summary>
    PostFfn,

    /// <summary>After the full layer (residual stream).</summary>
    PostLayer,

    /// <summary>Before the language model head (final projection).</summary>
    PreLmHead,

    /// <summary>After the language model head (logits).</summary>
    PostLmHead
}
