namespace DotLLM.Core.Configuration;

/// <summary>
/// Attention mechanism type used by the model.
/// </summary>
public enum AttentionType
{
    /// <summary>Grouped-Query Attention (subsumes MHA and MQA).</summary>
    GQA,

    /// <summary>Multi-head Latent Attention (DeepSeek).</summary>
    MLA
}
