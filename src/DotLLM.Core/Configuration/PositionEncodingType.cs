namespace DotLLM.Core.Configuration;

/// <summary>
/// Type of positional encoding used by the model.
/// </summary>
public enum PositionEncodingType
{
    /// <summary>Rotary Position Embeddings.</summary>
    RoPE,

    /// <summary>Attention with Linear Biases.</summary>
    ALiBi,

    /// <summary>Learned absolute position embeddings.</summary>
    Absolute,

    /// <summary>No positional encoding.</summary>
    None
}
