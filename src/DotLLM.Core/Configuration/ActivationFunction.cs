namespace DotLLM.Core.Configuration;

/// <summary>
/// Activation function used in feed-forward network layers.
/// </summary>
public enum ActivationFunction
{
    /// <summary>Sigmoid Linear Unit.</summary>
    SiLU,

    /// <summary>Gaussian Error Linear Unit.</summary>
    GELU,

    /// <summary>GELU with tanh approximation.</summary>
    GELUTanh
}
