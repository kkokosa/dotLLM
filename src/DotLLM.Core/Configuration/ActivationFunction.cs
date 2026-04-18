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
    GELUTanh,

    /// <summary>
    /// Squared ReLU: <c>y = max(0, x)^2</c>. Used by Nemotron-H FFN layers.
    /// </summary>
    ReluSquared
}
