namespace DotLLM.Core.Configuration;

/// <summary>
/// RoPE context-length scaling strategy.
/// </summary>
public enum RoPEScalingType
{
    /// <summary>No scaling applied.</summary>
    None,

    /// <summary>Linear interpolation of position indices.</summary>
    Linear,

    /// <summary>Yet another RoPE extensioN method.</summary>
    YaRN,

    /// <summary>Neural Tangent Kernel scaling.</summary>
    NTK,

    /// <summary>Dynamic NTK-aware scaling.</summary>
    DynamicNTK
}
