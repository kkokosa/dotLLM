namespace DotLLM.Core.Configuration;

/// <summary>
/// GGUF quantization type identifiers. Values match the GGUF spec.
/// </summary>
public enum QuantizationType
{
    /// <summary>32-bit IEEE float.</summary>
    F32 = 0,

    /// <summary>16-bit IEEE float.</summary>
    F16 = 1,

    /// <summary>4-bit quantization, group size 32, no min.</summary>
    Q4_0 = 2,

    /// <summary>4-bit quantization, group size 32, with min.</summary>
    Q4_1 = 3,

    /// <summary>5-bit quantization, group size 32, no min.</summary>
    Q5_0 = 6,

    /// <summary>5-bit quantization, group size 32, with min.</summary>
    Q5_1 = 7,

    /// <summary>8-bit quantization, group size 32.</summary>
    Q8_0 = 8,

    /// <summary>4-bit K-quant, super-block of 256.</summary>
    Q4_K = 12,

    /// <summary>5-bit K-quant, super-block of 256.</summary>
    Q5_K = 13,

    /// <summary>6-bit K-quant, super-block of 256.</summary>
    Q6_K = 14
}
