namespace DotLLM.Core.Configuration;

/// <summary>
/// Extension methods for <see cref="QuantizationType"/> providing byte-size calculations.
/// Block sizes match the GGUF spec and <see cref="Tensors.DType"/> static instances.
/// </summary>
public static class QuantizationTypeExtensions
{
    /// <summary>
    /// Computes the total byte count for <paramref name="elementCount"/> elements stored
    /// in the given quantization format.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Unknown quantization type.</exception>
    public static long ComputeByteCount(this QuantizationType qt, long elementCount) => qt switch
    {
        QuantizationType.F32 => elementCount * 4,
        QuantizationType.F16 => elementCount * 2,
        QuantizationType.Q4_0 => elementCount / 32 * 18,
        QuantizationType.Q4_1 => elementCount / 32 * 20,
        QuantizationType.Q5_0 => elementCount / 32 * 22,
        QuantizationType.Q5_1 => elementCount / 32 * 24,
        QuantizationType.Q8_0 => elementCount / 32 * 34,
        QuantizationType.Q4_K => elementCount / 256 * 144,
        QuantizationType.Q5_K => elementCount / 256 * 176,
        QuantizationType.Q6_K => elementCount / 256 * 210,
        _ => throw new ArgumentOutOfRangeException(nameof(qt), qt,
            $"Unknown quantization type: {qt}"),
    };
}
