namespace DotLLM.Core.Configuration;

/// <summary>
/// Configuration for KV-cache quantization. Allows separate quantization types for keys and values,
/// plus a mixed-precision window that keeps recent tokens in full precision.
/// </summary>
/// <param name="KeyDType">Quantization format for cached keys. <see cref="KvCacheDType.F32"/> = no quantization.</param>
/// <param name="ValueDType">Quantization format for cached values. <see cref="KvCacheDType.F32"/> = no quantization.</param>
/// <param name="MixedPrecisionWindowSize">
/// Number of recent tokens kept in full precision. Older tokens beyond this window
/// are stored in <paramref name="KeyDType"/>/<paramref name="ValueDType"/> format.
/// 0 = all tokens quantized immediately.
/// </param>
public readonly record struct KvCacheConfig(
    KvCacheDType KeyDType = KvCacheDType.F32,
    KvCacheDType ValueDType = KvCacheDType.F32,
    int MixedPrecisionWindowSize = 0)
{
    /// <summary>Default config: full precision, no quantization.</summary>
    public static KvCacheConfig Default => new();

    /// <summary>Returns true if any quantization is configured (key or value).</summary>
    public bool IsQuantized => KeyDType != KvCacheDType.F32 || ValueDType != KvCacheDType.F32;

    /// <summary>Parses a CLI string to <see cref="KvCacheDType"/>.</summary>
    public static KvCacheDType ParseDType(string value) => value.ToLowerInvariant() switch
    {
        "f32" or "fp32" => KvCacheDType.F32,
        "q8_0" or "q8" => KvCacheDType.Q8_0,
        "q4_0" or "q4" => KvCacheDType.Q4_0,
        _ => throw new ArgumentException($"Unknown KV-cache type: '{value}'. Supported: f32, q8_0, q4_0.")
    };
}
