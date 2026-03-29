namespace DotLLM.Core.Configuration;

/// <summary>
/// Data types supported for KV-cache storage. Distinct from <see cref="QuantizationType"/>
/// because KV-cache only supports a subset of quantization formats.
/// </summary>
public enum KvCacheDType
{
    /// <summary>Full precision: FP32 on CPU, FP16 on GPU.</summary>
    F32 = 0,

    /// <summary>Q8_0: 34 bytes per 32 elements (Half scale + 32 int8 values).</summary>
    Q8_0 = 1,

    /// <summary>Q4_0: 18 bytes per 32 elements (Half scale + 16 packed nibble bytes).</summary>
    Q4_0 = 2
}
