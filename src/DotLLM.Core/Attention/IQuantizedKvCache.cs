using DotLLM.Core.Configuration;

namespace DotLLM.Core.Attention;

/// <summary>
/// Extended KV-cache interface for quantized caches that store data in Q8_0/Q4_0 format
/// with an optional full-precision window for recent tokens. Attention kernels check
/// <c>if (kvCache is IQuantizedKvCache qkv)</c> to use the quantized code path.
/// </summary>
public interface IQuantizedKvCache : IKvCache
{
    /// <summary>Number of positions stored in quantized format.</summary>
    int QuantizedLength { get; }

    /// <summary>Number of positions in the full-precision window.</summary>
    int WindowLength { get; }

    /// <summary>Configured window capacity (the maximum window size, not current occupancy).</summary>
    int WindowCapacity { get; }

    /// <summary>Quantization type for cached keys.</summary>
    KvCacheDType KeyDType { get; }

    /// <summary>Quantization type for cached values.</summary>
    KvCacheDType ValueDType { get; }

    /// <summary>
    /// Gets pointer to quantized key data for a layer.
    /// Layout: <c>[QuantizedLength]</c> rows of quantized blocks.
    /// </summary>
    nint GetQuantizedKeysPtr(int layerIndex);

    /// <summary>
    /// Gets pointer to quantized value data for a layer.
    /// Layout: <c>[QuantizedLength]</c> rows of quantized blocks.
    /// </summary>
    nint GetQuantizedValuesPtr(int layerIndex);

    /// <summary>
    /// Gets pointer to full-precision window key data for a layer.
    /// Layout: <c>[WindowLength, kvStride]</c> in FP32 (CPU) or FP16 (GPU).
    /// </summary>
    nint GetWindowKeysPtr(int layerIndex);

    /// <summary>
    /// Gets pointer to full-precision window value data for a layer.
    /// Layout: <c>[WindowLength, kvStride]</c> in FP32 (CPU) or FP16 (GPU).
    /// </summary>
    nint GetWindowValuesPtr(int layerIndex);

    /// <summary>
    /// Returns the byte size of one quantized row for keys.
    /// </summary>
    int KeyQuantizedRowBytes { get; }

    /// <summary>
    /// Returns the byte size of one quantized row for values.
    /// </summary>
    int ValueQuantizedRowBytes { get; }
}
