using DotLLM.Core.Tensors;

namespace DotLLM.Core.Attention;

/// <summary>
/// Key-Value cache for autoregressive attention. Stores projected K and V tensors across decoding steps.
/// </summary>
public interface IKvCache : IDisposable
{
    /// <summary>Current number of cached positions.</summary>
    int CurrentLength { get; }

    /// <summary>Maximum number of positions this cache can hold.</summary>
    int MaxLength { get; }

    /// <summary>
    /// Appends new key and value projections at the given positions.
    /// </summary>
    /// <param name="keys">Key projections for the new tokens.</param>
    /// <param name="values">Value projections for the new tokens.</param>
    /// <param name="positions">Position indices for the new entries.</param>
    /// <param name="layerIndex">Transformer layer index.</param>
    void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex);

    /// <summary>Gets the cached key tensor for a given layer.</summary>
    /// <param name="layerIndex">Transformer layer index.</param>
    /// <returns>Key tensor covering all cached positions.</returns>
    ITensor GetKeys(int layerIndex);

    /// <summary>Gets the cached value tensor for a given layer.</summary>
    /// <param name="layerIndex">Transformer layer index.</param>
    /// <returns>Value tensor covering all cached positions.</returns>
    ITensor GetValues(int layerIndex);

    /// <summary>
    /// Zero-allocation update using <see cref="TensorRef"/>. Preferred on the inference hot path.
    /// </summary>
    /// <param name="keys">Key projections as a lightweight tensor reference.</param>
    /// <param name="values">Value projections as a lightweight tensor reference.</param>
    /// <param name="positions">Position indices for the new entries.</param>
    /// <param name="layerIndex">Transformer layer index.</param>
    void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex);

    /// <summary>Gets cached keys as a zero-allocation <see cref="TensorRef"/>.</summary>
    /// <param name="layerIndex">Transformer layer index.</param>
    TensorRef GetKeysRef(int layerIndex);

    /// <summary>Gets cached values as a zero-allocation <see cref="TensorRef"/>.</summary>
    /// <param name="layerIndex">Transformer layer index.</param>
    TensorRef GetValuesRef(int layerIndex);

    /// <summary>
    /// Rolls back the cache to the given length, discarding entries beyond that position.
    /// Used by speculative decoding to discard rejected draft tokens.
    /// Allocated memory is retained and overwritten on subsequent Update calls.
    /// </summary>
    /// <param name="length">The new current length (must be &lt;= <see cref="CurrentLength"/>).</param>
    void Rollback(int length);

    /// <summary>
    /// Attempts to reserve in-place write slots for the K and V projections at the given
    /// <paramref name="positions"/>. When successful, callers can target <paramref name="kDst"/>
    /// and <paramref name="vDst"/> as the K/V projection output buffers, and run the
    /// post-projection in-place pipeline (AddBias, LoRA delta, QK-norm, RoPE) directly on
    /// those spans — avoiding the scratch buffer and the subsequent <c>Update</c>
    /// memcpy. Length advancement is deferred to <see cref="CommitSlot"/>; the caller must
    /// invoke <see cref="CommitSlot"/> after writing to keep <see cref="CurrentLength"/>
    /// consistent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns <c>false</c> when the cache cannot expose an in-place slot for the given
    /// positions — most commonly because positions are non-contiguous, exceed
    /// <see cref="MaxLength"/>, would span a paged-block boundary, or the underlying storage
    /// is quantized / device-resident. The caller must then fall back to the existing
    /// scratch + <c>Update</c> path.
    /// </para>
    /// <para>
    /// The default implementation returns <c>false</c>, preserving backward compatibility
    /// for every <see cref="IKvCache"/> implementation that has not opted in.
    /// </para>
    /// </remarks>
    /// <param name="layerIndex">Transformer layer index.</param>
    /// <param name="positions">Position indices for the new entries. Must be contiguous for
    /// the slot to be reservable.</param>
    /// <param name="kDst">On success, span covering the K cache slot for these positions
    /// (<c>positions.Length * kvStride</c> FP32 elements). Undefined on failure.</param>
    /// <param name="vDst">On success, span covering the V cache slot for these positions.
    /// Undefined on failure.</param>
    /// <returns><c>true</c> when a slot was reserved and <paramref name="kDst"/>/<paramref name="vDst"/>
    /// are valid in-place targets; <c>false</c> otherwise.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Implementations that expose raw layer
    /// storage throw when <paramref name="layerIndex"/> is outside the cache's layer range.
    /// An out-of-range layer is a caller bug, not a "cannot reserve" condition.</exception>
    bool TryReserveSlot(
        int layerIndex,
        ReadOnlySpan<int> positions,
        out Span<float> kDst,
        out Span<float> vDst)
    {
        kDst = default;
        vDst = default;
        return false;
    }

    /// <summary>
    /// Commits a prior successful <see cref="TryReserveSlot"/> call by advancing
    /// <see cref="CurrentLength"/> based on <paramref name="positions"/>. Idempotent across
    /// layers within the same forward pass — the maximum-position computation matches
    /// <c>Update</c>'s semantics.
    /// </summary>
    /// <remarks>
    /// The default implementation is a no-op. Callers must only invoke this after a
    /// successful <see cref="TryReserveSlot"/> on the same cache for the same positions.
    /// </remarks>
    /// <param name="layerIndex">Transformer layer index.</param>
    /// <param name="positions">Position indices for the entries written during the slot.</param>
    void CommitSlot(int layerIndex, ReadOnlySpan<int> positions)
    {
    }
}
