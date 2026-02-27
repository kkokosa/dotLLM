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
}
