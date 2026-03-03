using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Cpu.PositionEncoding;

/// <summary>
/// CPU implementation of Rotary Position Embeddings (RoPE).
/// Manages pre-computed cos/sin frequency tables and delegates rotation to <see cref="RoPE"/>.
/// Applies rotation in-place to the underlying tensor data.
/// </summary>
public sealed class RoPEPositionEncoding : IPositionEncoding, IDisposable
{
    private float[]? _cosTable;
    private float[]? _sinTable;
    private int _maxSeqLen;
    private int _headDim;
    private bool _disposed;

    /// <inheritdoc/>
    public void PrecomputeTables(int maxSeqLen, ModelConfig config)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var ropeConfig = config.RoPEConfig
            ?? throw new InvalidOperationException("ModelConfig.RoPEConfig is required for RoPE position encoding.");

        int headDim = ropeConfig.DimensionCount > 0 ? ropeConfig.DimensionCount : config.HeadDim;
        int halfDim = headDim / 2;
        int tableLen = maxSeqLen * halfDim;

        _cosTable = new float[tableLen];
        _sinTable = new float[tableLen];
        _maxSeqLen = maxSeqLen;
        _headDim = headDim;

        RoPE.PrecomputeFrequencyTable(maxSeqLen, headDim, ropeConfig.Theta, _cosTable, _sinTable);
    }

    /// <inheritdoc/>
    public (ITensor Q, ITensor K) Apply(ITensor q, ITensor k, ReadOnlySpan<int> positions)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_cosTable is null || _sinTable is null)
            throw new InvalidOperationException("PrecomputeTables must be called before Apply.");

        if (q.DType != DType.Float32 || k.DType != DType.Float32)
            throw new ArgumentException("RoPE currently supports Float32 tensors only.");

        // Extract dimensions from tensor shapes.
        // Expected layout: [seqLen, numHeads * headDim] for Q, [seqLen, numKvHeads * headDim] for K.
        int seqLen = positions.Length;
        int qTotalDim = (int)(q.ElementCount / seqLen);
        int kTotalDim = (int)(k.ElementCount / seqLen);
        int numHeads = qTotalDim / _headDim;
        int numKvHeads = kTotalDim / _headDim;

        // Get mutable spans over tensor data — RoPE rotates in-place.
        unsafe
        {
            var qSpan = new Span<float>((void*)q.DataPointer, (int)q.ElementCount);
            var kSpan = new Span<float>((void*)k.DataPointer, (int)k.ElementCount);

            RoPE.Execute(qSpan, kSpan, positions,
                         numHeads, numKvHeads, _headDim,
                         _cosTable, _sinTable);
        }

        // Return same tensors — rotation was in-place.
        return (q, k);
    }

    /// <inheritdoc/>
    public void InvalidateCache()
    {
        _cosTable = null;
        _sinTable = null;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _cosTable = null;
            _sinTable = null;
            _disposed = true;
        }
    }
}
