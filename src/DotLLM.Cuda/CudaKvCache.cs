using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// GPU-resident KV-cache storing FP16 key and value vectors per layer.
/// Layout: [maxSeqLen, numKvHeads * headDim] per layer, FP16.
/// </summary>
public sealed class CudaKvCache : IKvCache
{
    private readonly nint[] _keys;
    private readonly nint[] _values;
    private readonly int _numLayers;
    private readonly int _kvStride;    // numKvHeads * headDim
    private readonly int _maxSeqLen;
    private int _currentLength;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>
    /// Allocates GPU KV-cache buffers for all layers.
    /// </summary>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="numKvHeads">Number of KV attention heads.</param>
    /// <param name="headDim">Dimension per head.</param>
    /// <param name="maxSeqLen">Maximum sequence length.</param>
    public CudaKvCache(int numLayers, int numKvHeads, int headDim, int maxSeqLen)
    {
        _numLayers = numLayers;
        _kvStride = numKvHeads * headDim;
        _maxSeqLen = maxSeqLen;
        _keys = new nint[numLayers];
        _values = new nint[numLayers];

        long bytesPerLayer = (long)maxSeqLen * _kvStride * sizeof(ushort); // FP16
        for (int i = 0; i < numLayers; i++)
        {
            CudaDriverApi.cuMemAlloc_v2(out _keys[i], (nuint)bytesPerLayer).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _values[i], (nuint)bytesPerLayer).ThrowOnError();
        }
    }

    /// <summary>
    /// Updates KV-cache from device pointers (used by <see cref="CudaTransformerModel"/>).
    /// </summary>
    /// <param name="keysDevice">Device pointer to new K data [seqLen, kvStride] FP16.</param>
    /// <param name="valuesDevice">Device pointer to new V data [seqLen, kvStride] FP16.</param>
    /// <param name="positions">Host-side positions for updating _currentLength.</param>
    /// <param name="seqLen">Number of new tokens.</param>
    /// <param name="layerIndex">Layer index.</param>
    /// <param name="stream">CUDA stream (currently unused — copies are synchronous).</param>
    internal void UpdateDevice(nint keysDevice, nint valuesDevice,
                                 ReadOnlySpan<int> positions, int seqLen,
                                 int layerIndex, nint stream)
    {
        long rowBytes = (long)_kvStride * sizeof(ushort); // FP16 KV-cache

        for (int i = 0; i < seqLen; i++)
        {
            int pos = positions[i];
            nint kDst = _keys[layerIndex] + (nint)(pos * rowBytes);
            nint vDst = _values[layerIndex] + (nint)(pos * rowBytes);
            nint kSrc = keysDevice + (nint)(i * rowBytes);
            nint vSrc = valuesDevice + (nint)(i * rowBytes);

            CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, kSrc, (nuint)rowBytes, stream).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, vSrc, (nuint)rowBytes, stream).ThrowOnError();
        }

        // Update length
        for (int i = 0; i < seqLen; i++)
        {
            int newLen = positions[i] + 1;
            if (newLen > _currentLength)
                _currentLength = newLen;
        }
    }

    /// <summary>Returns device pointer to cached keys for the given layer.</summary>
    internal nint GetKeysPtr(int layerIndex) => _keys[layerIndex];

    /// <summary>Returns device pointer to cached values for the given layer.</summary>
    internal nint GetValuesPtr(int layerIndex) => _values[layerIndex];

    // ── IKvCache interface implementation ─────────────────────────────

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
    {
        throw new NotSupportedException("CudaKvCache.Update(ITensor) not supported. Use UpdateDevice().");
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        throw new NotSupportedException("CudaKvCache.Update(TensorRef) not supported. Use UpdateDevice().");
    }

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
    {
        throw new NotSupportedException("CudaKvCache.GetKeys(ITensor) not supported. Use GetKeysPtr().");
    }

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
    {
        throw new NotSupportedException("CudaKvCache.GetValues(ITensor) not supported. Use GetValuesPtr().");
    }

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex) =>
        new(_currentLength, _kvStride, DType.Float16, 0, _keys[layerIndex]);

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex) =>
        new(_currentLength, _kvStride, DType.Float16, 0, _values[layerIndex]);

    /// <inheritdoc/>
    public void Dispose()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            if (_keys[i] != 0) { CudaDriverApi.cuMemFree_v2(_keys[i]); _keys[i] = 0; }
            if (_values[i] != 0) { CudaDriverApi.cuMemFree_v2(_values[i]); _values[i] = 0; }
        }
    }
}
