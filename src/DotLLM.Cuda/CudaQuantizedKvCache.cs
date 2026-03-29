using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// GPU-resident quantized KV-cache with dual-region storage:
/// quantized buffer (Q8_0/Q4_0 in device memory) + FP16 window for recent tokens.
/// <para>
/// Attention uses a temporary FP16 scratch buffer: dequant quantized region → scratch,
/// then combine with window data and call the regular attention kernel.
/// Memory savings come from permanent quantized storage between attention calls.
/// </para>
/// </summary>
public sealed class CudaQuantizedKvCache : IQuantizedKvCache
{
    private const int BlockSize = 32;
    private const int Q8_0BlockBytes = 34;
    private const int Q4_0BlockBytes = 18;

    private readonly nint[] _keysQuant;     // device ptrs: quantized K
    private readonly nint[] _valuesQuant;   // device ptrs: quantized V
    private readonly nint[]? _keysWindow;   // device ptrs: FP16 K window
    private readonly nint[]? _valuesWindow; // device ptrs: FP16 V window
    private readonly int _numLayers;
    private readonly int _kvStride;
    private readonly int _maxSeqLen;
    private readonly int _windowSize;
    private readonly int _keyQuantRowBytes;
    private readonly int _valueQuantRowBytes;
    private int _currentLength;
    private int _quantizedLength;

    // Scratch buffers for dequantized K/V during attention (one pair, reused across layers)
    private nint _kScratch;  // device ptr: FP16 [maxSeqLen, kvStride]
    private nint _vScratch;  // device ptr: FP16 [maxSeqLen, kvStride]

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <inheritdoc/>
    public int QuantizedLength => _quantizedLength;

    /// <inheritdoc/>
    public int WindowLength => _windowSize > 0 ? Math.Min(_currentLength, _windowSize) : 0;

    /// <inheritdoc/>
    public KvCacheDType KeyDType { get; }

    /// <inheritdoc/>
    public KvCacheDType ValueDType { get; }

    /// <inheritdoc/>
    public int KeyQuantizedRowBytes => _keyQuantRowBytes;

    /// <inheritdoc/>
    public int ValueQuantizedRowBytes => _valueQuantRowBytes;

    /// <summary>Total device memory allocated in bytes.</summary>
    public long AllocatedBytes { get; }

    /// <summary>
    /// Creates a GPU quantized KV-cache.
    /// </summary>
    public CudaQuantizedKvCache(int numLayers, int numKvHeads, int headDim, int maxSeqLen,
                                 KvCacheConfig config)
    {
        _numLayers = numLayers;
        _kvStride = numKvHeads * headDim;
        _maxSeqLen = maxSeqLen;
        _windowSize = config.MixedPrecisionWindowSize;
        KeyDType = config.KeyDType;
        ValueDType = config.ValueDType;

        if (_kvStride % BlockSize != 0)
            throw new ArgumentException(
                $"kvStride ({_kvStride}) must be a multiple of {BlockSize} for quantization.");

        _keyQuantRowBytes = ComputeQuantRowBytes(_kvStride, config.KeyDType);
        _valueQuantRowBytes = ComputeQuantRowBytes(_kvStride, config.ValueDType);

        _keysQuant = new nint[numLayers];
        _valuesQuant = new nint[numLayers];

        long totalBytes = 0;

        // Allocate quantized buffers
        for (int i = 0; i < numLayers; i++)
        {
            nuint kBytes = (nuint)((long)maxSeqLen * _keyQuantRowBytes);
            nuint vBytes = (nuint)((long)maxSeqLen * _valueQuantRowBytes);
            CudaDriverApi.cuMemAlloc_v2(out _keysQuant[i], kBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _valuesQuant[i], vBytes).ThrowOnError();
            totalBytes += (long)(kBytes + vBytes);
        }

        // Allocate FP16 window buffers
        if (_windowSize > 0)
        {
            _keysWindow = new nint[numLayers];
            _valuesWindow = new nint[numLayers];
            nuint windowBytes = (nuint)((long)_windowSize * _kvStride * sizeof(ushort));
            for (int i = 0; i < numLayers; i++)
            {
                CudaDriverApi.cuMemAlloc_v2(out _keysWindow[i], windowBytes).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out _valuesWindow[i], windowBytes).ThrowOnError();
                totalBytes += (long)(windowBytes * 2);
            }
        }

        // Allocate scratch buffers for attention dequant (one pair, reused across layers)
        long scratchBytes = (long)maxSeqLen * _kvStride * sizeof(ushort);
        CudaDriverApi.cuMemAlloc_v2(out _kScratch, (nuint)scratchBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _vScratch, (nuint)scratchBytes).ThrowOnError();
        totalBytes += scratchBytes * 2;

        AllocatedBytes = totalBytes;
    }

    /// <summary>
    /// Updates KV-cache from device FP16 data. Handles quantize-on-evict for the window.
    /// Position-addressed and idempotent: safe to call once per layer with the same positions.
    /// </summary>
    internal void UpdateDevice(nint keysDevice, nint valuesDevice,
                                ReadOnlySpan<int> positions, int seqLen,
                                int layerIndex, nint stream, CudaKernels kernels)
    {
        long fp16RowBytes = (long)_kvStride * sizeof(ushort);

        // Compute new sequence length (idempotent across layer calls with same positions).
        int maxPos = positions[0];
        for (int i = 1; i < seqLen; i++)
            if (positions[i] > maxPos) maxPos = positions[i];
        int newLength = maxPos + 1;

        if (_windowSize > 0)
        {
            // Evict positions that fall outside the window for THIS layer.
            int prevQuantLen = Math.Max(0, _currentLength - _windowSize);
            int newQuantLen = Math.Max(0, newLength - _windowSize);

            for (int evictPos = prevQuantLen; evictPos < newQuantLen; evictPos++)
            {
                int ringIdx = evictPos % _windowSize;

                nint evictedK = _keysWindow![layerIndex] + (nint)(ringIdx * fp16RowBytes);
                nint quantDstK = _keysQuant[layerIndex] + (nint)((long)evictPos * _keyQuantRowBytes);
                kernels.LaunchQuantKv(evictedK, quantDstK, _kvStride, KeyDType, stream);

                nint evictedV = _valuesWindow![layerIndex] + (nint)(ringIdx * fp16RowBytes);
                nint quantDstV = _valuesQuant[layerIndex] + (nint)((long)evictPos * _valueQuantRowBytes);
                kernels.LaunchQuantKv(evictedV, quantDstV, _kvStride, ValueDType, stream);
            }

            // Write new FP16 data into window ring buffer (position-addressed, idempotent).
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                int ringIdx = pos % _windowSize;
                nint kDst = _keysWindow![layerIndex] + (nint)(ringIdx * fp16RowBytes);
                nint vDst = _valuesWindow![layerIndex] + (nint)(ringIdx * fp16RowBytes);
                nint kSrc = keysDevice + (nint)(i * fp16RowBytes);
                nint vSrc = valuesDevice + (nint)(i * fp16RowBytes);

                CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, kSrc, (nuint)fp16RowBytes, stream).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, vSrc, (nuint)fp16RowBytes, stream).ThrowOnError();
            }

            _quantizedLength = newQuantLen;
        }
        else
        {
            // Pure quantized: quantize directly at each position (position-addressed).
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                nint kSrc = keysDevice + (nint)(i * fp16RowBytes);
                nint vSrc = valuesDevice + (nint)(i * fp16RowBytes);
                nint kDst = _keysQuant[layerIndex] + (nint)((long)pos * _keyQuantRowBytes);
                nint vDst = _valuesQuant[layerIndex] + (nint)((long)pos * _valueQuantRowBytes);

                kernels.LaunchQuantKv(kSrc, kDst, _kvStride, KeyDType, stream);
                kernels.LaunchQuantKv(vSrc, vDst, _kvStride, ValueDType, stream);
            }

            _quantizedLength = newLength;
        }

        _currentLength = newLength;
    }

    /// <summary>
    /// Prepares dequantized FP16 scratch buffers for attention. Returns device pointers
    /// to contiguous FP16 K/V covering the full sequence (quantized + window).
    /// </summary>
    internal (nint kPtr, nint vPtr) PrepareAttentionScratch(int layerIndex, nint stream, CudaKernels kernels)
    {
        long fp16RowBytes = (long)_kvStride * sizeof(ushort);

        // Phase 1: Dequant quantized region → scratch[0..quantizedLength)
        if (_quantizedLength > 0)
        {
            int totalElements = _quantizedLength * _kvStride;
            kernels.LaunchDequantToF16(
                _keysQuant[layerIndex],
                KeyDType == KvCacheDType.Q8_0 ? Core.Configuration.QuantizationType.Q8_0 : Core.Configuration.QuantizationType.Q4_0,
                _kScratch, totalElements, stream);

            kernels.LaunchDequantToF16(
                _valuesQuant[layerIndex],
                ValueDType == KvCacheDType.Q8_0 ? Core.Configuration.QuantizationType.Q8_0 : Core.Configuration.QuantizationType.Q4_0,
                _vScratch, totalElements, stream);
        }

        // Phase 2: Copy window region → scratch[quantizedLength..currentLength)
        int windowLen = WindowLength;
        if (windowLen > 0 && _keysWindow != null)
        {
            long windowBytes = (long)windowLen * fp16RowBytes;
            nint kDst = _kScratch + (nint)(_quantizedLength * fp16RowBytes);
            nint vDst = _vScratch + (nint)(_quantizedLength * fp16RowBytes);

            CudaDriverApi.cuMemcpyDtoDAsync_v2(kDst, _keysWindow[layerIndex], (nuint)windowBytes, stream).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync_v2(vDst, _valuesWindow![layerIndex], (nuint)windowBytes, stream).ThrowOnError();
        }

        return (_kScratch, _vScratch);
    }

    // ── IQuantizedKvCache implementation ────────────────────────────

    /// <inheritdoc/>
    public nint GetQuantizedKeysPtr(int layerIndex) => _keysQuant[layerIndex];

    /// <inheritdoc/>
    public nint GetQuantizedValuesPtr(int layerIndex) => _valuesQuant[layerIndex];

    /// <inheritdoc/>
    public nint GetWindowKeysPtr(int layerIndex)
        => _keysWindow != null ? _keysWindow[layerIndex] : 0;

    /// <inheritdoc/>
    public nint GetWindowValuesPtr(int layerIndex)
        => _valuesWindow != null ? _valuesWindow[layerIndex] : 0;

    // ── IKvCache interface (throw for unsupported host-side operations) ──

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("Use UpdateDevice() for GPU quantized cache.");

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("Use UpdateDevice() for GPU quantized cache.");

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
        => throw new NotSupportedException("Use PrepareAttentionScratch() for GPU quantized cache.");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
        => throw new NotSupportedException("Use PrepareAttentionScratch() for GPU quantized cache.");

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
        => new(WindowLength, _kvStride, DType.Float16, 0,
               _keysWindow != null ? _keysWindow[layerIndex] : 0);

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
        => new(WindowLength, _kvStride, DType.Float16, 0,
               _valuesWindow != null ? _valuesWindow[layerIndex] : 0);

    /// <inheritdoc/>
    public void Dispose()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            if (_keysQuant[i] != 0) { CudaDriverApi.cuMemFree_v2(_keysQuant[i]); _keysQuant[i] = 0; }
            if (_valuesQuant[i] != 0) { CudaDriverApi.cuMemFree_v2(_valuesQuant[i]); _valuesQuant[i] = 0; }

            if (_keysWindow != null && _keysWindow[i] != 0) { CudaDriverApi.cuMemFree_v2(_keysWindow[i]); _keysWindow[i] = 0; }
            if (_valuesWindow != null && _valuesWindow[i] != 0) { CudaDriverApi.cuMemFree_v2(_valuesWindow[i]); _valuesWindow[i] = 0; }
        }

        if (_kScratch != 0) { CudaDriverApi.cuMemFree_v2(_kScratch); _kScratch = 0; }
        if (_vScratch != 0) { CudaDriverApi.cuMemFree_v2(_vScratch); _vScratch = 0; }
    }

    private static int ComputeQuantRowBytes(int kvStride, KvCacheDType dtype) => dtype switch
    {
        KvCacheDType.F32 => kvStride * sizeof(ushort), // FP16 on GPU
        KvCacheDType.Q8_0 => kvStride / BlockSize * Q8_0BlockBytes,
        KvCacheDType.Q4_0 => kvStride / BlockSize * Q4_0BlockBytes,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype))
    };
}
