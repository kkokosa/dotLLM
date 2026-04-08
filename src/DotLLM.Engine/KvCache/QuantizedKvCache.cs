using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;

namespace DotLLM.Engine.KvCache;

/// <summary>
/// KV-cache with quantized storage (Q8_0 or Q4_0) and an optional full-precision window
/// for recent tokens. Uses a dual-region design:
/// <list type="bullet">
/// <item><b>Quantized buffer</b>: older tokens stored in Q8_0/Q4_0 format (append-only).</item>
/// <item><b>Full-precision ring buffer</b>: recent W tokens in FP32 (quantize-on-evict).</item>
/// </list>
/// When <c>windowSize == 0</c>, all tokens are quantized immediately (no ring buffer).
/// </summary>
public sealed unsafe class QuantizedKvCache : IQuantizedKvCache
{
    private const int BlockSize = 32;
    private const int Q8_0BlockBytes = 34;
    private const int Q4_0BlockBytes = 18;

    private readonly nint[] _keysQuant;     // [numLayers] quantized K buffers
    private readonly nint[] _valuesQuant;   // [numLayers] quantized V buffers
    private readonly nint[]? _keysWindow;   // [numLayers] FP32 ring buffers (null when windowSize=0)
    private readonly nint[]? _valuesWindow; // [numLayers] FP32 ring buffers (null when windowSize=0)
    private readonly int _numLayers;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _maxSeqLen;
    private readonly int _kvStride;          // numKvHeads * headDim
    private readonly int _windowSize;
    private readonly int _keyQuantRowBytes;
    private readonly int _valueQuantRowBytes;
    private readonly int[] _layerQuantizedLength; // per-layer eviction tracking
    private int _currentLength;
    private int _quantizedLength;
    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <inheritdoc/>
    public int QuantizedLength => _quantizedLength;

    /// <inheritdoc/>
    public int WindowLength => _windowSize > 0
        ? Math.Min(_currentLength, _windowSize)
        : 0;

    /// <inheritdoc/>
    public int WindowCapacity => _windowSize;

    /// <inheritdoc/>
    public KvCacheDType KeyDType { get; }

    /// <inheritdoc/>
    public KvCacheDType ValueDType { get; }

    /// <inheritdoc/>
    public int KeyQuantizedRowBytes => _keyQuantRowBytes;

    /// <inheritdoc/>
    public int ValueQuantizedRowBytes => _valueQuantRowBytes;

    /// <summary>Total bytes allocated for all KV-cache buffers (quantized + window).</summary>
    public long AllocatedBytes { get; }

    /// <summary>
    /// Creates a new quantized KV-cache.
    /// </summary>
    public QuantizedKvCache(int numLayers, int numKvHeads, int headDim, int maxSeqLen,
                             KvCacheDType keyDType, KvCacheDType valueDType, int windowSize)
    {
        if (keyDType == KvCacheDType.F32 && valueDType == KvCacheDType.F32)
            throw new ArgumentException("At least one of keyDType/valueDType must be quantized. Use SimpleKvCache for full precision.");
        if (windowSize < 0)
            throw new ArgumentOutOfRangeException(nameof(windowSize), windowSize, "Window size must be >= 0.");

        _numLayers = numLayers;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _maxSeqLen = maxSeqLen;
        _kvStride = numKvHeads * headDim;
        _windowSize = windowSize;
        KeyDType = keyDType;
        ValueDType = valueDType;

        if (_kvStride % BlockSize != 0)
            throw new ArgumentException(
                $"kvStride ({_kvStride}) must be a multiple of {BlockSize} for quantization.",
                nameof(headDim));
        System.Diagnostics.Debug.Assert(_kvStride % BlockSize == 0,
            $"kvStride ({_kvStride}) must be a multiple of {BlockSize}");

        _keyQuantRowBytes = ComputeQuantRowBytes(_kvStride, keyDType);
        _valueQuantRowBytes = ComputeQuantRowBytes(_kvStride, valueDType);
        _layerQuantizedLength = new int[numLayers];

        _keysQuant = new nint[numLayers];
        _valuesQuant = new nint[numLayers];

        long totalBytes = 0;

        // Allocate quantized buffers (sized for maxSeqLen — worst case all tokens quantized)
        for (int i = 0; i < numLayers; i++)
        {
            nuint kQuantBytes = (nuint)((long)maxSeqLen * _keyQuantRowBytes);
            nuint vQuantBytes = (nuint)((long)maxSeqLen * _valueQuantRowBytes);
            _keysQuant[i] = (nint)NativeMemory.AlignedAlloc(kQuantBytes, 64);
            _valuesQuant[i] = (nint)NativeMemory.AlignedAlloc(vQuantBytes, 64);
            totalBytes += (long)(kQuantBytes + vQuantBytes);
        }

        // Allocate window buffers (FP32)
        if (windowSize > 0)
        {
            _keysWindow = new nint[numLayers];
            _valuesWindow = new nint[numLayers];
            nuint windowBytes = (nuint)((long)windowSize * _kvStride * sizeof(float));
            for (int i = 0; i < numLayers; i++)
            {
                _keysWindow[i] = (nint)NativeMemory.AlignedAlloc(windowBytes, 64);
                _valuesWindow[i] = (nint)NativeMemory.AlignedAlloc(windowBytes, 64);
                totalBytes += (long)(windowBytes * 2);
            }
        }

        AllocatedBytes = totalBytes;
    }

    /// <summary>Releases unmanaged buffers if <see cref="Dispose()"/> was not called.</summary>
    ~QuantizedKvCache() => Dispose(disposing: false);

    /// <inheritdoc/>
    [SkipLocalsInit]
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        int seqLen = positions.Length;
        float* kSrc = (float*)keys.DataPointer;
        float* vSrc = (float*)values.DataPointer;
        int fpRowBytes = _kvStride * sizeof(float);

        // Compute new sequence length (idempotent across layer calls with same positions).
        int maxPos = positions[0];
        for (int i = 1; i < seqLen; i++)
            if (positions[i] > maxPos) maxPos = positions[i];
        int newLength = maxPos + 1;

        if (_windowSize > 0)
        {
            // Per-layer eviction: each layer independently tracks how far it has evicted.
            int prevQuantLen = _layerQuantizedLength[layerIndex];
            int newQuantLen = Math.Max(0, newLength - _windowSize);

            for (int evictPos = prevQuantLen; evictPos < newQuantLen; evictPos++)
            {
                int ringIdx = evictPos % _windowSize;
                QuantizeRow((float*)_keysWindow![layerIndex] + ringIdx * _kvStride,
                            (byte*)_keysQuant[layerIndex] + (long)evictPos * _keyQuantRowBytes,
                            _kvStride, KeyDType);
                QuantizeRow((float*)_valuesWindow![layerIndex] + ringIdx * _kvStride,
                            (byte*)_valuesQuant[layerIndex] + (long)evictPos * _valueQuantRowBytes,
                            _kvStride, ValueDType);
            }

            _layerQuantizedLength[layerIndex] = newQuantLen;

            // Write new FP32 data into window ring buffer (position-addressed, idempotent).
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                if ((uint)pos >= (uint)_maxSeqLen)
                    throw new ArgumentOutOfRangeException(nameof(positions),
                        $"Position {pos} exceeds max cache length {_maxSeqLen}.");

                int ringIdx = pos % _windowSize;
                Buffer.MemoryCopy(
                    kSrc + i * _kvStride,
                    (float*)_keysWindow![layerIndex] + ringIdx * _kvStride,
                    fpRowBytes, fpRowBytes);
                Buffer.MemoryCopy(
                    vSrc + i * _kvStride,
                    (float*)_valuesWindow![layerIndex] + ringIdx * _kvStride,
                    fpRowBytes, fpRowBytes);
            }

            _quantizedLength = newQuantLen;
        }
        else
        {
            // Pure quantized mode: quantize directly at each position (position-addressed).
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                if ((uint)pos >= (uint)_maxSeqLen)
                    throw new ArgumentOutOfRangeException(nameof(positions),
                        $"Position {pos} exceeds max cache length {_maxSeqLen}.");

                QuantizeRow(kSrc + i * _kvStride,
                            (byte*)_keysQuant[layerIndex] + (long)pos * _keyQuantRowBytes,
                            _kvStride, KeyDType);
                QuantizeRow(vSrc + i * _kvStride,
                            (byte*)_valuesQuant[layerIndex] + (long)pos * _valueQuantRowBytes,
                            _kvStride, ValueDType);
            }

            _quantizedLength = newLength;
        }

        _currentLength = newLength;
    }

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
    {
        var kRef = new TensorRef(positions.Length, _kvStride, keys.DType, keys.DeviceId, keys.DataPointer);
        var vRef = new TensorRef(positions.Length, _kvStride, values.DType, values.DeviceId, values.DataPointer);
        Update(kRef, vRef, positions, layerIndex);
    }

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public nint GetQuantizedKeysPtr(int layerIndex) => _keysQuant[layerIndex];

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public nint GetQuantizedValuesPtr(int layerIndex) => _valuesQuant[layerIndex];

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public nint GetWindowKeysPtr(int layerIndex)
        => _keysWindow != null ? _keysWindow[layerIndex] : 0;

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public nint GetWindowValuesPtr(int layerIndex)
        => _valuesWindow != null ? _valuesWindow[layerIndex] : 0;

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef GetKeysRef(int layerIndex)
    {
        // Return window data if available, otherwise empty ref.
        if (_keysWindow != null && WindowLength > 0)
            return new TensorRef(WindowLength, _kvStride, DType.Float32, -1, _keysWindow[layerIndex]);
        return default;
    }

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef GetValuesRef(int layerIndex)
    {
        if (_valuesWindow != null && WindowLength > 0)
            return new TensorRef(WindowLength, _kvStride, DType.Float32, -1, _valuesWindow[layerIndex]);
        return default;
    }

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
        if (_quantizedLength > length)
            _quantizedLength = length;
    }

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex) =>
        throw new NotSupportedException("Use IQuantizedKvCache methods for quantized cache access.");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex) =>
        throw new NotSupportedException("Use IQuantizedKvCache methods for quantized cache access.");

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    private void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;

        for (int i = 0; i < _numLayers; i++)
        {
            if (_keysQuant[i] != 0) { NativeMemory.AlignedFree((void*)_keysQuant[i]); _keysQuant[i] = 0; }
            if (_valuesQuant[i] != 0) { NativeMemory.AlignedFree((void*)_valuesQuant[i]); _valuesQuant[i] = 0; }

            if (_keysWindow != null && _keysWindow[i] != 0) { NativeMemory.AlignedFree((void*)_keysWindow[i]); _keysWindow[i] = 0; }
            if (_valuesWindow != null && _valuesWindow[i] != 0) { NativeMemory.AlignedFree((void*)_valuesWindow[i]); _valuesWindow[i] = 0; }
        }
    }

    // ──────────────────── Inline scalar quantization ────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ComputeQuantRowBytes(int kvStride, KvCacheDType dtype) => dtype switch
    {
        KvCacheDType.F32 => kvStride * sizeof(float), // should not happen, but safe
        KvCacheDType.Q8_0 => kvStride / BlockSize * Q8_0BlockBytes,
        KvCacheDType.Q4_0 => kvStride / BlockSize * Q4_0BlockBytes,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype))
    };

    /// <summary>
    /// Quantizes a single FP32 row to Q8_0 or Q4_0. Simple scalar implementation
    /// since this runs once per eviction (not on the hot attention path).
    /// </summary>
    [SkipLocalsInit]
    private static void QuantizeRow(float* src, byte* dest, int elementCount, KvCacheDType dtype)
    {
        if (dtype == KvCacheDType.F32) return; // no-op
        if (dtype == KvCacheDType.Q8_0)
            QuantizeRowQ8_0(src, dest, elementCount);
        else
            QuantizeRowQ4_0(src, dest, elementCount);
    }

    [SkipLocalsInit]
    private static void QuantizeRowQ8_0(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;
        for (int block = 0; block < blockCount; block++)
        {
            float* bs = src + block * BlockSize;
            byte* bd = dest + block * Q8_0BlockBytes;

            float maxAbs = 0;
            for (int i = 0; i < BlockSize; i++)
            {
                float abs = MathF.Abs(bs[i]);
                if (abs > maxAbs) maxAbs = abs;
            }

            float scale = maxAbs / 127.0f;
            Unsafe.WriteUnaligned(bd, (Half)scale);
            sbyte* qs = (sbyte*)(bd + 2);

            if (scale == 0)
            {
                for (int i = 0; i < BlockSize; i++) qs[i] = 0;
            }
            else
            {
                float invScale = 1.0f / scale;
                for (int i = 0; i < BlockSize; i++)
                    qs[i] = (sbyte)Math.Clamp((int)MathF.Round(bs[i] * invScale), -127, 127);
            }
        }
    }

    [SkipLocalsInit]
    private static void QuantizeRowQ4_0(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;
        for (int block = 0; block < blockCount; block++)
        {
            float* bs = src + block * BlockSize;
            byte* bd = dest + block * Q4_0BlockBytes;

            float maxAbs = 0;
            for (int i = 0; i < BlockSize; i++)
            {
                float abs = MathF.Abs(bs[i]);
                if (abs > maxAbs) maxAbs = abs;
            }

            float d = maxAbs / 7.0f;
            Unsafe.WriteUnaligned(bd, (Half)d);
            byte* qs = bd + 2;

            if (d == 0)
            {
                for (int j = 0; j < 16; j++) qs[j] = 0x88;
            }
            else
            {
                float invD = 1.0f / d;
                for (int j = 0; j < 16; j++)
                {
                    int lo = Math.Clamp((int)MathF.Round(bs[2 * j] * invD) + 8, 0, 15);
                    int hi = Math.Clamp((int)MathF.Round(bs[2 * j + 1] * invD) + 8, 0, 15);
                    qs[j] = (byte)((hi << 4) | lo);
                }
            }
        }
    }
}
