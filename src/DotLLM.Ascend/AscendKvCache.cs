using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Ascend;

/// <summary>
/// Contiguous FP16 device KV storage for one sequence. Length commits after a
/// successful model forward; rollback retains allocation and hides discarded rows.
/// </summary>
public sealed class AscendKvCache : IKvCache
{
    internal object SyncRoot { get; } = new();
    internal AscendTransformerModel Owner { get; }
    private readonly AscendSession _session;
    private readonly AscendTensor[] _keys, _values;
    private readonly bool[] _written;
    private readonly int _stride;
    private int _length, _pendingRows;
    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength { get { lock (SyncRoot) { Check(); return _length; } } }
    /// <inheritdoc/>
    public int MaxLength { get; }
    /// <summary>Total allocated device storage in bytes.</summary>
    public long AllocatedBytes => checked((long)_keys.Length * 4 * MaxLength * _stride);

    internal AscendKvCache(AscendTransformerModel owner, AscendSession session, ModelConfig c, int maxLength)
    {
        if (maxLength <= 0 || maxLength > c.MaxSequenceLength)
            throw new ArgumentOutOfRangeException(nameof(maxLength));
        Owner = owner; _session = session; MaxLength = maxLength;
        _stride = checked(c.NumKvHeads * c.HeadDim);
        _keys = new AscendTensor[c.NumLayers]; _values = new AscendTensor[c.NumLayers];
        _written = new bool[c.NumLayers];
        try
        {
            for (int i = 0; i < c.NumLayers; i++)
            {
                _keys[i] = session.AllocateFloat16(new(maxLength, _stride));
                _values[i] = session.AllocateFloat16(new(maxLength, _stride));
            }
        }
        catch { Dispose(); throw; }
    }

    internal void Check()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _session.ThrowIfUnavailable();
    }

    internal void Write(AscendTensor keys, AscendTensor values, int offset, int layer)
    {
        _session.CopyFloat16(keys, 0, _keys[layer], (long)offset * _stride, keys.ElementCount);
        _session.CopyFloat16(values, 0, _values[layer], (long)offset * _stride, values.ElementCount);
    }

    internal AscendTensor Keys(int layer) => _keys[layer];
    internal AscendTensor Values(int layer) => _values[layer];
    internal void Commit(int length) { _length = length; Abort(); }
    internal void Abort() { _pendingRows = 0; Array.Clear(_written); }

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
    {
        lock (SyncRoot)
        {
            Check();
            if ((uint)layerIndex >= (uint)_keys.Length) throw new ArgumentOutOfRangeException(nameof(layerIndex));
            if (keys is not AscendTensor k || values is not AscendTensor v ||
                k.Session != _session || v.Session != _session || k.IsDisposed || v.IsDisposed ||
                k.StorageShape != new TensorShape(positions.Length, _stride) || v.StorageShape != k.StorageShape)
                throw new ArgumentException("KV updates require live FP16 tensors from the cache's Ascend session.");
            if (positions.IsEmpty || positions.Length > MaxLength - _length ||
                (_pendingRows != 0 && _pendingRows != positions.Length) || _written[layerIndex])
                throw new ArgumentException("Invalid KV append length or duplicate layer.");
            for (int i = 0; i < positions.Length; i++)
                if (positions[i] != _length + i) throw new ArgumentException("KV positions must append contiguously.");
            try
            {
                Write(k, v, _length, layerIndex);
                _pendingRows = positions.Length; _written[layerIndex] = true;
                if (_written.All(x => x)) Commit(_length + positions.Length);
            }
            catch { Abort(); throw; }
        }
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("TensorRef lacks Ascend session ownership; use owned AscendTensor updates.");

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex) => GetTensor(layerIndex, true);
    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex) => GetTensor(layerIndex, false);
    private ITensor GetTensor(int layer, bool key)
    {
        lock (SyncRoot)
        {
            Check();
            if ((uint)layer >= (uint)_keys.Length) throw new ArgumentOutOfRangeException(nameof(layer));
            var t = key ? _keys[layer] : _values[layer];
            return new TensorView(new(_length, _stride), DType.Float16, _session.DeviceId, t.DataPointer);
        }
    }
    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex) => GetRef(layerIndex, true);
    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex) => GetRef(layerIndex, false);
    private TensorRef GetRef(int layer, bool key)
    {
        lock (SyncRoot)
        {
            Check();
            if ((uint)layer >= (uint)_keys.Length) throw new ArgumentOutOfRangeException(nameof(layer));
            return new(_length, _stride, DType.Float16, _session.DeviceId,
                (key ? _keys[layer] : _values[layer]).DataPointer);
        }
    }

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        lock (SyncRoot)
        {
            Check();
            if ((uint)length > (uint)_length) throw new ArgumentOutOfRangeException(nameof(length));
            Commit(length);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (SyncRoot)
        {
            if (_disposed) return;
            foreach (var t in _keys) t?.Dispose();
            foreach (var t in _values) t?.Dispose();
            _disposed = true;
        }
    }
}
