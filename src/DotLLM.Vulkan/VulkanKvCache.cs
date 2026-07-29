using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Vulkan-side KV cache. Per-layer device-local buffers of shape
/// <c>[maxSeqLen, numKvHeads * headDim]</c> FP32. The host never touches
/// cached K/V — updates are recorded as <c>vkCmdCopyBuffer</c> from the
/// host-visible activation buffers to the device-local cache, either
/// synchronously (legacy <see cref="UpdateDevice"/>) or appended to a caller-
/// supplied command buffer (<see cref="RecordUpdate"/>, used by the
/// fence-pipelined forward pass).
/// </summary>
/// <remarks>
/// <para>
/// Mirrors <c>DotLLM.Engine.KvCache.SimpleKvCache</c> semantics: <c>Update</c>
/// appends new K/V rows at the supplied position indices. The attention kernel
/// reads straight from the device buffers via <see cref="GetKeysBuffer"/> /
/// <see cref="GetValuesBuffer"/>; no staging copies are required.
/// </para>
/// <para>
/// Implements <see cref="IKvCache"/> so code that already knows about the CPU
/// cache semantics (text-generation loop, tests) can swap the backing store
/// transparently. The <see cref="IKvCache.Update(ITensor, ITensor, ReadOnlySpan{int}, int)"/>
/// and <see cref="IKvCache.Update(TensorRef, TensorRef, ReadOnlySpan{int}, int)"/>
/// overloads expect CPU-resident tensor pointers (the caller is responsible
/// for uploading); we only use the device-side path from
/// <see cref="VulkanTransformerModel"/>, but the IKvCache contract lets this
/// object satisfy the same API.
/// </para>
/// </remarks>
public sealed class VulkanKvCache : IKvCache
{
    private readonly VulkanDevice _device;
    private readonly VulkanDevice.Buffer[] _keys;
    private readonly VulkanDevice.Buffer[] _values;
    private readonly int _numLayers;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _maxSeqLen;
    private readonly int _kvStride;
    private int _currentLength;
    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>Creates the per-layer K/V buffers. Memory is not zeroed — the forward pass only reads positions it has written.</summary>
    public VulkanKvCache(VulkanDevice device, int numLayers, int numKvHeads, int headDim, int maxSeqLen)
    {
        ArgumentNullException.ThrowIfNull(device);
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));

        _device = device;
        _numLayers = numLayers;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _maxSeqLen = maxSeqLen;
        _kvStride = numKvHeads * headDim;

        _keys = new VulkanDevice.Buffer[numLayers];
        _values = new VulkanDevice.Buffer[numLayers];

        long bytesPerLayer = (long)maxSeqLen * _kvStride * sizeof(float);
        for (int i = 0; i < numLayers; i++)
        {
            _keys[i] = device.AllocateDeviceLocal(bytesPerLayer);
            _values[i] = device.AllocateDeviceLocal(bytesPerLayer);
        }
    }

    /// <summary>Returns the device buffer holding cached keys for the given layer.</summary>
    internal VulkanDevice.Buffer GetKeysBuffer(int layerIndex) => _keys[layerIndex];

    /// <summary>Returns the device buffer holding cached values for the given layer.</summary>
    internal VulkanDevice.Buffer GetValuesBuffer(int layerIndex) => _values[layerIndex];

    /// <summary>
    /// Copies new <paramref name="kDev"/> / <paramref name="vDev"/> rows into
    /// the device-local cached K/V buffers at the given positions. Source
    /// buffers are the current forward pass's host-visible K/V activations;
    /// the cache destination is device-local (VRAM on a dGPU, driver-tiled on
    /// UMA). Issues a synchronous <c>vkCmdCopyBuffer</c> + fence wait.
    /// Prefer <see cref="RecordUpdate"/> from the fence-pipelined forward pass.
    /// </summary>
    internal void UpdateDevice(
        VulkanDevice.Buffer kDev, VulkanDevice.Buffer vDev,
        ReadOnlySpan<int> positions, int seqLen, int layerIndex)
    {
        if (positions.Length != seqLen)
            throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));

        int rowBytes = _kvStride * sizeof(float);

        // Single contiguous range if positions are consecutive — one copy call
        // covers the whole seqLen. Otherwise fall back to per-row copies.
        int maxPos = ValidateAndFindMaxPos(positions, seqLen);
        bool contiguous = IsContiguousAscending(positions);

        if (contiguous)
        {
            int startPos = positions[0];
            ulong totalBytes = (ulong)rowBytes * (ulong)seqLen;
            _device.CopyBufferRangeSynchronous(kDev, _keys[layerIndex],
                srcOffset: 0, dstOffset: (ulong)((long)startPos * rowBytes), size: totalBytes);
            _device.CopyBufferRangeSynchronous(vDev, _values[layerIndex],
                srcOffset: 0, dstOffset: (ulong)((long)startPos * rowBytes), size: totalBytes);
        }
        else
        {
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                _device.CopyBufferRangeSynchronous(kDev, _keys[layerIndex],
                    srcOffset: (ulong)((long)i * rowBytes),
                    dstOffset: (ulong)((long)pos * rowBytes),
                    size: (ulong)rowBytes);
                _device.CopyBufferRangeSynchronous(vDev, _values[layerIndex],
                    srcOffset: (ulong)((long)i * rowBytes),
                    dstOffset: (ulong)((long)pos * rowBytes),
                    size: (ulong)rowBytes);
            }
        }

        int newLength = maxPos + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
    }

    /// <summary>
    /// Appends K/V copy commands onto the supplied <paramref name="cmdBuf"/>.
    /// The caller is responsible for the <c>TRANSFER → COMPUTE_SHADER</c>
    /// barrier that follows (so the attention kernel reads the freshly
    /// written cache rows), and for advancing <see cref="CurrentLength"/>
    /// after the batch commits.
    /// </summary>
    internal unsafe void RecordUpdate(
        nint cmdBuf,
        VulkanDevice.Buffer kDev, VulkanDevice.Buffer vDev,
        ReadOnlySpan<int> positions, int seqLen, int layerIndex)
    {
        if (positions.Length != seqLen)
            throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));

        int rowBytes = _kvStride * sizeof(float);
        int maxPos = ValidateAndFindMaxPos(positions, seqLen);
        bool contiguous = IsContiguousAscending(positions);

        if (contiguous)
        {
            int startPos = positions[0];
            var region = new VkBufferCopy
            {
                srcOffset = 0,
                dstOffset = (ulong)((long)startPos * rowBytes),
                size = (ulong)rowBytes * (ulong)seqLen,
            };
            VulkanApi.vkCmdCopyBuffer(cmdBuf, kDev.Handle, _keys[layerIndex].Handle, 1, region);
            VulkanApi.vkCmdCopyBuffer(cmdBuf, vDev.Handle, _values[layerIndex].Handle, 1, region);
        }
        else
        {
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                var region = new VkBufferCopy
                {
                    srcOffset = (ulong)((long)i * rowBytes),
                    dstOffset = (ulong)((long)pos * rowBytes),
                    size = (ulong)rowBytes,
                };
                VulkanApi.vkCmdCopyBuffer(cmdBuf, kDev.Handle, _keys[layerIndex].Handle, 1, region);
                VulkanApi.vkCmdCopyBuffer(cmdBuf, vDev.Handle, _values[layerIndex].Handle, 1, region);
            }
        }

        int newLength = maxPos + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
    }

    private int ValidateAndFindMaxPos(ReadOnlySpan<int> positions, int seqLen)
    {
        int maxPos = -1;
        for (int i = 0; i < seqLen; i++)
        {
            int pos = positions[i];
            if ((uint)pos >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {pos} exceeds max cache length {_maxSeqLen}.");
            if (pos > maxPos) maxPos = pos;
        }
        return maxPos;
    }

    private static bool IsContiguousAscending(ReadOnlySpan<int> positions)
    {
        for (int i = 1; i < positions.Length; i++)
        {
            if (positions[i] != positions[i - 1] + 1)
                return false;
        }
        return true;
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "VulkanKvCache is updated via UpdateDevice from the Vulkan forward pass; the host-side Update overload is not supported.");

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "VulkanKvCache is updated via UpdateDevice from the Vulkan forward pass; the host-side Update overload is not supported.");

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
        => throw new NotSupportedException("VulkanKvCache exposes device buffers via GetKeysBuffer, not TensorRef.");

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
        => throw new NotSupportedException("VulkanKvCache exposes device buffers via GetValuesBuffer, not TensorRef.");

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
        => throw new NotSupportedException("VulkanKvCache does not materialise cached keys as CPU tensors.");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
        => throw new NotSupportedException("VulkanKvCache does not materialise cached values as CPU tensors.");

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <summary>Resets the visible length. Used when starting a new sequence.</summary>
    public void Reset() => _currentLength = 0;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _numLayers; i++)
        {
            _keys[i]?.Dispose();
            _values[i]?.Dispose();
        }
    }
}
