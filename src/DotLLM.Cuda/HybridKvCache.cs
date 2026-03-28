using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;

namespace DotLLM.Cuda;

/// <summary>
/// Split KV-cache for hybrid CPU/GPU inference. Routes layers 0..N-1 to a GPU-resident
/// <see cref="CudaKvCache"/> and layers N..L-1 to a CPU-resident <see cref="SimpleKvCache"/>.
/// GPU layers are updated via <see cref="GpuCache"/> directly (device-side copies);
/// CPU layers use the standard <see cref="IKvCache"/> interface with remapped layer indices.
/// </summary>
public sealed class HybridKvCache : IKvCache
{
    private readonly int _numGpuLayers;

    /// <summary>GPU-side KV-cache for layers 0..N-1 (FP16, device memory).</summary>
    internal CudaKvCache GpuCache { get; }

    /// <summary>CPU-side KV-cache for layers N..L-1 (FP32, host memory).</summary>
    internal SimpleKvCache CpuCache { get; }

    /// <inheritdoc/>
    public int CurrentLength
    {
        get
        {
            Debug.Assert(GpuCache.CurrentLength == CpuCache.CurrentLength,
                "GPU and CPU KV-caches must advance in lockstep.");
            return CpuCache.CurrentLength;
        }
    }

    /// <inheritdoc/>
    public int MaxLength => CpuCache.MaxLength;

    /// <summary>
    /// Creates a hybrid KV-cache splitting layers between GPU and CPU.
    /// </summary>
    /// <param name="gpuCache">GPU cache for the first <paramref name="numGpuLayers"/> layers.</param>
    /// <param name="cpuCache">CPU cache for the remaining layers.</param>
    /// <param name="numGpuLayers">Number of layers handled by the GPU cache.</param>
    public HybridKvCache(CudaKvCache gpuCache, SimpleKvCache cpuCache, int numGpuLayers)
    {
        GpuCache = gpuCache;
        CpuCache = cpuCache;
        _numGpuLayers = numGpuLayers;
    }

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
    {
        Debug.Assert(layerIndex >= _numGpuLayers,
            $"Layer {layerIndex} is a GPU layer — should use GpuCache.UpdateDevice().");
        if (layerIndex < _numGpuLayers)
            throw new InvalidOperationException(
                $"Layer {layerIndex} is a GPU layer — use GpuCache.UpdateDevice() instead of IKvCache.Update().");

        CpuCache.Update(keys, values, positions, layerIndex - _numGpuLayers);
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        Debug.Assert(layerIndex >= _numGpuLayers,
            $"Layer {layerIndex} is a GPU layer — should use GpuCache.UpdateDevice().");
        if (layerIndex < _numGpuLayers)
            throw new InvalidOperationException(
                $"Layer {layerIndex} is a GPU layer — use GpuCache.UpdateDevice() instead of IKvCache.Update().");

        CpuCache.Update(keys, values, positions, layerIndex - _numGpuLayers);
    }

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
    {
        Debug.Assert(layerIndex >= _numGpuLayers,
            $"Layer {layerIndex} is a GPU layer — should use GpuCache.GetKeysPtr().");
        if (layerIndex < _numGpuLayers)
            throw new InvalidOperationException(
                $"Layer {layerIndex} is a GPU layer — use GpuCache.GetKeysPtr() instead.");

        return CpuCache.GetKeys(layerIndex - _numGpuLayers);
    }

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
    {
        Debug.Assert(layerIndex >= _numGpuLayers,
            $"Layer {layerIndex} is a GPU layer — should use GpuCache.GetValuesPtr().");
        if (layerIndex < _numGpuLayers)
            throw new InvalidOperationException(
                $"Layer {layerIndex} is a GPU layer — use GpuCache.GetValuesPtr() instead.");

        return CpuCache.GetValues(layerIndex - _numGpuLayers);
    }

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
    {
        if (layerIndex < _numGpuLayers)
            return GpuCache.GetKeysRef(layerIndex);

        return CpuCache.GetKeysRef(layerIndex - _numGpuLayers);
    }

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
    {
        if (layerIndex < _numGpuLayers)
            return GpuCache.GetValuesRef(layerIndex);

        return CpuCache.GetValuesRef(layerIndex - _numGpuLayers);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        GpuCache.Dispose();
        CpuCache.Dispose();
    }
}
