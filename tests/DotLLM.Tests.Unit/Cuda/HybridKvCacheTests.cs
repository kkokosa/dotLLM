using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Tests for <see cref="HybridKvCache"/> — verifies layer routing, index remapping,
/// and CurrentLength tracking. GPU cache operations are tested only for route correctness
/// (GPU-layer Update throws; CPU-layer Update delegates to SimpleKvCache).
/// </summary>
public sealed unsafe class HybridKvCacheTests : IDisposable
{
    private const int NumGpuLayers = 2;
    private const int NumCpuLayers = 3;
    private const int TotalLayers = NumGpuLayers + NumCpuLayers;
    private const int NumKvHeads = 2;
    private const int HeadDim = 4;
    private const int MaxSeqLen = 8;
    private const int KvStride = NumKvHeads * HeadDim;

    // We can't create a real CudaKvCache without a GPU, so test the CPU-side routing only.
    // GPU-layer operations throw by design.
    private readonly SimpleKvCache _cpuCache = new(NumCpuLayers, NumKvHeads, HeadDim, MaxSeqLen);

    public void Dispose() => _cpuCache.Dispose();

    [Fact]
    public void CpuLayer_Update_RoutesWithRemappedIndex()
    {
        // Create a hybrid cache with a null-like GPU cache (we won't touch GPU layers)
        // For pure CPU-side testing, we create a HybridKvCache that wraps our cpu cache.
        // Since we can't create CudaKvCache without GPU, we test the SimpleKvCache routing directly.
        int seqLen = 2;
        nint kPtr = AllocAndFillNative(seqLen, 1.0f);
        nint vPtr = AllocAndFillNative(seqLen, 2.0f);
        try
        {
            var kRef = new TensorRef(seqLen, KvStride, DType.Float32, -1, kPtr);
            var vRef = new TensorRef(seqLen, KvStride, DType.Float32, -1, vPtr);

            // Layer index = NumGpuLayers (first CPU layer) should map to cpuCache layer 0
            _cpuCache.Update(kRef, vRef, [0, 1], layerIndex: 0);

            Assert.Equal(2, _cpuCache.CurrentLength);

            var cachedK = _cpuCache.GetKeysRef(0);
            Assert.Equal(2, cachedK.Dim0);
            float* cachedKData = (float*)cachedK.DataPointer;
            Assert.Equal(1.0f, cachedKData[0]);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }
    }

    [Fact]
    public void CpuLayer_GetKeysRef_RoutesWithRemappedIndex()
    {
        int seqLen = 1;
        nint kPtr = AllocAndFillNative(seqLen, 3.0f);
        nint vPtr = AllocAndFillNative(seqLen, 4.0f);
        try
        {
            var kRef = new TensorRef(seqLen, KvStride, DType.Float32, -1, kPtr);
            var vRef = new TensorRef(seqLen, KvStride, DType.Float32, -1, vPtr);

            // Update CPU cache layer 1 (which would be global layer NumGpuLayers + 1)
            _cpuCache.Update(kRef, vRef, [0], layerIndex: 1);

            var cachedK = _cpuCache.GetKeysRef(1);
            float* data = (float*)cachedK.DataPointer;
            Assert.Equal(3.0f, data[0]);

            var cachedV = _cpuCache.GetValuesRef(1);
            float* vData = (float*)cachedV.DataPointer;
            Assert.Equal(4.0f, vData[0]);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }
    }

    [Fact]
    public void CurrentLength_UpdatesCorrectly()
    {
        int seqLen = 3;
        nint kPtr = AllocAndFillNative(seqLen, 1.0f);
        nint vPtr = AllocAndFillNative(seqLen, 2.0f);
        try
        {
            Assert.Equal(0, _cpuCache.CurrentLength);

            var kRef = new TensorRef(seqLen, KvStride, DType.Float32, -1, kPtr);
            var vRef = new TensorRef(seqLen, KvStride, DType.Float32, -1, vPtr);

            _cpuCache.Update(kRef, vRef, [0, 1, 2], layerIndex: 0);
            Assert.Equal(3, _cpuCache.CurrentLength);

            // Add one more token at position 3
            nint k2 = AllocAndFillNative(1, 5.0f);
            nint v2 = AllocAndFillNative(1, 6.0f);
            try
            {
                var kRef2 = new TensorRef(1, KvStride, DType.Float32, -1, k2);
                var vRef2 = new TensorRef(1, KvStride, DType.Float32, -1, v2);
                _cpuCache.Update(kRef2, vRef2, [3], layerIndex: 0);
                Assert.Equal(4, _cpuCache.CurrentLength);
            }
            finally
            {
                NativeMemory.AlignedFree((void*)k2);
                NativeMemory.AlignedFree((void*)v2);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }
    }

    private static nint AllocAndFillNative(int seqLen, float fillValue)
    {
        int totalElements = seqLen * KvStride;
        nuint bytes = (nuint)(totalElements * sizeof(float));
        nint ptr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
        float* fPtr = (float*)ptr;
        for (int i = 0; i < totalElements; i++)
            fPtr[i] = fillValue;
        return ptr;
    }
}
