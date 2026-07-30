using DotLLM.Core.Tensors;
using DotLLM.Metal;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

public sealed class MetalKvCacheTests
{
    [MetalTestFact]
    public void Initialization_SetsCorrectProperties()
    {
        using var ctx = new MetalContext();
        int numLayers = 2;
        int numKvHeads = 4;
        int headDim = 64;
        int maxSeqLen = 128;

        using var cache = new MetalKvCache(ctx, numLayers, numKvHeads, headDim, maxSeqLen);

        Assert.Equal(0, cache.CurrentLength);
        Assert.Equal(maxSeqLen, cache.MaxLength);

        // FP16 = 2 bytes. 2 buffers per layer (K and V).
        long expectedBytes = (long)numLayers * 2 * maxSeqLen * (numKvHeads * headDim) * 2;
        Assert.Equal(expectedBytes, cache.AllocatedBytes);
    }

    [MetalTestFact]
    public unsafe void Update_IncreasesCurrentLength()
    {
        using var ctx = new MetalContext();
        int kvStride = 8;
        using var cache = new MetalKvCache(ctx, numLayers: 1, numKvHeads: 1, headDim: kvStride, maxSeqLen: 10);

        var s = sizeof(Half);

        // Add 3 new tokens at positions 0, 1, 2
        Half[] dummyData = new Half[3 * kvStride];
        for (int i = 0; i < dummyData.Length; i++) dummyData[i] = (Half)i;

        TensorRef keysBeforeUpdate = cache.GetKeysRef(0);
        Assert.Equal(0, keysBeforeUpdate.Dim0);
        Assert.Equal(kvStride, keysBeforeUpdate.Dim1);

        fixed (Half* ptr = dummyData)
        {
            var kRef = new TensorRef(3, kvStride, DType.Float16, -1, (nint)ptr);
            var vRef = new TensorRef(3, kvStride, DType.Float16, -1, (nint)ptr);

            cache.Update(kRef, vRef, [0, 1, 2], layerIndex: 0);
        }

        Assert.Equal(3, cache.CurrentLength);

        // Check that data are accessible from GetKeysRef
        TensorRef keys = cache.GetKeysRef(0);
        Assert.Equal(3, keys.Dim0);
        Assert.Equal(kvStride, keys.Dim1);

        Half* keysPtr = (Half*)keys.DataPointer;
        for (int i = 0; i < 3 * kvStride; i++)
        {
            Assert.Equal((Half)i, keysPtr[i]);
        }

        // Add 1 token at position 3
        Half[] singleToken = new Half[kvStride];
        for (int i = 0; i < kvStride; i++) singleToken[i] = (Half)(100 + i);

        fixed (Half* ptr = singleToken)
        {
            var kRef = new TensorRef(1, kvStride, DType.Float16, -1, (nint)ptr);
            var vRef = new TensorRef(1, kvStride, DType.Float16, -1, (nint)ptr);

            cache.Update(kRef, vRef, [3], layerIndex: 0);
        }

        Assert.Equal(4, cache.CurrentLength);

        // Check token
        keys = cache.GetKeysRef(0);
        keysPtr = (Half*)keys.DataPointer;
        for (int i = 0; i < kvStride; i++)
        {
            Assert.Equal((Half)(100 + i), keysPtr[3 * kvStride + i]);
        }
    }

    [MetalTestFact]
    public void Rollback_DecreasesCurrentLength()
    {
        using var ctx = new MetalContext();
        using var cache = new MetalKvCache(ctx, 1, 1, 8, 10);

        cache.SetCurrentLength(5);
        Assert.Equal(5, cache.CurrentLength);

        cache.Rollback(2);
        Assert.Equal(2, cache.CurrentLength);
    }

    [MetalTestFact]
    public void SetCurrentLength_Throws_WhenOutOfBounds()
    {
        using var ctx = new MetalContext();
        using var cache = new MetalKvCache(ctx, 1, 1, 8, 10);

        Assert.Throws<ArgumentOutOfRangeException>(() => cache.SetCurrentLength(11));
    }

    [MetalTestFact]
    public void Rollback_Throws_WhenIncreasingLength()
    {
        using var ctx = new MetalContext();
        using var cache = new MetalKvCache(ctx, 1, 1, 8, 10);

        cache.SetCurrentLength(5);
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Rollback(6));
    }
}
