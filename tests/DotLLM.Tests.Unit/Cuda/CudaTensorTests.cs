using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Tests for GPU tensor allocation and host↔device memory copies.
/// </summary>
[Trait("Category", "GPU")]
public class CudaTensorTests : IDisposable
{
    private readonly CudaContext? _ctx;

    public CudaTensorTests()
    {
        if (CudaDevice.IsAvailable())
            _ctx = CudaContext.Create(0);
    }

    [SkippableFact]
    public void AllocateTensor_Succeeds()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        var shape = new TensorShape(16, 32);
        using var tensor = CudaTensor.Allocate(shape, DType.Float16, 0);

        Assert.Equal(16 * 32, tensor.ElementCount);
        Assert.Equal(16 * 32 * 2, tensor.ByteCount); // FP16 = 2 bytes
        Assert.Equal(0, tensor.DeviceId);
        Assert.NotEqual(0, tensor.DataPointer);
    }

    [SkippableFact]
    public unsafe void RoundTrip_HostToDeviceToHost_PreservesData()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        // Create test data on host
        int n = 256;
        float[] hostSrc = new float[n];
        for (int i = 0; i < n; i++)
            hostSrc[i] = i * 0.1f;

        // Allocate on device
        long bytes = (long)n * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)bytes).ThrowOnError();

        try
        {
            // H2D
            fixed (float* srcPtr = hostSrc)
                CudaDriverApi.cuMemcpyHtoD_v2(devPtr, (nint)srcPtr, (nuint)bytes).ThrowOnError();

            // D2H into a new array
            float[] hostDst = new float[n];
            fixed (float* dstPtr = hostDst)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)dstPtr, devPtr, (nuint)bytes).ThrowOnError();

            // Verify
            for (int i = 0; i < n; i++)
                Assert.Equal(hostSrc[i], hostDst[i]);
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devPtr);
        }
    }

    public void Dispose()
    {
        _ctx?.Dispose();
    }
}
