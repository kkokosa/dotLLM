using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Tests individual CUDA kernels against CPU reference implementations.
/// </summary>
[Trait("Category", "GPU")]
public class CudaKernelTests : IDisposable
{
    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaKernels? _kernels;

    public CudaKernelTests()
    {
        if (!CudaDevice.IsAvailable()) return;

        _ctx = CudaContext.Create(0);
        _stream = CudaStream.Create();

        // Find PTX directory
        string ptxDir = FindPtxDir();
        if (ptxDir != null)
            _kernels = new CudaKernels(ptxDir);
    }

    private static string? FindPtxDir()
    {
        // Try relative to test assembly, then relative to repo root
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };

        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }

    [SkippableFact]
    public unsafe void Add_MatchesCpuReference()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");

        int n = 128;
        nint s = _stream!.Handle;

        // Generate random FP16 data on host
        var rng = new Random(42);
        ushort[] aHost = new ushort[n];
        ushort[] bHost = new ushort[n];
        for (int i = 0; i < n; i++)
        {
            aHost[i] = BitConverter.HalfToUInt16Bits((Half)(rng.NextSingle() * 2 - 1));
            bHost[i] = BitConverter.HalfToUInt16Bits((Half)(rng.NextSingle() * 2 - 1));
        }

        long bytes = (long)n * sizeof(ushort);

        // Allocate device memory
        CudaDriverApi.cuMemAlloc_v2(out nint devA, (nuint)bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devB, (nuint)bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devC, (nuint)bytes).ThrowOnError();

        try
        {
            // Upload
            fixed (ushort* pA = aHost) CudaDriverApi.cuMemcpyHtoD_v2(devA, (nint)pA, (nuint)bytes).ThrowOnError();
            fixed (ushort* pB = bHost) CudaDriverApi.cuMemcpyHtoD_v2(devB, (nint)pB, (nuint)bytes).ThrowOnError();

            // Launch kernel
            _kernels!.LaunchAdd(devA, devB, devC, n, s);
            _stream!.Synchronize();

            // Download result
            ushort[] cHost = new ushort[n];
            fixed (ushort* pC = cHost) CudaDriverApi.cuMemcpyDtoH_v2((nint)pC, devC, (nuint)bytes).ThrowOnError();

            // Compare with CPU reference
            for (int i = 0; i < n; i++)
            {
                float expected = (float)BitConverter.UInt16BitsToHalf(aHost[i]) +
                                 (float)BitConverter.UInt16BitsToHalf(bHost[i]);
                float actual = (float)BitConverter.UInt16BitsToHalf(cHost[i]);
                Assert.True(MathF.Abs(expected - actual) < 0.01f,
                    $"Mismatch at {i}: expected {expected}, got {actual}");
            }
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devA);
            CudaDriverApi.cuMemFree_v2(devB);
            CudaDriverApi.cuMemFree_v2(devC);
        }
    }

    [SkippableFact]
    public unsafe void ConvertF32ToF16_RoundTrip()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");

        int n = 64;
        nint s = _stream!.Handle;

        // Source F32 data
        float[] srcHost = new float[n];
        for (int i = 0; i < n; i++)
            srcHost[i] = (i - 32) * 0.5f;

        long f32Bytes = (long)n * sizeof(float);
        long f16Bytes = (long)n * sizeof(ushort);

        CudaDriverApi.cuMemAlloc_v2(out nint devF32, (nuint)f32Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devF16, (nuint)f16Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devF32Back, (nuint)f32Bytes).ThrowOnError();

        try
        {
            // Upload F32
            fixed (float* p = srcHost) CudaDriverApi.cuMemcpyHtoD_v2(devF32, (nint)p, (nuint)f32Bytes).ThrowOnError();

            // F32 → F16
            _kernels!.LaunchConvertF32ToF16(devF32, devF16, n, s);

            // F16 → F32
            _kernels!.LaunchConvertF16ToF32(devF16, devF32Back, n, s);
            _stream!.Synchronize();

            // Download
            float[] dstHost = new float[n];
            fixed (float* p = dstHost) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devF32Back, (nuint)f32Bytes).ThrowOnError();

            // Compare (FP16 round-trip loses precision)
            for (int i = 0; i < n; i++)
            {
                float expected = (float)(Half)srcHost[i]; // simulate FP16 round-trip
                Assert.True(MathF.Abs(expected - dstHost[i]) < 0.001f,
                    $"Mismatch at {i}: expected {expected}, got {dstHost[i]}");
            }
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devF32);
            CudaDriverApi.cuMemFree_v2(devF16);
            CudaDriverApi.cuMemFree_v2(devF32Back);
        }
    }

    [SkippableFact]
    public void LaunchAttention_ThrowsForExcessiveSharedMemory()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");

        nint s = _stream!.Handle;

        // seqKv = 100_000 requires ~400 KB shared memory → exceeds any GPU's limit.
        // All pointer args can be zero since the kernel should never launch.
        var ex = Assert.Throws<InvalidOperationException>(() =>
            _kernels!.LaunchAttention(
                q: 0, k: 0, v: 0, output: 0,
                seqQ: 1, seqKv: 100_000,
                numHeads: 1, numKvHeads: 1, headDim: 128,
                positionOffset: 0, slidingWindow: 0, stream: s));

        Assert.Contains("shared memory", ex.Message);
        Assert.Contains("100000", ex.Message);
    }

    [SkippableFact]
    public void LaunchAttentionF32_ThrowsForExcessiveSharedMemory()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");

        nint s = _stream!.Handle;

        var ex = Assert.Throws<InvalidOperationException>(() =>
            _kernels!.LaunchAttentionF32(
                q: 0, k: 0, v: 0, output: 0,
                seqQ: 1, seqKv: 100_000,
                numHeads: 1, numKvHeads: 1, headDim: 128,
                positionOffset: 0, slidingWindow: 0, stream: s));

        Assert.Contains("shared memory", ex.Message);
    }

    public void Dispose()
    {
        _kernels?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
    }
}
