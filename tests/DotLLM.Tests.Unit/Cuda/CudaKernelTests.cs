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

    /// <summary>
    /// Verifies <c>quantized_gemv_q8_0_fused3</c> produces the same outputs as three
    /// back-to-back <c>quantized_gemv_q8_0</c> launches. The fused kernel is just a
    /// block-index dispatch wrapper around <c>q8_0_gemv_one_row</c> (the same body the
    /// unfused kernel uses), so any divergence indicates a wiring / addressing bug
    /// rather than numerical drift.
    /// </summary>
    [SkippableFact]
    public unsafe void FusedQuantizedGemvQ8_0_3_MatchesUnfusedSequence()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");

        // Realistic per-layer Q/K/V dims for a small GQA model: K=576 (hidden),
        // n0=576 (Q), n1=192 (K), n2=192 (V) — like SmolLM-135M.
        const int K = 576;
        const int n0 = 576, n1 = 192, n2 = 192;
        const int Q8_0BlockBytes = 34;
        const int Q8_0GroupSize = 32;
        Assert.Equal(0, K % Q8_0GroupSize);

        nint s = _stream!.Handle;
        int blocksPerRow = K / Q8_0GroupSize;
        long w0Bytes = (long)n0 * blocksPerRow * Q8_0BlockBytes;
        long w1Bytes = (long)n1 * blocksPerRow * Q8_0BlockBytes;
        long w2Bytes = (long)n2 * blocksPerRow * Q8_0BlockBytes;
        long xBytes = (long)K * sizeof(ushort);
        long y0Bytes = (long)n0 * sizeof(ushort);
        long y1Bytes = (long)n1 * sizeof(ushort);
        long y2Bytes = (long)n2 * sizeof(ushort);

        // Synthesize Q8_0 weight blocks: random Half scale + random sbyte qs.
        var rng = new Random(7);
        byte[] w0Host = NewQ8Weights(rng, n0, blocksPerRow);
        byte[] w1Host = NewQ8Weights(rng, n1, blocksPerRow);
        byte[] w2Host = NewQ8Weights(rng, n2, blocksPerRow);
        ushort[] xHost = new ushort[K];
        for (int i = 0; i < K; i++)
            xHost[i] = BitConverter.HalfToUInt16Bits((Half)(rng.NextSingle() * 2 - 1));

        CudaDriverApi.cuMemAlloc_v2(out nint dW0, (nuint)w0Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dW1, (nuint)w1Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dW2, (nuint)w2Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dX, (nuint)xBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dY0Fused, (nuint)y0Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dY1Fused, (nuint)y1Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dY2Fused, (nuint)y2Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dY0Ref, (nuint)y0Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dY1Ref, (nuint)y1Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dY2Ref, (nuint)y2Bytes).ThrowOnError();

        try
        {
            fixed (byte* p = w0Host) CudaDriverApi.cuMemcpyHtoD_v2(dW0, (nint)p, (nuint)w0Bytes).ThrowOnError();
            fixed (byte* p = w1Host) CudaDriverApi.cuMemcpyHtoD_v2(dW1, (nint)p, (nuint)w1Bytes).ThrowOnError();
            fixed (byte* p = w2Host) CudaDriverApi.cuMemcpyHtoD_v2(dW2, (nint)p, (nuint)w2Bytes).ThrowOnError();
            fixed (ushort* p = xHost) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)p, (nuint)xBytes).ThrowOnError();

            // Reference: 3 separate GEMV launches.
            _kernels!.LaunchQuantizedGemv(dW0, QuantizationType.Q8_0, dX, dY0Ref, n0, K, s);
            _kernels!.LaunchQuantizedGemv(dW1, QuantizationType.Q8_0, dX, dY1Ref, n1, K, s);
            _kernels!.LaunchQuantizedGemv(dW2, QuantizationType.Q8_0, dX, dY2Ref, n2, K, s);

            // Under test: single fused launch.
            _kernels!.LaunchFusedQuantizedGemv3(
                dW0, dW1, dW2,
                dY0Fused, dY1Fused, dY2Fused,
                dX, QuantizationType.Q8_0,
                n0, n1, n2, K, s);
            _stream!.Synchronize();

            AssertOutputsBitEqual(dY0Ref, dY0Fused, n0, "Q");
            AssertOutputsBitEqual(dY1Ref, dY1Fused, n1, "K");
            AssertOutputsBitEqual(dY2Ref, dY2Fused, n2, "V");
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(dW0); CudaDriverApi.cuMemFree_v2(dW1); CudaDriverApi.cuMemFree_v2(dW2);
            CudaDriverApi.cuMemFree_v2(dX);
            CudaDriverApi.cuMemFree_v2(dY0Fused); CudaDriverApi.cuMemFree_v2(dY1Fused); CudaDriverApi.cuMemFree_v2(dY2Fused);
            CudaDriverApi.cuMemFree_v2(dY0Ref); CudaDriverApi.cuMemFree_v2(dY1Ref); CudaDriverApi.cuMemFree_v2(dY2Ref);
        }
    }

    /// <summary>
    /// Verifies <c>quantized_gemv_q8_0_fused2</c> matches two back-to-back
    /// <c>quantized_gemv_q8_0</c> launches (Gate/Up FFN projections).
    /// </summary>
    [SkippableFact]
    public unsafe void FusedQuantizedGemvQ8_0_2_MatchesUnfusedSequence()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");

        // Realistic Gate/Up dims: K=576 (hidden), n0=n1=1536 (intermediate).
        const int K = 576;
        const int n0 = 1536, n1 = 1536;
        const int Q8_0BlockBytes = 34;
        const int Q8_0GroupSize = 32;

        nint s = _stream!.Handle;
        int blocksPerRow = K / Q8_0GroupSize;
        long wBytes = (long)n0 * blocksPerRow * Q8_0BlockBytes;
        long xBytes = (long)K * sizeof(ushort);
        long yBytes = (long)n0 * sizeof(ushort);

        var rng = new Random(11);
        byte[] w0Host = NewQ8Weights(rng, n0, blocksPerRow);
        byte[] w1Host = NewQ8Weights(rng, n1, blocksPerRow);
        ushort[] xHost = new ushort[K];
        for (int i = 0; i < K; i++)
            xHost[i] = BitConverter.HalfToUInt16Bits((Half)(rng.NextSingle() * 2 - 1));

        CudaDriverApi.cuMemAlloc_v2(out nint dW0, (nuint)wBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dW1, (nuint)wBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dX, (nuint)xBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dY0Fused, (nuint)yBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dY1Fused, (nuint)yBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dY0Ref, (nuint)yBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dY1Ref, (nuint)yBytes).ThrowOnError();

        try
        {
            fixed (byte* p = w0Host) CudaDriverApi.cuMemcpyHtoD_v2(dW0, (nint)p, (nuint)wBytes).ThrowOnError();
            fixed (byte* p = w1Host) CudaDriverApi.cuMemcpyHtoD_v2(dW1, (nint)p, (nuint)wBytes).ThrowOnError();
            fixed (ushort* p = xHost) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)p, (nuint)xBytes).ThrowOnError();

            _kernels!.LaunchQuantizedGemv(dW0, QuantizationType.Q8_0, dX, dY0Ref, n0, K, s);
            _kernels!.LaunchQuantizedGemv(dW1, QuantizationType.Q8_0, dX, dY1Ref, n1, K, s);

            _kernels!.LaunchFusedQuantizedGemv2(
                dW0, dW1,
                dY0Fused, dY1Fused,
                dX, QuantizationType.Q8_0,
                n0, n1, K, s);
            _stream!.Synchronize();

            AssertOutputsBitEqual(dY0Ref, dY0Fused, n0, "Gate");
            AssertOutputsBitEqual(dY1Ref, dY1Fused, n1, "Up");
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(dW0); CudaDriverApi.cuMemFree_v2(dW1);
            CudaDriverApi.cuMemFree_v2(dX);
            CudaDriverApi.cuMemFree_v2(dY0Fused); CudaDriverApi.cuMemFree_v2(dY1Fused);
            CudaDriverApi.cuMemFree_v2(dY0Ref); CudaDriverApi.cuMemFree_v2(dY1Ref);
        }
    }

    /// <summary>
    /// Synthesizes a Q8_0 weight tensor: per-block Half scale + 32 sbyte values, packed
    /// into <c>34 × blocksPerRow × rows</c> bytes matching GGUF layout.
    /// </summary>
    private static byte[] NewQ8Weights(Random rng, int rows, int blocksPerRow)
    {
        const int Q8_0BlockBytes = 34;
        const int Q8_0GroupSize = 32;
        byte[] w = new byte[(long)rows * blocksPerRow * Q8_0BlockBytes];
        unsafe
        {
            fixed (byte* p = w)
            {
                byte* cur = p;
                for (int row = 0; row < rows; row++)
                {
                    for (int b = 0; b < blocksPerRow; b++)
                    {
                        // Half scale, small positive (typical of GGML quantization output).
                        Half scale = (Half)(rng.NextSingle() * 0.1f + 0.001f);
                        ushort scaleBits = BitConverter.HalfToUInt16Bits(scale);
                        cur[0] = (byte)(scaleBits & 0xFF);
                        cur[1] = (byte)(scaleBits >> 8);
                        for (int j = 0; j < Q8_0GroupSize; j++)
                            cur[2 + j] = (byte)(sbyte)rng.Next(-127, 128);
                        cur += Q8_0BlockBytes;
                    }
                }
            }
        }
        return w;
    }

    /// <summary>
    /// Asserts two device FP16 result vectors are bit-identical. Fused vs unfused
    /// kernels share the same per-row dot-product body, so any difference would
    /// indicate an addressing or block-dispatch bug — not floating-point drift.
    /// </summary>
    private unsafe void AssertOutputsBitEqual(nint dRef, nint dActual, int n, string label)
    {
        ushort[] refHost = new ushort[n];
        ushort[] actHost = new ushort[n];
        long bytes = (long)n * sizeof(ushort);
        fixed (ushort* p = refHost) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dRef, (nuint)bytes).ThrowOnError();
        fixed (ushort* p = actHost) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dActual, (nuint)bytes).ThrowOnError();

        int mismatches = 0;
        int firstMismatch = -1;
        for (int i = 0; i < n; i++)
        {
            if (refHost[i] != actHost[i])
            {
                if (mismatches == 0) firstMismatch = i;
                mismatches++;
            }
        }
        Assert.True(mismatches == 0,
            $"{label}: {mismatches}/{n} elements differ; first mismatch at index {firstMismatch} " +
            $"(ref={(firstMismatch >= 0 ? (float)BitConverter.UInt16BitsToHalf(refHost[firstMismatch]) : 0f):R} " +
            $"actual={(firstMismatch >= 0 ? (float)BitConverter.UInt16BitsToHalf(actHost[firstMismatch]) : 0f):R})");
    }

    public void Dispose()
    {
        _kernels?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
    }
}
