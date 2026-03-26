using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Per-kernel comparison tests: run each CUDA kernel and compare output against CPU reference.
/// Uses SmolLM-135M dimensions: hidden=576, heads=9, kv_heads=3, head_dim=64.
/// </summary>
[Trait("Category", "GPU")]
public class CudaKernelComparisonTests : IDisposable
{
    private const int HiddenSize = 576;
    private const int NumHeads = 9;
    private const int NumKvHeads = 3;
    private const int HeadDim = 64;
    private const float RmsEps = 1e-5f;
    private const float RopeTheta = 10000.0f;

    private readonly ITestOutputHelper _output;
    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaKernels? _kernels;
    private readonly bool _available;

    public CudaKernelComparisonTests(ITestOutputHelper output)
    {
        _output = output;
        if (!CudaDevice.IsAvailable()) return;

        _ctx = CudaContext.Create(0);
        _stream = CudaStream.Create();

        string? ptxDir = FindPtxDir();
        if (ptxDir != null)
            _kernels = new CudaKernels(ptxDir);

        _available = _kernels != null;
    }

    private static string? FindPtxDir()
    {
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

    // ─────────────────── RmsNorm ───────────────────

    [SkippableFact]
    public unsafe void RmsNormF32_MatchesCpuReference()
    {
        SkipIfUnavailable();

        int n = HiddenSize;
        int rows = 1;
        var rng = new Random(42);

        float[] input = RandomF32(rng, n);
        float[] weight = RandomF32(rng, n, scale: 1.0f);

        // CPU reference
        float[] cpuResult = new float[n];
        RmsNorm.Execute(input, weight, RmsEps, cpuResult);

        // GPU
        float[] gpuResult = RunGpuRmsNorm(input, weight, n, rows);

        CompareResults("RmsNorm", cpuResult, gpuResult, tolerance: 0.001f);
    }

    [SkippableFact]
    public unsafe void RmsNormF32_MultiRow_MatchesCpuReference()
    {
        SkipIfUnavailable();

        int n = HiddenSize;
        int rows = 4;
        var rng = new Random(42);

        float[] input = RandomF32(rng, n * rows);
        float[] weight = RandomF32(rng, n, scale: 1.0f);

        // CPU reference (per-row)
        float[] cpuResult = new float[n * rows];
        for (int r = 0; r < rows; r++)
            RmsNorm.Execute(input.AsSpan(r * n, n), weight, RmsEps, cpuResult.AsSpan(r * n, n));

        // GPU
        float[] gpuResult = RunGpuRmsNorm(input, weight, n, rows);

        CompareResults("RmsNorm(4 rows)", cpuResult, gpuResult, tolerance: 0.001f);
    }

    private unsafe float[] RunGpuRmsNorm(float[] input, float[] weight, int n, int rows)
    {
        nint s = _stream!.Handle;
        long inputBytes = (long)input.Length * sizeof(float);
        long weightBytes = (long)weight.Length * sizeof(float);
        long outputBytes = inputBytes;

        CudaDriverApi.cuMemAlloc_v2(out nint devInput, (nuint)inputBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devWeight, (nuint)weightBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devOutput, (nuint)outputBytes).ThrowOnError();

        try
        {
            fixed (float* pIn = input) CudaDriverApi.cuMemcpyHtoD_v2(devInput, (nint)pIn, (nuint)inputBytes).ThrowOnError();
            fixed (float* pW = weight) CudaDriverApi.cuMemcpyHtoD_v2(devWeight, (nint)pW, (nuint)weightBytes).ThrowOnError();

            _kernels!.LaunchRmsNormF32(devInput, devWeight, devOutput, n, RmsEps, rows, s);
            _stream!.Synchronize();

            float[] result = new float[input.Length];
            fixed (float* pOut = result) CudaDriverApi.cuMemcpyDtoH_v2((nint)pOut, devOutput, (nuint)outputBytes).ThrowOnError();
            return result;
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devInput);
            CudaDriverApi.cuMemFree_v2(devWeight);
            CudaDriverApi.cuMemFree_v2(devOutput);
        }
    }

    // ─────────────────── RoPE ───────────────────

    [SkippableFact]
    public unsafe void RoPEF32_MatchesCpuReference()
    {
        SkipIfUnavailable();

        int seqLen = 4;
        int ropeDim = HeadDim; // full rotation
        int halfRope = ropeDim / 2;
        var rng = new Random(42);

        float[] q = RandomF32(rng, seqLen * NumHeads * HeadDim);
        float[] k = RandomF32(rng, seqLen * NumKvHeads * HeadDim);
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        // CPU reference: precompute freq table, then apply
        int maxPos = seqLen + 1;
        float[] cosTable = new float[maxPos * halfRope];
        float[] sinTable = new float[maxPos * halfRope];
        RoPE.PrecomputeFrequencyTable(maxPos, ropeDim, RopeTheta, cosTable, sinTable);

        float[] cpuQ = (float[])q.Clone();
        float[] cpuK = (float[])k.Clone();
        RoPE.Execute(cpuQ, cpuK, positions, NumHeads, NumKvHeads, HeadDim, ropeDim,
                     cosTable, sinTable, Core.Configuration.RoPEType.Norm);

        // GPU: RoPE computes angles inline (no precomputed table)
        float[] gpuQ = (float[])q.Clone();
        float[] gpuK = (float[])k.Clone();
        RunGpuRoPE(gpuQ, gpuK, positions, seqLen, NumHeads, NumKvHeads, HeadDim, ropeDim,
                   RopeTheta, ropeType: 0); // 0 = Norm/interleaved in CUDA kernel

        CompareResults("RoPE Q", cpuQ, gpuQ, tolerance: 0.001f);
        CompareResults("RoPE K", cpuK, gpuK, tolerance: 0.001f);
    }

    [SkippableFact]
    public unsafe void RoPEF32_NeoX_MatchesCpuReference()
    {
        SkipIfUnavailable();

        // Use Qwen2.5-0.5B dimensions: hidden=896, heads=14, kv_heads=2, head_dim=64
        int numHeads = 14;
        int numKvHeads = 2;
        int headDim = 64;
        float theta = 1000000.0f; // Qwen2.5 uses 1M theta
        int seqLen = 4;
        int ropeDim = headDim; // full rotation
        int halfRope = ropeDim / 2;
        var rng = new Random(42);

        float[] q = RandomF32(rng, seqLen * numHeads * headDim);
        float[] k = RandomF32(rng, seqLen * numKvHeads * headDim);
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        // CPU reference: NeoX (non-interleaved)
        int maxPos = seqLen + 1;
        float[] cosTable = new float[maxPos * halfRope];
        float[] sinTable = new float[maxPos * halfRope];
        RoPE.PrecomputeFrequencyTable(maxPos, ropeDim, theta, cosTable, sinTable);

        float[] cpuQ = (float[])q.Clone();
        float[] cpuK = (float[])k.Clone();
        RoPE.Execute(cpuQ, cpuK, positions, numHeads, numKvHeads, headDim, ropeDim,
                     cosTable, sinTable, Core.Configuration.RoPEType.NeoX);

        // GPU: NeoX (rope_type=1)
        float[] gpuQ = (float[])q.Clone();
        float[] gpuK = (float[])k.Clone();
        RunGpuRoPE(gpuQ, gpuK, positions, seqLen, numHeads, numKvHeads, headDim, ropeDim,
                   theta, ropeType: 1); // 1 = NeoX in CUDA kernel

        CompareResults("RoPE-NeoX Q", cpuQ, gpuQ, tolerance: 0.001f);
        CompareResults("RoPE-NeoX K", cpuK, gpuK, tolerance: 0.001f);
    }

    private unsafe void RunGpuRoPE(float[] q, float[] k, int[] positions,
                                    int seqLen, int numHeads, int numKvHeads, int headDim,
                                    int ropeDim, float theta, int ropeType)
    {
        nint s = _stream!.Handle;
        long qBytes = (long)q.Length * sizeof(float);
        long kBytes = (long)k.Length * sizeof(float);
        long posBytes = (long)positions.Length * sizeof(int);

        CudaDriverApi.cuMemAlloc_v2(out nint devQ, (nuint)qBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devK, (nuint)kBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devPos, (nuint)posBytes).ThrowOnError();

        try
        {
            fixed (float* pQ = q) CudaDriverApi.cuMemcpyHtoD_v2(devQ, (nint)pQ, (nuint)qBytes).ThrowOnError();
            fixed (float* pK = k) CudaDriverApi.cuMemcpyHtoD_v2(devK, (nint)pK, (nuint)kBytes).ThrowOnError();
            fixed (int* pP = positions) CudaDriverApi.cuMemcpyHtoD_v2(devPos, (nint)pP, (nuint)posBytes).ThrowOnError();

            _kernels!.LaunchRoPEF32(devQ, devK, devPos, seqLen, numHeads, numKvHeads,
                                    headDim, ropeDim, theta, ropeType, s);
            _stream!.Synchronize();

            fixed (float* pQ = q) CudaDriverApi.cuMemcpyDtoH_v2((nint)pQ, devQ, (nuint)qBytes).ThrowOnError();
            fixed (float* pK = k) CudaDriverApi.cuMemcpyDtoH_v2((nint)pK, devK, (nuint)kBytes).ThrowOnError();
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devQ);
            CudaDriverApi.cuMemFree_v2(devK);
            CudaDriverApi.cuMemFree_v2(devPos);
        }
    }

    // ─────────────────── SwiGLU ───────────────────

    [SkippableFact]
    public unsafe void SwiGLUF32_MatchesCpuReference()
    {
        SkipIfUnavailable();

        int n = HiddenSize; // intermediate size per token
        int seqLen = 4;
        var rng = new Random(42);

        float[] gate = RandomF32(rng, n * seqLen);
        float[] up = RandomF32(rng, n * seqLen);

        // CPU reference
        float[] cpuResult = new float[n * seqLen];
        FusedOps.SwiGLU(gate, up, cpuResult);

        // GPU
        float[] gpuResult = RunGpuSwiGLU(gate, up, n, seqLen);

        CompareResults("SwiGLU", cpuResult, gpuResult, tolerance: 0.001f);
    }

    private unsafe float[] RunGpuSwiGLU(float[] gate, float[] up, int n, int seqLen)
    {
        nint s = _stream!.Handle;
        long totalBytes = (long)gate.Length * sizeof(float);

        CudaDriverApi.cuMemAlloc_v2(out nint devGate, (nuint)totalBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devUp, (nuint)totalBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devOut, (nuint)totalBytes).ThrowOnError();

        try
        {
            fixed (float* pG = gate) CudaDriverApi.cuMemcpyHtoD_v2(devGate, (nint)pG, (nuint)totalBytes).ThrowOnError();
            fixed (float* pU = up) CudaDriverApi.cuMemcpyHtoD_v2(devUp, (nint)pU, (nuint)totalBytes).ThrowOnError();

            _kernels!.LaunchSwiGLUF32(devGate, devUp, devOut, n, seqLen, s);
            _stream!.Synchronize();

            float[] result = new float[gate.Length];
            fixed (float* pO = result) CudaDriverApi.cuMemcpyDtoH_v2((nint)pO, devOut, (nuint)totalBytes).ThrowOnError();
            return result;
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devGate);
            CudaDriverApi.cuMemFree_v2(devUp);
            CudaDriverApi.cuMemFree_v2(devOut);
        }
    }

    // ─────────────────── Attention ───────────────────

    [SkippableFact]
    public unsafe void AttentionF32_SingleToken_MatchesCpuReference()
    {
        SkipIfUnavailable();

        // Decode step: 1 query token, 4 KV tokens
        int seqQ = 1, seqKv = 4;
        int posOffset = seqKv - 1; // last position
        var rng = new Random(42);

        float[] q = RandomF32(rng, seqQ * NumHeads * HeadDim, scale: 0.1f);
        float[] k = RandomF32(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);
        float[] v = RandomF32(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);

        // CPU reference
        float[] cpuOutput = new float[seqQ * NumHeads * HeadDim];
        Attention.Execute(q, k, v, cpuOutput, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset);

        // GPU
        float[] gpuOutput = RunGpuAttention(q, k, v, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, 0);

        CompareResults("Attention(1q,4kv)", cpuOutput, gpuOutput, tolerance: 0.01f);
    }

    [SkippableFact]
    public unsafe void AttentionF32_Prefill_MatchesCpuReference()
    {
        SkipIfUnavailable();

        // Prefill: 4 query tokens, 4 KV tokens
        int seqQ = 4, seqKv = 4;
        int posOffset = 0;
        var rng = new Random(42);

        float[] q = RandomF32(rng, seqQ * NumHeads * HeadDim, scale: 0.1f);
        float[] k = RandomF32(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);
        float[] v = RandomF32(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);

        // CPU reference
        float[] cpuOutput = new float[seqQ * NumHeads * HeadDim];
        Attention.Execute(q, k, v, cpuOutput, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset);

        // GPU
        float[] gpuOutput = RunGpuAttention(q, k, v, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, 0);

        CompareResults("Attention(4q,4kv)", cpuOutput, gpuOutput, tolerance: 0.01f);
    }

    private unsafe float[] RunGpuAttention(float[] q, float[] k, float[] v,
                                            int seqQ, int seqKv,
                                            int numHeads, int numKvHeads, int headDim,
                                            int posOffset, int slidingWindow)
    {
        nint s = _stream!.Handle;
        long qBytes = (long)q.Length * sizeof(float);
        long kBytes = (long)k.Length * sizeof(float);
        long vBytes = (long)v.Length * sizeof(float);
        long outBytes = qBytes;

        CudaDriverApi.cuMemAlloc_v2(out nint devQ, (nuint)qBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devK, (nuint)kBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devV, (nuint)vBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devOut, (nuint)outBytes).ThrowOnError();

        try
        {
            fixed (float* pQ = q) CudaDriverApi.cuMemcpyHtoD_v2(devQ, (nint)pQ, (nuint)qBytes).ThrowOnError();
            fixed (float* pK = k) CudaDriverApi.cuMemcpyHtoD_v2(devK, (nint)pK, (nuint)kBytes).ThrowOnError();
            fixed (float* pV = v) CudaDriverApi.cuMemcpyHtoD_v2(devV, (nint)pV, (nuint)vBytes).ThrowOnError();

            _kernels!.LaunchAttentionF32(devQ, devK, devV, devOut,
                seqQ, seqKv, numHeads, numKvHeads, headDim, posOffset, slidingWindow, s);
            _stream!.Synchronize();

            float[] result = new float[q.Length];
            fixed (float* pO = result) CudaDriverApi.cuMemcpyDtoH_v2((nint)pO, devOut, (nuint)outBytes).ThrowOnError();
            return result;
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devQ);
            CudaDriverApi.cuMemFree_v2(devK);
            CudaDriverApi.cuMemFree_v2(devV);
            CudaDriverApi.cuMemFree_v2(devOut);
        }
    }

    // ─────────────────── Embedding (F32 table) ───────────────────

    [SkippableFact]
    public unsafe void EmbeddingF32_MatchesCpuReference()
    {
        SkipIfUnavailable();

        int vocabSize = 64;
        int seqLen = 4;
        var rng = new Random(42);

        float[] embedTable = RandomF32(rng, vocabSize * HiddenSize);
        int[] tokenIds = new int[seqLen];
        for (int i = 0; i < seqLen; i++) tokenIds[i] = rng.Next(vocabSize);

        // CPU reference: simple row copy
        float[] cpuResult = new float[seqLen * HiddenSize];
        for (int t = 0; t < seqLen; t++)
            Array.Copy(embedTable, tokenIds[t] * HiddenSize, cpuResult, t * HiddenSize, HiddenSize);

        // GPU
        float[] gpuResult = RunGpuEmbeddingF32(embedTable, tokenIds, seqLen, HiddenSize);

        CompareResults("Embedding(F32)", cpuResult, gpuResult, tolerance: 0.0f);
    }

    private unsafe float[] RunGpuEmbeddingF32(float[] embedTable, int[] tokenIds, int seqLen, int hiddenSize)
    {
        nint s = _stream!.Handle;
        long tableBytes = (long)embedTable.Length * sizeof(float);
        long idsBytes = (long)tokenIds.Length * sizeof(int);
        long outBytes = (long)seqLen * hiddenSize * sizeof(float);

        CudaDriverApi.cuMemAlloc_v2(out nint devTable, (nuint)tableBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devIds, (nuint)idsBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devOut, (nuint)outBytes).ThrowOnError();

        try
        {
            fixed (float* pT = embedTable) CudaDriverApi.cuMemcpyHtoD_v2(devTable, (nint)pT, (nuint)tableBytes).ThrowOnError();
            fixed (int* pI = tokenIds) CudaDriverApi.cuMemcpyHtoD_v2(devIds, (nint)pI, (nuint)idsBytes).ThrowOnError();

            _kernels!.LaunchEmbeddingLookupF32(devTable, Core.Configuration.QuantizationType.F32,
                devIds, devOut, seqLen, hiddenSize, s);
            _stream!.Synchronize();

            float[] result = new float[seqLen * hiddenSize];
            fixed (float* pO = result) CudaDriverApi.cuMemcpyDtoH_v2((nint)pO, devOut, (nuint)outBytes).ThrowOnError();
            return result;
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devTable);
            CudaDriverApi.cuMemFree_v2(devIds);
            CudaDriverApi.cuMemFree_v2(devOut);
        }
    }

    // ─────────────────── Embedding (Q8_0 table) ───────────────────

    [SkippableFact]
    public unsafe void EmbeddingQ8_0_MatchesCpuReference()
    {
        SkipIfUnavailable();

        int vocabSize = 64;
        int seqLen = 4;
        var rng = new Random(42);

        int[] tokenIds = new int[seqLen];
        for (int i = 0; i < seqLen; i++) tokenIds[i] = rng.Next(vocabSize);

        // Build Q8_0 embedding table: each row = hidden/32 blocks, each block = 2 bytes scale + 32 int8s = 34 bytes
        int blocksPerRow = HiddenSize / 32;
        int bytesPerRow = blocksPerRow * 34;
        byte[] q8Table = new byte[vocabSize * bytesPerRow];
        float[] expectedDequant = new float[vocabSize * HiddenSize];

        for (int tok = 0; tok < vocabSize; tok++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int blockOffset = tok * bytesPerRow + b * 34;
                float scale = (float)(rng.NextDouble() * 0.1 + 0.01);
                ushort scaleF16 = BitConverter.HalfToUInt16Bits((Half)scale);
                q8Table[blockOffset] = (byte)(scaleF16 & 0xFF);
                q8Table[blockOffset + 1] = (byte)(scaleF16 >> 8);

                float actualScale = (float)BitConverter.UInt16BitsToHalf(scaleF16);
                for (int j = 0; j < 32; j++)
                {
                    sbyte val = (sbyte)rng.Next(-127, 128);
                    q8Table[blockOffset + 2 + j] = (byte)val;
                    expectedDequant[tok * HiddenSize + b * 32 + j] = actualScale * val;
                }
            }
        }

        // CPU reference: dequant for each selected token
        float[] cpuResult = new float[seqLen * HiddenSize];
        for (int t = 0; t < seqLen; t++)
            Array.Copy(expectedDequant, tokenIds[t] * HiddenSize, cpuResult, t * HiddenSize, HiddenSize);

        // GPU
        float[] gpuResult = RunGpuEmbeddingQ8(q8Table, tokenIds, seqLen, HiddenSize);

        CompareResults("Embedding(Q8_0)", cpuResult, gpuResult, tolerance: 0.001f);
    }

    private unsafe float[] RunGpuEmbeddingQ8(byte[] q8Table, int[] tokenIds, int seqLen, int hiddenSize)
    {
        nint s = _stream!.Handle;
        long tableBytes = q8Table.Length;
        long idsBytes = (long)tokenIds.Length * sizeof(int);
        long outBytes = (long)seqLen * hiddenSize * sizeof(float);

        CudaDriverApi.cuMemAlloc_v2(out nint devTable, (nuint)tableBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devIds, (nuint)idsBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devOut, (nuint)outBytes).ThrowOnError();

        try
        {
            fixed (byte* pT = q8Table) CudaDriverApi.cuMemcpyHtoD_v2(devTable, (nint)pT, (nuint)tableBytes).ThrowOnError();
            fixed (int* pI = tokenIds) CudaDriverApi.cuMemcpyHtoD_v2(devIds, (nint)pI, (nuint)idsBytes).ThrowOnError();

            _kernels!.LaunchEmbeddingLookupF32(devTable, Core.Configuration.QuantizationType.Q8_0,
                devIds, devOut, seqLen, hiddenSize, s);
            _stream!.Synchronize();

            float[] result = new float[seqLen * hiddenSize];
            fixed (float* pO = result) CudaDriverApi.cuMemcpyDtoH_v2((nint)pO, devOut, (nuint)outBytes).ThrowOnError();
            return result;
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devTable);
            CudaDriverApi.cuMemFree_v2(devIds);
            CudaDriverApi.cuMemFree_v2(devOut);
        }
    }

    // ─────────────────── Quantized GEMV (Q8_0 weight, F32 input) ───────────────────

    [SkippableFact]
    public unsafe void QuantizedGemvQ8F32_MatchesCpuReference()
    {
        SkipIfUnavailable();

        // Matrix: n=64 output rows, k=576 (hidden) input columns
        int n = 64, k = HiddenSize;
        var rng = new Random(42);

        float[] x = RandomF32(rng, k, scale: 0.5f);

        // Build Q8_0 weight matrix: n rows, each row has k/32 blocks of 34 bytes
        int blocksPerRow = k / 32;
        int bytesPerRow = blocksPerRow * 34;
        byte[] q8Weight = new byte[n * bytesPerRow];
        float[,] dequantWeight = new float[n, k]; // for CPU reference

        for (int row = 0; row < n; row++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int blockOffset = row * bytesPerRow + b * 34;
                float scale = (float)(rng.NextDouble() * 0.1 + 0.01);
                ushort scaleF16 = BitConverter.HalfToUInt16Bits((Half)scale);
                q8Weight[blockOffset] = (byte)(scaleF16 & 0xFF);
                q8Weight[blockOffset + 1] = (byte)(scaleF16 >> 8);

                float actualScale = (float)BitConverter.UInt16BitsToHalf(scaleF16);
                for (int j = 0; j < 32; j++)
                {
                    sbyte val = (sbyte)rng.Next(-127, 128);
                    q8Weight[blockOffset + 2 + j] = (byte)val;
                    dequantWeight[row, b * 32 + j] = actualScale * val;
                }
            }
        }

        // CPU reference: y[row] = dequant_weight[row,:] · x
        float[] cpuResult = new float[n];
        for (int row = 0; row < n; row++)
        {
            float acc = 0;
            for (int j = 0; j < k; j++)
                acc += dequantWeight[row, j] * x[j];
            cpuResult[row] = acc;
        }

        // GPU
        float[] gpuResult = RunGpuQuantizedGemv(q8Weight, x, n, k);

        CompareResults("QuantizedGEMV(Q8_0)", cpuResult, gpuResult, tolerance: 0.05f);
    }

    [SkippableFact]
    public unsafe void QuantizedGemvQ8F32_LargeMatrix_MatchesCpuReference()
    {
        SkipIfUnavailable();

        // Full-size: n=576 (hidden), k=576 (hidden) — like a projection layer
        int n = HiddenSize, k = HiddenSize;
        var rng = new Random(42);

        float[] x = RandomF32(rng, k, scale: 0.5f);

        int blocksPerRow = k / 32;
        int bytesPerRow = blocksPerRow * 34;
        byte[] q8Weight = new byte[n * bytesPerRow];
        float[,] dequantWeight = new float[n, k];

        for (int row = 0; row < n; row++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int blockOffset = row * bytesPerRow + b * 34;
                float scale = (float)(rng.NextDouble() * 0.1 + 0.01);
                ushort scaleF16 = BitConverter.HalfToUInt16Bits((Half)scale);
                q8Weight[blockOffset] = (byte)(scaleF16 & 0xFF);
                q8Weight[blockOffset + 1] = (byte)(scaleF16 >> 8);

                float actualScale = (float)BitConverter.UInt16BitsToHalf(scaleF16);
                for (int j = 0; j < 32; j++)
                {
                    sbyte val = (sbyte)rng.Next(-127, 128);
                    q8Weight[blockOffset + 2 + j] = (byte)val;
                    dequantWeight[row, b * 32 + j] = actualScale * val;
                }
            }
        }

        float[] cpuResult = new float[n];
        for (int row = 0; row < n; row++)
        {
            float acc = 0;
            for (int j = 0; j < k; j++)
                acc += dequantWeight[row, j] * x[j];
            cpuResult[row] = acc;
        }

        float[] gpuResult = RunGpuQuantizedGemv(q8Weight, x, n, k);

        CompareResults("QuantizedGEMV(Q8_0,576x576)", cpuResult, gpuResult, tolerance: 0.05f);
    }

    private unsafe float[] RunGpuQuantizedGemv(byte[] q8Weight, float[] x, int n, int k)
    {
        nint s = _stream!.Handle;
        long wBytes = q8Weight.Length;
        long xBytes = (long)x.Length * sizeof(float);
        long yBytes = (long)n * sizeof(float);

        CudaDriverApi.cuMemAlloc_v2(out nint devW, (nuint)wBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devX, (nuint)xBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devY, (nuint)yBytes).ThrowOnError();

        try
        {
            fixed (byte* pW = q8Weight) CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)pW, (nuint)wBytes).ThrowOnError();
            fixed (float* pX = x) CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)pX, (nuint)xBytes).ThrowOnError();

            _kernels!.LaunchQuantizedGemvF32In(devW, devX, devY, n, k, s);
            _stream!.Synchronize();

            float[] result = new float[n];
            fixed (float* pY = result) CudaDriverApi.cuMemcpyDtoH_v2((nint)pY, devY, (nuint)yBytes).ThrowOnError();
            return result;
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devW);
            CudaDriverApi.cuMemFree_v2(devX);
            CudaDriverApi.cuMemFree_v2(devY);
        }
    }

    // ─────────────────── KV-Cache Roundtrip ───────────────────

    /// <summary>
    /// Writes K/V data to CudaKvCache positions, then runs attention against the cache.
    /// Compares with direct attention (no cache). If the cache corrupts data, this fails.
    /// </summary>
    [SkippableFact]
    public unsafe void KvCacheRoundtrip_AttentionMatchesDirect()
    {
        SkipIfUnavailable();

        int seqQ = 1, seqKv = 4;
        int posOffset = seqKv - 1;
        var rng = new Random(42);

        float[] q = RandomF32(rng, seqQ * NumHeads * HeadDim, scale: 0.1f);
        float[] k = RandomF32(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);
        float[] v = RandomF32(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);

        // 1. Direct attention (no cache)
        float[] directOutput = RunGpuAttention(q, k, v, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, 0);

        // 2. Write K/V to cache, then attention against cache
        float[] cacheOutput = RunGpuAttentionViaKvCache(q, k, v, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset);

        CompareResults("KvCache-Roundtrip(1q,4kv)", directOutput, cacheOutput, tolerance: 0.0f);
    }

    /// <summary>
    /// Simulates prefill + decode: write 4 tokens to cache, run attention for token 5.
    /// Then write token 5 to cache, run attention for token 6.
    /// Compares decode output against direct attention over all tokens.
    /// </summary>
    [SkippableFact]
    public unsafe void KvCacheRoundtrip_PrefillThenDecode_MatchesDirect()
    {
        SkipIfUnavailable();

        int prefillLen = 4;
        int totalLen = 6; // prefill 4, then decode 2 more
        var rng = new Random(42);

        // Pre-generate all K/V/Q data for all tokens
        float[] allK = RandomF32(rng, totalLen * NumKvHeads * HeadDim, scale: 0.1f);
        float[] allV = RandomF32(rng, totalLen * NumKvHeads * HeadDim, scale: 0.1f);
        float[] allQ = RandomF32(rng, totalLen * NumHeads * HeadDim, scale: 0.1f);

        nint s = _stream!.Handle;
        int kvRowBytes = NumKvHeads * HeadDim * sizeof(float);
        int qRowBytes = NumHeads * HeadDim * sizeof(float);

        // Allocate a KV-cache (simulating CudaKvCache manually)
        int maxSeqLen = totalLen + 1;
        long cacheLayerBytes = (long)maxSeqLen * kvRowBytes;
        CudaDriverApi.cuMemAlloc_v2(out nint cacheK, (nuint)cacheLayerBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint cacheV, (nuint)cacheLayerBytes).ThrowOnError();

        // Upload all K/V to a temp buffer
        long allKBytes = (long)allK.Length * sizeof(float);
        long allVBytes = (long)allV.Length * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint devAllK, (nuint)allKBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devAllV, (nuint)allVBytes).ThrowOnError();

        try
        {
            fixed (float* pK = allK) CudaDriverApi.cuMemcpyHtoD_v2(devAllK, (nint)pK, (nuint)allKBytes).ThrowOnError();
            fixed (float* pV = allV) CudaDriverApi.cuMemcpyHtoD_v2(devAllV, (nint)pV, (nuint)allVBytes).ThrowOnError();

            // 1. Prefill: copy tokens 0..3 into cache positions 0..3
            for (int i = 0; i < prefillLen; i++)
            {
                CudaDriverApi.cuMemcpyDtoDAsync_v2(
                    cacheK + (nint)(i * kvRowBytes),
                    devAllK + (nint)(i * kvRowBytes),
                    (nuint)kvRowBytes, s).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoDAsync_v2(
                    cacheV + (nint)(i * kvRowBytes),
                    devAllV + (nint)(i * kvRowBytes),
                    (nuint)kvRowBytes, s).ThrowOnError();
            }

            // Now test decode steps
            for (int decodePos = prefillLen; decodePos < totalLen; decodePos++)
            {
                // Store new K/V at this position
                CudaDriverApi.cuMemcpyDtoDAsync_v2(
                    cacheK + (nint)(decodePos * kvRowBytes),
                    devAllK + (nint)(decodePos * kvRowBytes),
                    (nuint)kvRowBytes, s).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoDAsync_v2(
                    cacheV + (nint)(decodePos * kvRowBytes),
                    devAllV + (nint)(decodePos * kvRowBytes),
                    (nuint)kvRowBytes, s).ThrowOnError();

                int seqKv = decodePos + 1;
                int posOffset = decodePos;

                // GPU attention via cache
                long qBytes = (long)qRowBytes;
                CudaDriverApi.cuMemAlloc_v2(out nint devQ, (nuint)qBytes).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out nint devOut, (nuint)qBytes).ThrowOnError();
                try
                {
                    fixed (float* pQ = allQ.AsSpan(decodePos * NumHeads * HeadDim, NumHeads * HeadDim))
                        CudaDriverApi.cuMemcpyHtoD_v2(devQ, (nint)pQ, (nuint)qBytes).ThrowOnError();

                    _kernels!.LaunchAttentionF32(devQ, cacheK, cacheV, devOut,
                        1, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, 0, s);
                    _stream!.Synchronize();

                    float[] gpuCacheResult = new float[NumHeads * HeadDim];
                    fixed (float* pO = gpuCacheResult)
                        CudaDriverApi.cuMemcpyDtoH_v2((nint)pO, devOut, (nuint)qBytes).ThrowOnError();

                    // CPU reference: attention over all tokens 0..decodePos
                    float[] qSlice = allQ.AsSpan(decodePos * NumHeads * HeadDim, NumHeads * HeadDim).ToArray();
                    float[] kSlice = allK.AsSpan(0, seqKv * NumKvHeads * HeadDim).ToArray();
                    float[] vSlice = allV.AsSpan(0, seqKv * NumKvHeads * HeadDim).ToArray();
                    float[] cpuResult = new float[NumHeads * HeadDim];
                    Attention.Execute(qSlice, kSlice, vSlice, cpuResult,
                        1, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset);

                    CompareResults($"KvCache-Decode(pos={decodePos},seqKv={seqKv})",
                        cpuResult, gpuCacheResult, tolerance: 0.01f);
                }
                finally
                {
                    CudaDriverApi.cuMemFree_v2(devQ);
                    CudaDriverApi.cuMemFree_v2(devOut);
                }
            }
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(cacheK);
            CudaDriverApi.cuMemFree_v2(cacheV);
            CudaDriverApi.cuMemFree_v2(devAllK);
            CudaDriverApi.cuMemFree_v2(devAllV);
        }
    }

    private unsafe float[] RunGpuAttentionViaKvCache(float[] q, float[] k, float[] v,
                                                      int seqQ, int seqKv,
                                                      int numHeads, int numKvHeads, int headDim,
                                                      int posOffset)
    {
        nint s = _stream!.Handle;
        int kvStride = numKvHeads * headDim;
        int kvRowBytes = kvStride * sizeof(float);

        // Allocate cache large enough
        long cacheBytes = (long)seqKv * kvRowBytes;
        CudaDriverApi.cuMemAlloc_v2(out nint cacheK, (nuint)cacheBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint cacheV, (nuint)cacheBytes).ThrowOnError();

        long qBytes = (long)q.Length * sizeof(float);
        long outBytes = qBytes;
        CudaDriverApi.cuMemAlloc_v2(out nint devQ, (nuint)qBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devOut, (nuint)outBytes).ThrowOnError();

        // Upload K/V to temp device buffer, then copy row-by-row into cache (simulating UpdateDevice)
        long kBytes = (long)k.Length * sizeof(float);
        long vBytes = (long)v.Length * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint devK, (nuint)kBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devV, (nuint)vBytes).ThrowOnError();

        try
        {
            fixed (float* pK = k) CudaDriverApi.cuMemcpyHtoD_v2(devK, (nint)pK, (nuint)kBytes).ThrowOnError();
            fixed (float* pV = v) CudaDriverApi.cuMemcpyHtoD_v2(devV, (nint)pV, (nuint)vBytes).ThrowOnError();

            // Write each KV row to cache (simulating CudaKvCache.UpdateDevice)
            for (int i = 0; i < seqKv; i++)
            {
                CudaDriverApi.cuMemcpyDtoDAsync_v2(
                    cacheK + (nint)(i * kvRowBytes),
                    devK + (nint)(i * kvRowBytes),
                    (nuint)kvRowBytes, s).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoDAsync_v2(
                    cacheV + (nint)(i * kvRowBytes),
                    devV + (nint)(i * kvRowBytes),
                    (nuint)kvRowBytes, s).ThrowOnError();
            }

            // Run attention against cache
            fixed (float* pQ = q) CudaDriverApi.cuMemcpyHtoD_v2(devQ, (nint)pQ, (nuint)qBytes).ThrowOnError();

            _kernels!.LaunchAttentionF32(devQ, cacheK, cacheV, devOut,
                seqQ, seqKv, numHeads, numKvHeads, headDim, posOffset, 0, s);
            _stream!.Synchronize();

            float[] result = new float[q.Length];
            fixed (float* pO = result) CudaDriverApi.cuMemcpyDtoH_v2((nint)pO, devOut, (nuint)outBytes).ThrowOnError();
            return result;
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(cacheK);
            CudaDriverApi.cuMemFree_v2(cacheV);
            CudaDriverApi.cuMemFree_v2(devQ);
            CudaDriverApi.cuMemFree_v2(devOut);
            CudaDriverApi.cuMemFree_v2(devK);
            CudaDriverApi.cuMemFree_v2(devV);
        }
    }

    // ─────────────────── Chained: RmsNorm → GEMV → RoPE → Attention ───────────────────

    /// <summary>
    /// Tests the full attention block chain on GPU vs CPU: RmsNorm → Q/K/V GEMV → RoPE → Attention.
    /// This tests the composition of kernels, not individual kernels.
    /// </summary>
    [SkippableFact]
    public unsafe void ChainedAttentionBlock_MatchesCpuReference()
    {
        SkipIfUnavailable();

        int seqLen = 1; // decode step
        int ropeDim = HeadDim;
        int halfRope = ropeDim / 2;
        int qDim = NumHeads * HeadDim;     // 576
        int kvDim = NumKvHeads * HeadDim;  // 192
        var rng = new Random(42);

        // Input: hidden state from previous layer (FP32)
        float[] hidden = RandomF32(rng, seqLen * HiddenSize);
        float[] normWeight = RandomF32(rng, HiddenSize, scale: 1.0f);

        // Q/K/V weight matrices as Q8_0
        byte[] qWeight = RandomQ8_0(rng, qDim, HiddenSize);
        byte[] kWeight = RandomQ8_0(rng, kvDim, HiddenSize);
        byte[] vWeight = RandomQ8_0(rng, kvDim, HiddenSize);
        float[,] qWeightDequant = DequantQ8_0(qWeight, qDim, HiddenSize);
        float[,] kWeightDequant = DequantQ8_0(kWeight, kvDim, HiddenSize);
        float[,] vWeightDequant = DequantQ8_0(vWeight, kvDim, HiddenSize);

        int[] positions = [3]; // position 3 (simulating decode after 3 prefill tokens)

        // Fake KV-cache: 3 existing entries + 1 new = 4 total
        float[] cachedK = RandomF32(rng, 3 * kvDim, scale: 0.1f);
        float[] cachedV = RandomF32(rng, 3 * kvDim, scale: 0.1f);

        // === CPU Reference ===
        // 1. RmsNorm
        float[] cpuNormOut = new float[HiddenSize];
        RmsNorm.Execute(hidden, normWeight, RmsEps, cpuNormOut);

        // 2. Q/K/V projection (dequant weight @ norm output)
        float[] cpuQ = new float[qDim];
        float[] cpuK = new float[kvDim];
        float[] cpuV = new float[kvDim];
        for (int r = 0; r < qDim; r++)
        {
            float acc = 0;
            for (int j = 0; j < HiddenSize; j++) acc += qWeightDequant[r, j] * cpuNormOut[j];
            cpuQ[r] = acc;
        }
        for (int r = 0; r < kvDim; r++)
        {
            float kAcc = 0, vAcc = 0;
            for (int j = 0; j < HiddenSize; j++)
            {
                kAcc += kWeightDequant[r, j] * cpuNormOut[j];
                vAcc += vWeightDequant[r, j] * cpuNormOut[j];
            }
            cpuK[r] = kAcc;
            cpuV[r] = vAcc;
        }

        // 3. RoPE (interleaved, rope_type=0)
        float[] cosTable = new float[(positions[0] + 2) * halfRope];
        float[] sinTable = new float[(positions[0] + 2) * halfRope];
        RoPE.PrecomputeFrequencyTable(positions[0] + 2, ropeDim, RopeTheta, cosTable, sinTable);
        RoPE.Execute(cpuQ, cpuK, positions, NumHeads, NumKvHeads, HeadDim, ropeDim,
                     cosTable, sinTable, Core.Configuration.RoPEType.Norm);

        // 4. Assemble full KV (cache + new) for attention
        int seqKv = 4; // 3 cached + 1 new
        float[] fullK = new float[seqKv * kvDim];
        float[] fullV = new float[seqKv * kvDim];
        Array.Copy(cachedK, 0, fullK, 0, 3 * kvDim);
        Array.Copy(cpuK, 0, fullK, 3 * kvDim, kvDim);
        Array.Copy(cachedV, 0, fullV, 0, 3 * kvDim);
        Array.Copy(cpuV, 0, fullV, 3 * kvDim, kvDim);

        // 5. Attention
        float[] cpuAttnOut = new float[qDim];
        Attention.Execute(cpuQ, fullK, fullV, cpuAttnOut,
            1, seqKv, NumHeads, NumKvHeads, HeadDim, positions[0]);

        // === GPU ===
        nint s = _stream!.Handle;
        long hiddenBytes = (long)HiddenSize * sizeof(float);
        long qBytes = (long)qDim * sizeof(float);
        long kvBytes = (long)kvDim * sizeof(float);
        long normWBytes = hiddenBytes;
        long cacheKvLayerBytes = (long)8 * kvDim * sizeof(float); // max 8 positions

        CudaDriverApi.cuMemAlloc_v2(out nint devHidden, (nuint)hiddenBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devNormW, (nuint)normWBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devNormOut, (nuint)hiddenBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devQ, (nuint)qBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devK, (nuint)kvBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devV, (nuint)kvBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devPos, (nuint)(positions.Length * sizeof(int))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devCacheK, (nuint)cacheKvLayerBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devCacheV, (nuint)cacheKvLayerBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devAttnOut, (nuint)qBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devQW, (nuint)qWeight.Length).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devKW, (nuint)kWeight.Length).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devVW, (nuint)vWeight.Length).ThrowOnError();

        try
        {
            // Upload
            fixed (float* p = hidden) CudaDriverApi.cuMemcpyHtoD_v2(devHidden, (nint)p, (nuint)hiddenBytes).ThrowOnError();
            fixed (float* p = normWeight) CudaDriverApi.cuMemcpyHtoD_v2(devNormW, (nint)p, (nuint)normWBytes).ThrowOnError();
            fixed (int* p = positions) CudaDriverApi.cuMemcpyHtoD_v2(devPos, (nint)p, (nuint)(positions.Length * sizeof(int))).ThrowOnError();
            fixed (byte* p = qWeight) CudaDriverApi.cuMemcpyHtoD_v2(devQW, (nint)p, (nuint)qWeight.Length).ThrowOnError();
            fixed (byte* p = kWeight) CudaDriverApi.cuMemcpyHtoD_v2(devKW, (nint)p, (nuint)kWeight.Length).ThrowOnError();
            fixed (byte* p = vWeight) CudaDriverApi.cuMemcpyHtoD_v2(devVW, (nint)p, (nuint)vWeight.Length).ThrowOnError();

            // Upload cached K/V to cache (positions 0,1,2)
            fixed (float* pCK = cachedK) CudaDriverApi.cuMemcpyHtoD_v2(devCacheK, (nint)pCK, (nuint)(3 * kvBytes)).ThrowOnError();
            fixed (float* pCV = cachedV) CudaDriverApi.cuMemcpyHtoD_v2(devCacheV, (nint)pCV, (nuint)(3 * kvBytes)).ThrowOnError();

            // 1. RmsNorm
            _kernels!.LaunchRmsNormF32(devHidden, devNormW, devNormOut, HiddenSize, RmsEps, 1, s);

            // 2. Q/K/V GEMV
            _kernels.LaunchQuantizedGemvF32In(devQW, devNormOut, devQ, qDim, HiddenSize, s);
            _kernels.LaunchQuantizedGemvF32In(devKW, devNormOut, devK, kvDim, HiddenSize, s);
            _kernels.LaunchQuantizedGemvF32In(devVW, devNormOut, devV, kvDim, HiddenSize, s);

            // 3. RoPE
            _kernels.LaunchRoPEF32(devQ, devK, devPos, 1, NumHeads, NumKvHeads,
                HeadDim, ropeDim, RopeTheta, 0, s);

            // 4. Store new K/V in cache at position 3
            int kvRowBytes = kvDim * sizeof(float);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(
                devCacheK + (nint)(3 * kvRowBytes), devK, (nuint)kvRowBytes, s).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync_v2(
                devCacheV + (nint)(3 * kvRowBytes), devV, (nuint)kvRowBytes, s).ThrowOnError();

            // 5. Attention (1 query, 4 KV from cache)
            _kernels.LaunchAttentionF32(devQ, devCacheK, devCacheV, devAttnOut,
                1, seqKv, NumHeads, NumKvHeads, HeadDim, positions[0], 0, s);

            _stream!.Synchronize();

            // Download intermediate results for debugging
            float[] gpuNormOut = new float[HiddenSize];
            float[] gpuQ = new float[qDim];
            float[] gpuK = new float[kvDim];
            float[] gpuV = new float[kvDim];
            float[] gpuAttnOut = new float[qDim];

            fixed (float* p = gpuNormOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devNormOut, (nuint)hiddenBytes).ThrowOnError();
            fixed (float* p = gpuQ) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devQ, (nuint)qBytes).ThrowOnError();
            fixed (float* p = gpuK) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devK, (nuint)kvBytes).ThrowOnError();
            fixed (float* p = gpuV) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devV, (nuint)kvBytes).ThrowOnError();
            fixed (float* p = gpuAttnOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devAttnOut, (nuint)qBytes).ThrowOnError();

            // Compare at each stage
            CompareResults("Chain-RmsNorm", cpuNormOut, gpuNormOut, tolerance: 0.001f);
            CompareResults("Chain-Q-GEMV", cpuQ, gpuQ, tolerance: 0.05f);
            CompareResults("Chain-K-GEMV", cpuK, gpuK, tolerance: 0.05f);
            CompareResults("Chain-V-GEMV", cpuV, gpuV, tolerance: 0.05f);

            // Compare Q/K after RoPE
            float[] cpuQAfterRope = cpuQ; // already has RoPE applied in-place
            float[] cpuKAfterRope = cpuK;
            CompareResults("Chain-Q-RoPE", cpuQAfterRope, gpuQ, tolerance: 0.05f);
            CompareResults("Chain-K-RoPE", cpuKAfterRope, gpuK, tolerance: 0.05f);

            CompareResults("Chain-Attention", cpuAttnOut, gpuAttnOut, tolerance: 0.1f);
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devHidden);
            CudaDriverApi.cuMemFree_v2(devNormW);
            CudaDriverApi.cuMemFree_v2(devNormOut);
            CudaDriverApi.cuMemFree_v2(devQ);
            CudaDriverApi.cuMemFree_v2(devK);
            CudaDriverApi.cuMemFree_v2(devV);
            CudaDriverApi.cuMemFree_v2(devPos);
            CudaDriverApi.cuMemFree_v2(devCacheK);
            CudaDriverApi.cuMemFree_v2(devCacheV);
            CudaDriverApi.cuMemFree_v2(devAttnOut);
            CudaDriverApi.cuMemFree_v2(devQW);
            CudaDriverApi.cuMemFree_v2(devKW);
            CudaDriverApi.cuMemFree_v2(devVW);
        }
    }

    // ─────────────────── Q8_0 Helpers ───────────────────

    private static byte[] RandomQ8_0(Random rng, int outputDim, int inputDim)
    {
        int blocksPerRow = inputDim / 32;
        int bytesPerRow = blocksPerRow * 34;
        byte[] data = new byte[outputDim * bytesPerRow];

        for (int row = 0; row < outputDim; row++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int offset = row * bytesPerRow + b * 34;
                float scale = (float)(rng.NextDouble() * 0.1 + 0.01);
                ushort scaleF16 = BitConverter.HalfToUInt16Bits((Half)scale);
                data[offset] = (byte)(scaleF16 & 0xFF);
                data[offset + 1] = (byte)(scaleF16 >> 8);
                for (int j = 0; j < 32; j++)
                    data[offset + 2 + j] = (byte)(sbyte)rng.Next(-127, 128);
            }
        }
        return data;
    }

    private static float[,] DequantQ8_0(byte[] data, int outputDim, int inputDim)
    {
        int blocksPerRow = inputDim / 32;
        int bytesPerRow = blocksPerRow * 34;
        float[,] result = new float[outputDim, inputDim];

        for (int row = 0; row < outputDim; row++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int offset = row * bytesPerRow + b * 34;
                ushort scaleF16 = (ushort)(data[offset] | (data[offset + 1] << 8));
                float scale = (float)BitConverter.UInt16BitsToHalf(scaleF16);
                for (int j = 0; j < 32; j++)
                    result[row, b * 32 + j] = scale * (sbyte)data[offset + 2 + j];
            }
        }
        return result;
    }

    // ─────────────────── Helpers ───────────────────

    private void SkipIfUnavailable()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(!_available, "PTX files not found");
    }

    private static float[] RandomF32(Random rng, int count, float scale = 1.0f)
    {
        float[] arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return arr;
    }

    private void CompareResults(string name, float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);

        float maxDiff = 0, sumDiff = 0;
        int maxIdx = 0;
        int mismatchCount = 0;

        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            sumDiff += diff;
            if (diff > maxDiff)
            {
                maxDiff = diff;
                maxIdx = i;
            }
            if (diff > tolerance)
                mismatchCount++;
        }

        float meanDiff = sumDiff / expected.Length;

        _output.WriteLine($"[{name}] n={expected.Length}  maxDiff={maxDiff:E4} @idx={maxIdx}  " +
                          $"meanDiff={meanDiff:E4}  mismatches(>{tolerance})={mismatchCount}/{expected.Length}");

        if (maxDiff > tolerance)
        {
            // Print first 10 mismatches for debugging
            int printed = 0;
            for (int i = 0; i < expected.Length && printed < 10; i++)
            {
                float diff = MathF.Abs(expected[i] - actual[i]);
                if (diff > tolerance)
                {
                    _output.WriteLine($"  [{i}] expected={expected[i]:F6}  gpu={actual[i]:F6}  diff={diff:E4}");
                    printed++;
                }
            }
        }

        Assert.True(mismatchCount == 0,
            $"[{name}] {mismatchCount}/{expected.Length} elements exceed tolerance {tolerance}. " +
            $"maxDiff={maxDiff:E4} at index {maxIdx}, meanDiff={meanDiff:E4}");
    }

    public void Dispose()
    {
        _kernels?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
    }
}
