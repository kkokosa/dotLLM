using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Compares three GEMV computation paths using real model weights to determine
/// whether the ~0.4 logit diff is from Q8_0 input quantization (expected)
/// or from a real kernel bug.
///
/// Path A (CPU actual): MatMul.GemvQ8_0 — quantizes x to Q8_0, then Q8_0×Q8_0 dot
/// Path B (GPU actual): quantized_gemv_q8_0_f32in — Q8_0 weight × F32 input
/// Path C (F32 reference): dequant weight × F32 input (scalar, full precision)
///
/// If |A-C| ≈ |B-C| ≈ 0.4 and |A-B| ≈ 0.4, the diff is from Q8 input quantization.
/// If |B-C| >> |A-C|, there's a GPU kernel bug.
/// </summary>
[Trait("Category", "GPU")]
public class CudaGemvPathComparisonTest
{
    private readonly ITestOutputHelper _out;
    public CudaGemvPathComparisonTest(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    [SkippableFact]
    public unsafe void CompareGemvPaths_RealModelWeights_Layer0_QProjection()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");
        Skip.If(!File.Exists(modelPath), "SmolLM-135M Q8_0 GGUF not found");

        var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);

        ref readonly var lw = ref cpuWeights.Layers[0];
        _out.WriteLine($"Layer 0 Q: quant={lw.QQuantType}, output={lw.QOutputDim}, input={lw.QInputDim}");
        _out.WriteLine($"Layer 0 K: quant={lw.KQuantType}, output={lw.KOutputDim}, input={lw.KInputDim}");

        // Generate a realistic input vector (RmsNorm output — typically small magnitudes)
        var rng = new Random(42);
        float[] x = new float[lw.QInputDim];
        for (int i = 0; i < x.Length; i++)
            x[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;

        int n = lw.QOutputDim, k = lw.QInputDim;

        // === Path A: CPU MatMul.GemvQ8_0 (quantizes x to Q8_0 internally) ===
        float[] pathA = new float[n];
        fixed (float* pX = x, pY = pathA)
            MatMul.GemvQ8_0((byte*)lw.QWeight, pX, pY, n, k);

        // === Path C: F32 scalar reference (dequant weight × F32 input) ===
        float[] pathC = DequantAndGemvReference((byte*)lw.QWeight, x, n, k);

        // === Path B: GPU quantized_gemv_q8_0_f32in ===
        float[] pathB;
        {
            using var ctx = CudaContext.Create(0);
            using var stream = CudaStream.Create();
            string? ptxDir = FindPtxDir();
            Skip.If(ptxDir == null, "PTX files not found");
            using var kernels = new CudaKernels(ptxDir!);

            nint s = stream.Handle;
            long wBytes = Dequantize.RowByteSize(k, QuantizationType.Q8_0) * n;
            long xBytes = (long)k * sizeof(float);
            long yBytes = (long)n * sizeof(float);

            CudaDriverApi.cuMemAlloc_v2(out nint devW, (nuint)wBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devX, (nuint)xBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devY, (nuint)yBytes).ThrowOnError();

            // Upload the SAME weight bytes from mmap to GPU
            CudaDriverApi.cuMemcpyHtoD_v2(devW, lw.QWeight, (nuint)wBytes).ThrowOnError();
            fixed (float* pX = x)
                CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)pX, (nuint)xBytes).ThrowOnError();

            kernels.LaunchQuantizedGemvF32In(devW, devX, devY, n, k, s);
            stream.Synchronize();

            pathB = new float[n];
            fixed (float* pY = pathB)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)pY, devY, (nuint)yBytes).ThrowOnError();

            CudaDriverApi.cuMemFree_v2(devW);
            CudaDriverApi.cuMemFree_v2(devX);
            CudaDriverApi.cuMemFree_v2(devY);
        }

        // === Compare all pairs ===
        var (abMax, abMean) = CompareArrays(pathA, pathB);
        var (acMax, acMean) = CompareArrays(pathA, pathC);
        var (bcMax, bcMean) = CompareArrays(pathB, pathC);

        _out.WriteLine($"\n  |A-B| (CPU vs GPU):      max={abMax:E4}  mean={abMean:E4}");
        _out.WriteLine($"  |A-C| (CPU vs F32ref):   max={acMax:E4}  mean={acMean:E4}");
        _out.WriteLine($"  |B-C| (GPU vs F32ref):   max={bcMax:E4}  mean={bcMean:E4}");
        _out.WriteLine($"\n  If |A-B| ≈ |A-C| >> |B-C|, diff is from CPU's Q8 input quantization.");
        _out.WriteLine($"  If |B-C| >> |A-C|, the GPU kernel has a bug.");

        // Also check: which path is closer to the F32 reference?
        _out.WriteLine($"\n  GPU is closer to F32 ref than CPU: {bcMean < acMean}");
    }

    [SkippableFact]
    public unsafe void CompareGemvPaths_AllProjections_Layer0()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");
        Skip.If(!File.Exists(modelPath), "SmolLM-135M Q8_0 GGUF not found");

        var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);

        var rng = new Random(42);

        ref readonly var lw = ref cpuWeights.Layers[0];
        float[] input576 = RandomF32(rng, config.HiddenSize, 0.5f);
        float[] input1536 = RandomF32(rng, config.IntermediateSize, 0.5f);

        // Test each projection
        TestProjection("Q",    (byte*)lw.QWeight,    lw.QQuantType,    lw.QOutputDim,    lw.QInputDim,    input576, kernels, stream);
        TestProjection("K",    (byte*)lw.KWeight,    lw.KQuantType,    lw.KOutputDim,    lw.KInputDim,    input576, kernels, stream);
        TestProjection("V",    (byte*)lw.VWeight,    lw.VQuantType,    lw.VOutputDim,    lw.VInputDim,    input576, kernels, stream);
        TestProjection("O",    (byte*)lw.OWeight,    lw.OQuantType,    lw.OOutputDim,    lw.OInputDim,    input576, kernels, stream);
        TestProjection("Gate", (byte*)lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim, input576, kernels, stream);
        TestProjection("Up",   (byte*)lw.UpWeight,   lw.UpQuantType,   lw.UpOutputDim,   lw.UpInputDim,   input576, kernels, stream);
        TestProjection("Down", (byte*)lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim, input1536, kernels, stream);

        // Also test LM head
        TestProjection("LMHead", (byte*)cpuWeights.OutputWeight, cpuWeights.OutputQuantType,
            cpuWeights.OutputOutputDim, cpuWeights.OutputInputDim, input576, kernels, stream);

        cpuWeights.Dispose();
    }

    private unsafe void TestProjection(string name, byte* weightPtr, QuantizationType qt,
                                        int n, int k, float[] x, CudaKernels kernels, CudaStream stream)
    {
        if (qt != QuantizationType.Q8_0)
        {
            _out.WriteLine($"[{name}] skipped (quant={qt}, not Q8_0)");
            return;
        }

        // Path A: CPU Q8_0×Q8_0
        float[] pathA = new float[n];
        fixed (float* pX = x, pY = pathA)
            MatMul.GemvQ8_0(weightPtr, pX, pY, n, k);

        // Path C: F32 scalar reference
        float[] pathC = DequantAndGemvReference(weightPtr, x, n, k);

        // Path B: GPU Q8_0×F32
        float[] pathB = RunGpuGemv((nint)weightPtr, x, n, k, kernels, stream);

        var (abMax, abMean) = CompareArrays(pathA, pathB);
        var (acMax, acMean) = CompareArrays(pathA, pathC);
        var (bcMax, bcMean) = CompareArrays(pathB, pathC);

        _out.WriteLine($"[{name}] {n}×{k}  |A-B|={abMean:E3}  |A-C|={acMean:E3}  |B-C|={bcMean:E3}  GPU closer: {bcMean < acMean}");
    }

    private unsafe float[] RunGpuGemv(nint weightHost, float[] x, int n, int k,
                                       CudaKernels kernels, CudaStream stream)
    {
        nint s = stream.Handle;
        long wBytes = Dequantize.RowByteSize(k, QuantizationType.Q8_0) * n;
        long xBytes = (long)k * sizeof(float);
        long yBytes = (long)n * sizeof(float);

        CudaDriverApi.cuMemAlloc_v2(out nint devW, (nuint)wBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devX, (nuint)xBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devY, (nuint)yBytes).ThrowOnError();

        try
        {
            CudaDriverApi.cuMemcpyHtoD_v2(devW, weightHost, (nuint)wBytes).ThrowOnError();
            fixed (float* pX = x)
                CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)pX, (nuint)xBytes).ThrowOnError();

            kernels.LaunchQuantizedGemvF32In(devW, devX, devY, n, k, s);
            stream.Synchronize();

            float[] result = new float[n];
            fixed (float* pY = result)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)pY, devY, (nuint)yBytes).ThrowOnError();
            return result;
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devW);
            CudaDriverApi.cuMemFree_v2(devX);
            CudaDriverApi.cuMemFree_v2(devY);
        }
    }

    /// <summary>
    /// Ground-truth: dequantize Q8_0 weight to F32, then F32 dot product.
    /// </summary>
    private static unsafe float[] DequantAndGemvReference(byte* weight, float[] x, int n, int k)
    {
        int blocksPerRow = k / 32;
        int bytesPerRow = blocksPerRow * 34;
        float[] result = new float[n];

        for (int row = 0; row < n; row++)
        {
            byte* wRow = weight + row * bytesPerRow;
            float acc = 0;
            for (int b = 0; b < blocksPerRow; b++)
            {
                byte* block = wRow + b * 34;
                float d = (float)BitConverter.UInt16BitsToHalf(*(ushort*)block);
                sbyte* qs = (sbyte*)(block + 2);
                for (int j = 0; j < 32; j++)
                    acc += d * qs[j] * x[b * 32 + j];
            }
            result[row] = acc;
        }
        return result;
    }

    private static (float maxDiff, float meanDiff) CompareArrays(float[] a, float[] b)
    {
        float maxDiff = 0, sumDiff = 0;
        for (int i = 0; i < a.Length; i++)
        {
            float diff = MathF.Abs(a[i] - b[i]);
            sumDiff += diff;
            if (diff > maxDiff) maxDiff = diff;
        }
        return (maxDiff, sumDiff / a.Length);
    }

    private static float[] RandomF32(Random rng, int count, float scale)
    {
        float[] arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return arr;
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
}
