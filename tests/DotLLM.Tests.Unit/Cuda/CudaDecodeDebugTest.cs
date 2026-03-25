using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Runs GPU and CPU forward passes on the failing prompt to pinpoint where degeneration starts.
/// Generates 15 tokens and compares logits at each step.
/// </summary>
[Trait("Category", "GPU")]
public class CudaDecodeDebugTest
{
    private readonly ITestOutputHelper _out;
    public CudaDecodeDebugTest(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public unsafe void DebugDecode_FailingPrompt_15Tokens()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");
        Skip.If(!File.Exists(modelPath), "SmolLM-135M Q8_0 GGUF not found");

        var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        // CPU model
        var cpuModel = TransformerModel.LoadFromGguf(gguf, config);
        var cpuKv = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, 64);

        // GPU model
        string ptxDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"));
        var gpuModel = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptxDir);
        var gpuKv = gpuModel.CreateKvCache(64);

        // Failing prompt
        int[] promptTokens = tokenizer.Encode("Tell best meal is");
        _out.WriteLine($"Prompt tokens ({promptTokens.Length}): [{string.Join(", ", promptTokens)}]");

        int[] positions = new int[promptTokens.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;

        // Prefill
        using var cpuLogits = cpuModel.Forward(promptTokens, positions, -1, cpuKv);
        using var gpuLogits = gpuModel.Forward(promptTokens, positions, 0, gpuKv);

        int cpuToken = ArgMax((float*)cpuLogits.DataPointer, config.VocabSize);
        int gpuToken = ArgMax((float*)gpuLogits.DataPointer, config.VocabSize);
        var (maxDiff, meanDiff) = CompareLogitArrays((float*)cpuLogits.DataPointer,
            (float*)gpuLogits.DataPointer, config.VocabSize);

        _out.WriteLine($"Prefill → CPU: {cpuToken} ({tokenizer.Decode([cpuToken])}), " +
                        $"GPU: {gpuToken} ({tokenizer.Decode([gpuToken])})  " +
                        $"maxDiff={maxDiff:F4} meanDiff={meanDiff:F4}");

        // Decode 15 tokens — always force SAME token (CPU's choice) for both
        int nextToken = cpuToken;
        int nextPos = promptTokens.Length;

        for (int step = 0; step < 15; step++)
        {
            using var cpuL = cpuModel.Forward([nextToken], [nextPos], -1, cpuKv);
            using var gpuL = gpuModel.Forward([nextToken], [nextPos], 0, gpuKv);

            int cpuT = ArgMax((float*)cpuL.DataPointer, config.VocabSize);
            int gpuT = ArgMax((float*)gpuL.DataPointer, config.VocabSize);
            var (md, ad) = CompareLogitArrays((float*)cpuL.DataPointer,
                (float*)gpuL.DataPointer, config.VocabSize);

            string cpuStr = tokenizer.Decode([cpuT]);
            string gpuStr = tokenizer.Decode([gpuT]);
            string marker = cpuT != gpuT ? " *** DIVERGED ***" : "";

            _out.WriteLine($"Decode[{step}] pos={nextPos} → CPU: {cpuT} ({cpuStr}), " +
                            $"GPU: {gpuT} ({gpuStr})  maxDiff={md:F4} meanDiff={ad:F4}{marker}");

            // Dump top-5 when they diverge
            if (cpuT != gpuT)
            {
                var cpuTop = TopK((float*)cpuL.DataPointer, config.VocabSize, 5);
                var gpuTop = TopK((float*)gpuL.DataPointer, config.VocabSize, 5);
                _out.WriteLine("  CPU top-5: " + string.Join(", ", cpuTop.Select(t => $"[{t.idx}]={t.val:F3}")));
                _out.WriteLine("  GPU top-5: " + string.Join(", ", gpuTop.Select(t => $"[{t.idx}]={t.val:F3}")));
            }

            nextToken = cpuT; // Force CPU token for next step
            nextPos++;
        }

        cpuModel.Dispose();
        gpuModel.Dispose();
        cpuKv.Dispose();
        gpuKv.Dispose();
    }

    private static unsafe (float maxDiff, float meanDiff) CompareLogitArrays(float* a, float* b, int n)
    {
        float maxDiff = 0, sumDiff = 0;
        for (int i = 0; i < n; i++)
        {
            float diff = MathF.Abs(a[i] - b[i]);
            sumDiff += diff;
            if (diff > maxDiff) maxDiff = diff;
        }
        return (maxDiff, sumDiff / n);
    }

    private static unsafe (int idx, float val)[] TopK(float* data, int n, int k)
    {
        var top = new (int idx, float val)[k];
        for (int i = 0; i < k; i++) top[i] = (-1, float.MinValue);
        for (int i = 0; i < n; i++)
        {
            if (data[i] > top[k - 1].val)
            {
                top[k - 1] = (i, data[i]);
                Array.Sort(top, (a, b) => b.val.CompareTo(a.val));
            }
        }
        return top;
    }

    private static unsafe int ArgMax(float* data, int n)
    {
        int best = 0;
        for (int i = 1; i < n; i++)
            if (data[i] > data[best]) best = i;
        return best;
    }
}
