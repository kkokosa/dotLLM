using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Compares CPU vs GPU logits token by token to diagnose divergence.
/// </summary>
[Trait("Category", "GPU")]
public class CudaLogitComparisonTest
{
    private readonly ITestOutputHelper _out;

    public CudaLogitComparisonTest(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public unsafe void CompareLogits_PrefillAndDecode()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");
        string? ggufPath = File.Exists(modelPath) ? modelPath : null;
        Skip.If(ggufPath == null, "SmolLM-135M Q8_0 GGUF not found (run: dotllm run QuantFactory/SmolLM-135M-GGUF -q Q8_0)");

        var gguf = GgufFile.Open(ggufPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        _out.WriteLine($"Model: {config.Architecture} {config.NumLayers}L/{config.HiddenSize}H");

        // Load CPU model
        var cpuModel = TransformerModel.LoadFromGguf(gguf, config);
        var cpuKv = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, 64);

        // Load GPU model
        string ptxDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"));
        var gpuModel = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptxDir);
        var gpuKv = gpuModel.CreateKvCache(64);

        // Prompt tokens: "The capital of France is" for SmolLM
        int[] promptTokens = [510, 5765, 302, 6181, 349]; // approximate; use actual tokenizer
        var tokenizer = DotLLM.Models.Gguf.GgufBpeTokenizerFactory.Load(gguf.Metadata);
        promptTokens = tokenizer.Encode("The capital of France is");
        _out.WriteLine($"Prompt tokens ({promptTokens.Length}): [{string.Join(", ", promptTokens)}]");

        int[] positions = new int[promptTokens.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;

        // Step 1: Prefill
        using var cpuLogits1 = cpuModel.Forward(promptTokens, positions, -1, cpuKv);
        using var gpuLogits1 = gpuModel.Forward(promptTokens, positions, 0, gpuKv);

        CompareLogits("Prefill", cpuLogits1, gpuLogits1, config.VocabSize);

        // Sample greedy from CPU
        int token1 = ArgMax((float*)cpuLogits1.DataPointer, config.VocabSize);
        int gpuToken1 = ArgMax((float*)gpuLogits1.DataPointer, config.VocabSize);
        _out.WriteLine($"Prefill → CPU token: {token1} ({tokenizer.Decode([token1])}), GPU token: {gpuToken1} ({tokenizer.Decode([gpuToken1])})");

        // Step 2: First decode — FORCE same token for both
        int pos1 = promptTokens.Length;
        using var cpuLogits2 = cpuModel.Forward([token1], [pos1], -1, cpuKv);
        using var gpuLogits2 = gpuModel.Forward([token1], [pos1], 0, gpuKv); // use CPU token for GPU too

        CompareLogits("Decode1 (same input token)", cpuLogits2, gpuLogits2, config.VocabSize);

        int token2cpu = ArgMax((float*)cpuLogits2.DataPointer, config.VocabSize);
        int token2gpu = ArgMax((float*)gpuLogits2.DataPointer, config.VocabSize);
        _out.WriteLine($"Decode1 → CPU token: {token2cpu} ({tokenizer.Decode([token2cpu])}), GPU token: {token2gpu} ({tokenizer.Decode([token2gpu])})");

        // Step 3: Second decode — FORCE same token for both
        int pos2 = pos1 + 1;
        int forcedToken2 = token2cpu;
        using var cpuLogits3 = cpuModel.Forward([forcedToken2], [pos2], -1, cpuKv);
        using var gpuLogits3 = gpuModel.Forward([forcedToken2], [pos2], 0, gpuKv);

        CompareLogits("Decode2 (forced same token)", cpuLogits3, gpuLogits3, config.VocabSize);

        int token3cpu = ArgMax((float*)cpuLogits3.DataPointer, config.VocabSize);
        int token3gpu = ArgMax((float*)gpuLogits3.DataPointer, config.VocabSize);
        _out.WriteLine($"Decode2 → CPU token: {token3cpu} ({tokenizer.Decode([token3cpu])}), GPU token: {token3gpu} ({tokenizer.Decode([token3gpu])})");

        cpuModel.Dispose();
        gpuModel.Dispose();
        cpuKv.Dispose();
        gpuKv.Dispose();
    }

    private unsafe void CompareLogits(string step, ITensor cpuLogits, ITensor gpuLogits, int vocabSize)
    {
        float* cpu = (float*)cpuLogits.DataPointer;
        float* gpu = (float*)gpuLogits.DataPointer;

        // Find top-5 for both
        var cpuTop = TopK(cpu, vocabSize, 5);
        var gpuTop = TopK(gpu, vocabSize, 5);

        _out.WriteLine($"\n--- {step} ---");
        _out.WriteLine("CPU top-5: " + string.Join(", ", cpuTop.Select(t => $"[{t.idx}]={t.val:F3}")));
        _out.WriteLine("GPU top-5: " + string.Join(", ", gpuTop.Select(t => $"[{t.idx}]={t.val:F3}")));

        // Compute max absolute diff and mean absolute diff
        float maxDiff = 0, sumDiff = 0;
        int maxDiffIdx = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            float diff = MathF.Abs(cpu[i] - gpu[i]);
            sumDiff += diff;
            if (diff > maxDiff) { maxDiff = diff; maxDiffIdx = i; }
        }
        _out.WriteLine($"Max diff: {maxDiff:F4} at [{maxDiffIdx}], Mean diff: {sumDiff / vocabSize:F4}");
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
