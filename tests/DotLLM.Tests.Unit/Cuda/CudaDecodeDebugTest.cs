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
    public unsafe void DebugDecode_SmolLM_15Tokens()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");
        Skip.If(!File.Exists(modelPath), "SmolLM-135M Q8_0 GGUF not found");

        RunDecodeComparison(modelPath, "Tell best meal is", 15);
    }

    [SkippableFact]
    public unsafe void DebugDecode_Qwen25_15Tokens()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf");
        Skip.If(!File.Exists(modelPath), "Qwen2.5-0.5B-Instruct Q8_0 GGUF not found");

        RunDecodeComparison(modelPath, "The capital of France is", 15);
    }

    [SkippableFact]
    public unsafe void DebugDecode_Qwen3_15Tokens()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "Qwen", "Qwen3-0.6B-GGUF", "qwen3-0.6b-q8_0.gguf");
        Skip.If(!File.Exists(modelPath), "Qwen3-0.6B Q8_0 GGUF not found");

        RunDecodeComparison(modelPath, "The capital of France is", 15);
    }

    /// <summary>
    /// Llama-3.2-1B: larger model, check if per-layer error is similar to Qwen.
    /// </summary>
    [SkippableFact]
    public unsafe void DebugDecode_Llama1B_LayerBisect()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf");
        Skip.If(!File.Exists(modelPath), "Llama-3.2-1B-Instruct Q8_0 GGUF not found");

        RunLayerBisect(modelPath, "The capital of France is");
    }

    /// <summary>
    /// Baseline: SmolLM layer bisect for comparison against Qwen.
    /// </summary>
    [SkippableFact]
    public unsafe void DebugDecode_SmolLM_LayerBisect()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");
        Skip.If(!File.Exists(modelPath), "SmolLM-135M Q8_0 GGUF not found");

        RunLayerBisect(modelPath, "The capital of France is");
    }

    /// <summary>
    /// Bisects which layer first causes divergence by running 1..N layers on both CPU and GPU.
    /// </summary>
    [SkippableFact]
    public unsafe void DebugDecode_Qwen25_LayerBisect()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf");
        Skip.If(!File.Exists(modelPath), "Qwen2.5-0.5B-Instruct Q8_0 GGUF not found");

        RunLayerBisect(modelPath, "The capital of France is");
    }

    /// <summary>
    /// Isolates whether the error comes from NeoX RoPE, Q/K biases, or both.
    /// Runs GPU with each feature disabled independently and compares error vs baseline.
    /// </summary>
    [SkippableFact]
    public unsafe void DebugDecode_Qwen25_FeatureIsolation()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf");
        Skip.If(!File.Exists(modelPath), "Qwen2.5-0.5B-Instruct Q8_0 GGUF not found");

        var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        int[] promptTokens = GgufBpeTokenizerFactory.Load(gguf.Metadata).Encode("The capital of France is");
        int[] positions = new int[promptTokens.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;

        string ptxDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"));

        // Baseline: full GPU (NeoX RoPE + biases)
        RunFeatureTest("Baseline (NeoX+bias)", gguf, config, promptTokens, positions, ptxDir,
            ropeOverride: -1, skipBias: false);
        // Test 1: Force Norm RoPE (wrong results, but shows error contribution of RoPE type)
        RunFeatureTest("Force Norm RoPE", gguf, config, promptTokens, positions, ptxDir,
            ropeOverride: 0, skipBias: false);
        // Test 2: Skip biases (wrong results, but shows error contribution of biases)
        RunFeatureTest("Skip biases", gguf, config, promptTokens, positions, ptxDir,
            ropeOverride: -1, skipBias: true);
        // Test 3: Both disabled
        RunFeatureTest("Norm RoPE + no bias", gguf, config, promptTokens, positions, ptxDir,
            ropeOverride: 0, skipBias: true);
    }

    private unsafe void RunFeatureTest(string label, GgufFile gguf, ModelConfig config,
        int[] promptTokens, int[] positions, string ptxDir, int ropeOverride, bool skipBias)
    {
        var cpuModel = TransformerModel.LoadFromGguf(gguf, config);
        cpuModel.DebugMaxLayers = 1;

        var gpuModel = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptxDir);
        gpuModel.DebugMaxLayers = 1;
        gpuModel.DebugRopeTypeOverride = ropeOverride;
        gpuModel.DebugSkipBias = skipBias;

        using var cpuLogits = cpuModel.Forward(promptTokens, positions, -1);
        using var gpuLogits = gpuModel.Forward(promptTokens, positions, 0);

        var (maxDiff, meanDiff) = CompareLogitArrays((float*)cpuLogits.DataPointer,
            (float*)gpuLogits.DataPointer, config.VocabSize);

        _out.WriteLine($"{label,-25} → maxDiff={maxDiff:F4} meanDiff={meanDiff:F4}");

        cpuModel.Dispose();
        gpuModel.Dispose();
    }

    private unsafe void RunLayerBisect(string modelPath, string prompt)
    {
        var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        _out.WriteLine($"Model: {Path.GetFileName(modelPath)}");
        _out.WriteLine($"Config: {config.Architecture} {config.NumLayers}L/{config.HiddenSize}H " +
                       $"heads={config.NumAttentionHeads} kvHeads={config.NumKvHeads} headDim={config.HeadDim}");
        _out.WriteLine($"RoPE: type={config.RoPEConfig?.Type} dim={config.RoPEConfig?.DimensionCount} theta={config.RoPEConfig?.Theta}");

        int[] promptTokens = tokenizer.Encode(prompt);
        int[] positions = new int[promptTokens.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;

        string ptxDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"));

        // -1 = embedding+LM head only, then layers 1..6 (fine-grained), then 8, 12, 16, 20, 24
        foreach (int maxLayers in new[] { -1, 1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 24 })
        {
            if (maxLayers > config.NumLayers) break;

            var cpuModel = TransformerModel.LoadFromGguf(gguf, config);
            cpuModel.DebugMaxLayers = maxLayers;
            var gpuModel = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptxDir);
            gpuModel.DebugMaxLayers = maxLayers;

            using var cpuLogits = cpuModel.Forward(promptTokens, positions, -1);
            using var gpuLogits = gpuModel.Forward(promptTokens, positions, 0);

            int cpuToken = ArgMax((float*)cpuLogits.DataPointer, config.VocabSize);
            int gpuToken = ArgMax((float*)gpuLogits.DataPointer, config.VocabSize);
            var (maxDiff, meanDiff) = CompareLogitArrays((float*)cpuLogits.DataPointer,
                (float*)gpuLogits.DataPointer, config.VocabSize);

            string marker = cpuToken != gpuToken ? " *** DIVERGED ***" : "";
            _out.WriteLine($"Layers={maxLayers:D2} → CPU:{cpuToken} GPU:{gpuToken}  " +
                            $"maxDiff={maxDiff:F4} meanDiff={meanDiff:F4}{marker}");

            cpuModel.Dispose();
            gpuModel.Dispose();
        }
    }

    private unsafe void RunDecodeComparison(string modelPath, string prompt, int decodeSteps)
    {
        var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        _out.WriteLine($"Model: {Path.GetFileName(modelPath)}");
        _out.WriteLine($"Config: {config.Architecture} {config.NumLayers}L/{config.HiddenSize}H " +
                       $"heads={config.NumAttentionHeads} kvHeads={config.NumKvHeads} headDim={config.HeadDim}");
        _out.WriteLine($"RoPE: type={config.RoPEConfig?.Type} dim={config.RoPEConfig?.DimensionCount} theta={config.RoPEConfig?.Theta}");

        // CPU model
        var cpuModel = TransformerModel.LoadFromGguf(gguf, config);
        var cpuKv = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, 64);

        // GPU model
        string ptxDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"));
        var gpuModel = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptxDir);
        var gpuKv = gpuModel.CreateKvCache(64);

        int[] promptTokens = tokenizer.Encode(prompt);
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

        // Decode tokens — always force SAME token (CPU's choice) for both
        int nextToken = cpuToken;
        int nextPos = promptTokens.Length;

        for (int step = 0; step < decodeSteps; step++)
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
