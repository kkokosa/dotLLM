using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CPU-only prefill sanity check. Addresses the indexing foot-gun in
/// <see cref="CudaLogitComparisonTest"/>: for a multi-token prefill,
/// <see cref="TransformerModel.Forward"/> returns logits for ALL positions
/// (shape [seqLen, vocabSize]) but the CUDA model returns logits for the
/// last position only (shape [1, vocabSize]). Reading offset 0 on both
/// therefore compares CPU's position-0 logits against GPU's position-(N-1)
/// logits — apples vs oranges, not a real numerical divergence.
///
/// This test asserts the CPU's LAST-token logits for "The capital of France is"
/// pick token 7042 (" Paris"), i.e. the same token GPU picks at the same
/// position. It requires only a CPU and the SmolLM-135M Q4_K_M GGUF.
/// </summary>
public class CpuPrefillLastTokenLogitTest
{
    private readonly ITestOutputHelper _out;

    public CpuPrefillLastTokenLogitTest(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public unsafe void CpuPrefill_LastTokenArgmax_IsParis()
    {
        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q4_K_M.gguf");
        Skip.If(!File.Exists(modelPath),
            "SmolLM-135M.Q4_K_M.gguf not found (run: dotllm run QuantFactory/SmolLM-135M-GGUF)");

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] promptTokens = tokenizer.Encode("The capital of France is");
        int[] positions = new int[promptTokens.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;

        using var model = TransformerModel.LoadFromGguf(gguf, config);
        using var kv = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, 64);

        using var logits = model.Forward(promptTokens, positions, -1, kv);

        int vocab = config.VocabSize;
        int seqLen = promptTokens.Length;

        // CPU Forward returns [seqLen, vocabSize]. Greedy-decode from the LAST position.
        float* last = (float*)logits.DataPointer + (long)(seqLen - 1) * vocab;
        int best = 0;
        float bestVal = last[0];
        for (int i = 1; i < vocab; i++)
            if (last[i] > bestVal) { bestVal = last[i]; best = i; }

        // Diagnostics: first-token argmax too, so a failure tells us what
        // confused the original CudaLogitComparisonTest.
        float* first = (float*)logits.DataPointer;
        int firstBest = 0; float firstBestVal = first[0];
        for (int i = 1; i < vocab; i++)
            if (first[i] > firstBestVal) { firstBestVal = first[i]; firstBest = i; }

        string decodedFirst = tokenizer.Decode([firstBest]);
        string decodedLast = tokenizer.Decode([best]);
        _out.WriteLine($"CPU prefill: position-0 argmax = {firstBest} ('{decodedFirst}'), " +
                       $"position-{seqLen - 1} argmax = {best} ('{decodedLast}'), logit = {bestVal:F3}");

        // 7042 = " Paris" in SmolLM's tokenizer.
        Assert.Equal(7042, best);
    }
}
