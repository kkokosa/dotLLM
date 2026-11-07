using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Tests for fused QKV tensor split logic -- verifies that byte offset calculations
/// correctly partition a fused [Q|K|V] weight tensor into separate Q, K, V components.
/// </summary>
public class FusedQkvSplitTests
{
    /// <summary>
    /// Verifies RowByteSize computation for common quantization types (used in fused QKV pointer arithmetic).
    /// </summary>
    [Theory]
    [InlineData(512, QuantizationType.F32, 2048)]
    [InlineData(256, QuantizationType.F16, 512)]
    [InlineData(256, QuantizationType.Q8_0, 256L / 32 * 34)]
    public void RowByteSize_ComputesCorrectly(long elementCount, QuantizationType qt, long expected)
    {
        long actual = Dequantize.RowByteSize(elementCount, qt);
        Assert.Equal(expected, actual);
    }

    /// <summary>
    /// Verifies that fused QKV offset arithmetic produces correct non-overlapping regions.
    /// For a model with 8 heads, 2 KV heads, headDim=64:
    ///   Q dim = 8*64 = 512, K dim = 2*64 = 128, V dim = 2*64 = 128
    ///   Total rows = 512 + 128 + 128 = 768
    /// </summary>
    [Fact]
    public void FusedQkvOffsets_F32_AreCorrect()
    {
        int numHeads = 8, numKvHeads = 2, headDim = 64;
        int inputDim = 512; // hidden_size
        var qt = QuantizationType.F32;
        long rowBytes = Dequantize.RowByteSize(inputDim, qt); // 512 * 4 = 2048

        int qDim = numHeads * headDim;     // 512
        int kvDim = numKvHeads * headDim;   // 128

        // Q starts at offset 0
        long qOffset = 0;
        // K starts after Q rows
        long kOffset = qDim * rowBytes;     // 512 * 2048 = 1,048,576
        // V starts after Q + K rows
        long vOffset = (qDim + kvDim) * rowBytes; // 640 * 2048 = 1,310,720

        Assert.Equal(0, qOffset);
        Assert.Equal(1_048_576, kOffset);
        Assert.Equal(1_310_720, vOffset);

        // Total = (512 + 128 + 128) * 2048 = 768 * 2048 = 1,572,864
        long totalBytes = (qDim + 2 * kvDim) * rowBytes;
        Assert.Equal(1_572_864, totalBytes);

        // No overlaps
        Assert.True(kOffset > qOffset);
        Assert.True(vOffset > kOffset);
        Assert.True(kOffset >= qDim * rowBytes);
        Assert.True(vOffset >= (qDim + kvDim) * rowBytes);
    }

    /// <summary>
    /// Verifies fused QKV bias split: Q bias [qDim], K bias [kvDim], V bias [kvDim]
    /// are contiguous in the fused bias array.
    /// </summary>
    [Fact]
    public void FusedQkvBiasOffsets_AreCorrect()
    {
        int numHeads = 8, numKvHeads = 2, headDim = 64;
        int qDim = numHeads * headDim;     // 512
        int kvDim = numKvHeads * headDim;   // 128

        // Bias is F32: element offsets (in bytes) for sizeof(float)
        long qBiasOffset = 0;
        long kBiasOffset = qDim * sizeof(float);          // 512 * 4 = 2048
        long vBiasOffset = (qDim + kvDim) * sizeof(float); // 640 * 4 = 2560

        Assert.Equal(0, qBiasOffset);
        Assert.Equal(2048, kBiasOffset);
        Assert.Equal(2560, vBiasOffset);

        long totalBiasBytes = (qDim + 2 * kvDim) * sizeof(float); // 768 * 4 = 3072
        Assert.Equal(3072, totalBiasBytes);
    }

    /// <summary>
    /// Verifies fused QKV with Q8_0 quantized weights: row byte size uses block layout.
    /// </summary>
    [Fact]
    public void FusedQkvOffsets_Q8_0_AreCorrect()
    {
        int numHeads = 4, numKvHeads = 2, headDim = 32;
        int inputDim = 128; // hidden_size, must be divisible by 32 for Q8_0
        var qt = QuantizationType.Q8_0;
        long rowBytes = Dequantize.RowByteSize(inputDim, qt); // 128/32 * 34 = 136

        int qDim = numHeads * headDim;     // 128
        int kvDim = numKvHeads * headDim;   // 64

        long kOffset = qDim * rowBytes;             // 128 * 136 = 17,408
        long vOffset = (qDim + kvDim) * rowBytes;   // 192 * 136 = 26,112

        Assert.Equal(17_408, kOffset);
        Assert.Equal(26_112, vOffset);
    }

    [Fact]
    public unsafe void TransformerModel_Forward_LastLogitOnly_MatchesLastRowAndSavesLogitScratch()
    {
        const int vocabSize = 16;
        const int hiddenSize = 8;
        const int numHeads = 2;
        const int numKvHeads = 2;
        const int headDim = 4;
        const int intermediateSize = 16;

        static byte[] MakeF32Bytes(int count, int seed)
        {
            var rng = new Random(seed);
            float[] vals = new float[count];
            for (int i = 0; i < count; i++)
                vals[i] = (rng.NextSingle() - 0.5f) * 0.2f;
            byte[] bytes = new byte[count * sizeof(float)];
            Buffer.BlockCopy(vals, 0, bytes, 0, bytes.Length);
            return bytes;
        }

        var ggufData = new GgufTestData(version: 3)
            .AddString("general.architecture", "llama")
            .AddTensor("token_embd.weight", [hiddenSize, vocabSize], 0u, MakeF32Bytes(hiddenSize * vocabSize, 1))
            .AddTensor("output_norm.weight", [hiddenSize], 0u, MakeF32Bytes(hiddenSize, 2))
            .AddTensor("output.weight", [hiddenSize, vocabSize], 0u, MakeF32Bytes(hiddenSize * vocabSize, 3))
            .AddTensor("blk.0.attn_norm.weight", [hiddenSize], 0u, MakeF32Bytes(hiddenSize, 4))
            .AddTensor("blk.0.attn_q.weight", [hiddenSize, numHeads * headDim], 0u, MakeF32Bytes(hiddenSize * numHeads * headDim, 5))
            .AddTensor("blk.0.attn_k.weight", [hiddenSize, numKvHeads * headDim], 0u, MakeF32Bytes(hiddenSize * numKvHeads * headDim, 6))
            .AddTensor("blk.0.attn_v.weight", [hiddenSize, numKvHeads * headDim], 0u, MakeF32Bytes(hiddenSize * numKvHeads * headDim, 7))
            .AddTensor("blk.0.attn_output.weight", [numHeads * headDim, hiddenSize], 0u, MakeF32Bytes(numHeads * headDim * hiddenSize, 8))
            .AddTensor("blk.0.ffn_norm.weight", [hiddenSize], 0u, MakeF32Bytes(hiddenSize, 9))
            .AddTensor("blk.0.ffn_gate.weight", [hiddenSize, intermediateSize], 0u, MakeF32Bytes(hiddenSize * intermediateSize, 10))
            .AddTensor("blk.0.ffn_up.weight", [hiddenSize, intermediateSize], 0u, MakeF32Bytes(hiddenSize * intermediateSize, 11))
            .AddTensor("blk.0.ffn_down.weight", [intermediateSize, hiddenSize], 0u, MakeF32Bytes(intermediateSize * hiddenSize, 12));

        string tempPath = ggufData.WriteToTempFile();
        try
        {
            var config = new ModelConfig
            {
                Architecture = Architecture.Llama,
                VocabSize = vocabSize,
                HiddenSize = hiddenSize,
                NumLayers = 1,
                NumAttentionHeads = numHeads,
                NumKvHeads = numKvHeads,
                HeadDim = headDim,
                IntermediateSize = intermediateSize,
                MaxSequenceLength = 64,
                NormEpsilon = 1e-5f,
            };

            using var ggufAll = GgufFile.Open(tempPath);
            using var modelAll = TransformerModel.LoadFromGguf(ggufAll, config);

            using var ggufLast = GgufFile.Open(tempPath);
            using var modelLast = TransformerModel.LoadFromGguf(ggufLast, config);

            int[] tokens = [1, 3, 5, 7, 9, 11, 13, 15];
            int[] positions = [0, 1, 2, 3, 4, 5, 6, 7];

            using var allLogits = modelAll.Forward(tokens, positions, -1, kvCache: null, lastLogitOnly: false);
            using var lastLogits = modelLast.Forward(tokens, positions, -1, kvCache: null, lastLogitOnly: true);

            Assert.Equal(tokens.Length, allLogits.Shape[0]);
            Assert.Equal(vocabSize, allLogits.Shape[1]);
            Assert.Equal(1, lastLogits.Shape[0]);
            Assert.Equal(vocabSize, lastLogits.Shape[1]);

            var allLastRow = new ReadOnlySpan<float>((float*)allLogits.DataPointer + (tokens.Length - 1) * vocabSize, vocabSize);
            var onlyLastRow = new ReadOnlySpan<float>((float*)lastLogits.DataPointer, vocabSize);

            for (int v = 0; v < vocabSize; v++)
                Assert.Equal(allLastRow[v], onlyLastRow[v], 1e-6f);

            long expectedSavedBytes = (long)(tokens.Length - 1) * vocabSize * sizeof(float);
            Assert.Equal(expectedSavedBytes, modelAll.ComputeMemoryBytes - modelLast.ComputeMemoryBytes);

            // Also verify that calling lastLogitOnly: false (e.g., SpeculativeDecoder with 4 tokens)
            // on the SAME modelLast instance after a larger lastLogitOnly: true prefill (seqLen=8, logitRows=1)
            // grows the Logits buffer to 4 rows even though _currentSeqLen (8) >= 4.
            int[] verifyTokens = [2, 4, 6, 8];
            int[] verifyPos = [0, 1, 2, 3];
            using var verifyLogits = modelLast.Forward(verifyTokens, verifyPos, -1, kvCache: null, lastLogitOnly: false);
            Assert.Equal(4, verifyLogits.Shape[0]);
            Assert.Equal(vocabSize, verifyLogits.Shape[1]);
        }
        finally
        {
            try { File.Delete(tempPath); } catch { /* best-effort */ }
        }
    }

    [Fact]
    public unsafe void TransformerModel_Forward_LastLogitOnly_Q8_0_RepackedOutputAndKvCache_MatchFullPrefill()
    {
        // Q8_0 requires input dimensions divisible by 32 and OutputDim divisible by 4 for R4 repacking.
        const int vocabSize = 32;
        const int hiddenSize = 32;
        const int numHeads = 2;
        const int numKvHeads = 2;
        const int headDim = 16;
        const int intermediateSize = 32;
        const uint ggmlTypeQ8_0 = 8u; // QuantizationType.Q8_0

        static byte[] MakeF32Bytes(int count, int seed)
        {
            var rng = new Random(seed);
            float[] vals = new float[count];
            for (int i = 0; i < count; i++)
                vals[i] = 0.8f + (rng.NextSingle() * 0.4f);
            byte[] bytes = new byte[count * sizeof(float)];
            Buffer.BlockCopy(vals, 0, bytes, 0, bytes.Length);
            return bytes;
        }

        static byte[] MakeQ8_0Bytes(int rows, int cols, int seed)
        {
            var rng = new Random(seed);
            int blocksPerRow = cols / 32;
            byte[] bytes = new byte[rows * blocksPerRow * 34];
            ushort scaleBits = BitConverter.HalfToUInt16Bits((Half)0.02f);
            int offset = 0;
            for (int r = 0; r < rows; r++)
            {
                for (int b = 0; b < blocksPerRow; b++)
                {
                    bytes[offset] = (byte)(scaleBits & 0xFF);
                    bytes[offset + 1] = (byte)(scaleBits >> 8);
                    for (int i = 0; i < 32; i++)
                        bytes[offset + 2 + i] = (byte)(sbyte)rng.Next(-16, 17);
                    offset += 34;
                }
            }
            return bytes;
        }

        var ggufData = new GgufTestData(version: 3)
            .AddString("general.architecture", "llama")
            .AddTensor("token_embd.weight", [hiddenSize, vocabSize], ggmlTypeQ8_0, MakeQ8_0Bytes(vocabSize, hiddenSize, 1))
            .AddTensor("output_norm.weight", [hiddenSize], 0u, MakeF32Bytes(hiddenSize, 2))
            .AddTensor("output.weight", [hiddenSize, vocabSize], ggmlTypeQ8_0, MakeQ8_0Bytes(vocabSize, hiddenSize, 3))
            .AddTensor("blk.0.attn_norm.weight", [hiddenSize], 0u, MakeF32Bytes(hiddenSize, 4))
            .AddTensor("blk.0.attn_q.weight", [hiddenSize, numHeads * headDim], ggmlTypeQ8_0, MakeQ8_0Bytes(numHeads * headDim, hiddenSize, 5))
            .AddTensor("blk.0.attn_k.weight", [hiddenSize, numKvHeads * headDim], ggmlTypeQ8_0, MakeQ8_0Bytes(numKvHeads * headDim, hiddenSize, 6))
            .AddTensor("blk.0.attn_v.weight", [hiddenSize, numKvHeads * headDim], ggmlTypeQ8_0, MakeQ8_0Bytes(numKvHeads * headDim, hiddenSize, 7))
            .AddTensor("blk.0.attn_output.weight", [numHeads * headDim, hiddenSize], ggmlTypeQ8_0, MakeQ8_0Bytes(hiddenSize, numHeads * headDim, 8))
            .AddTensor("blk.0.ffn_norm.weight", [hiddenSize], 0u, MakeF32Bytes(hiddenSize, 9))
            .AddTensor("blk.0.ffn_gate.weight", [hiddenSize, intermediateSize], ggmlTypeQ8_0, MakeQ8_0Bytes(intermediateSize, hiddenSize, 10))
            .AddTensor("blk.0.ffn_up.weight", [hiddenSize, intermediateSize], ggmlTypeQ8_0, MakeQ8_0Bytes(intermediateSize, hiddenSize, 11))
            .AddTensor("blk.0.ffn_down.weight", [intermediateSize, hiddenSize], ggmlTypeQ8_0, MakeQ8_0Bytes(hiddenSize, intermediateSize, 12));

        string tempPath = ggufData.WriteToTempFile();
        try
        {
            var config = new ModelConfig
            {
                Architecture = Architecture.Llama,
                VocabSize = vocabSize,
                HiddenSize = hiddenSize,
                NumLayers = 1,
                NumAttentionHeads = numHeads,
                NumKvHeads = numKvHeads,
                HeadDim = headDim,
                IntermediateSize = intermediateSize,
                MaxSequenceLength = 64,
                NormEpsilon = 1e-5f,
            };

            using var ggufAll = GgufFile.Open(tempPath);
            using var modelAll = TransformerModel.LoadFromGguf(ggufAll, config);
            using var kvAll = new DotLLM.Engine.KvCache.SimpleKvCache(1, numKvHeads, headDim, 64);

            using var ggufLast = GgufFile.Open(tempPath);
            using var modelLast = TransformerModel.LoadFromGguf(ggufLast, config);
            using var kvLast = new DotLLM.Engine.KvCache.SimpleKvCache(1, numKvHeads, headDim, 64);

            int[] promptTokens = [1, 4, 7, 10, 13, 16];
            int[] promptPositions = [0, 1, 2, 3, 4, 5];

            // Prefill: modelAll uses n=6 GemmR4TiledQ8_0 on OutputWeight, modelLast uses n=1 GemvInterleaved on RepackedOutput.
            using var prefillAll = modelAll.Forward(promptTokens, promptPositions, -1, kvAll, lastLogitOnly: false);
            using var prefillLast = modelLast.Forward(promptTokens, promptPositions, -1, kvLast, lastLogitOnly: true);

            var allLastRow = new ReadOnlySpan<float>((float*)prefillAll.DataPointer + (promptTokens.Length - 1) * vocabSize, vocabSize);
            var onlyLastRow = new ReadOnlySpan<float>((float*)prefillLast.DataPointer, vocabSize);
            for (int v = 0; v < vocabSize; v++)
                Assert.Equal(allLastRow[v], onlyLastRow[v], 1e-4f);

            // Subsequent decode step at pos=6 must see identical KV cache state in both models.
            int[] nextToken = [19];
            int[] nextPos = [6];
            using var decodeAll = modelAll.Forward(nextToken, nextPos, -1, kvAll, lastLogitOnly: true);
            using var decodeLast = modelLast.Forward(nextToken, nextPos, -1, kvLast, lastLogitOnly: true);

            var decodeAllRow = new ReadOnlySpan<float>((float*)decodeAll.DataPointer, vocabSize);
            var decodeLastRow = new ReadOnlySpan<float>((float*)decodeLast.DataPointer, vocabSize);
            for (int v = 0; v < vocabSize; v++)
                Assert.Equal(decodeAllRow[v], decodeLastRow[v], 1e-5f);
        }
        finally
        {
            try { File.Delete(tempPath); } catch { /* best-effort */ }
        }
    }
}
