using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Llama-family forward pass: embedding lookup → N × transformer blocks → final norm → LM head → logits.
/// Operates entirely on the CPU using pre-allocated scratch buffers for zero-allocation inference.
/// </summary>
public sealed unsafe class LlamaModel : IModel
{
    /// <summary>Q8_0 block: 2 bytes (Half scale) + 32 bytes (sbyte values).</summary>
    private const int Q8_0BlockBytes = 34;

    /// <summary>Elements per Q8_0 block.</summary>
    private const int Q8_0GroupSize = 32;

    private readonly LlamaWeights _weights;
    private readonly LlamaForwardState _state;
    private readonly GgufFile _gguf; // prevent premature GC of mmap
    private readonly int _ropeDim;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <summary>Total bytes allocated for inference scratch buffers.</summary>
    public long ComputeMemoryBytes => _state.AllocatedBytes;

    private LlamaModel(ModelConfig config, LlamaWeights weights, LlamaForwardState state, GgufFile gguf, int ropeDim)
    {
        Config = config;
        _weights = weights;
        _state = state;
        _gguf = gguf;
        _ropeDim = ropeDim;
    }

    /// <summary>
    /// Loads a Llama model from an opened GGUF file. The <paramref name="gguf"/> must remain
    /// alive for the lifetime of the returned model (the model stores a reference).
    /// </summary>
    public static LlamaModel LoadFromGguf(GgufFile gguf, ModelConfig config)
    {
        var weights = LlamaWeights.LoadFromGguf(gguf, config);

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;

        var state = new LlamaForwardState(
            config.HiddenSize,
            config.NumAttentionHeads,
            config.NumKvHeads,
            config.HeadDim,
            config.IntermediateSize,
            config.VocabSize,
            config.MaxSequenceLength,
            ropeDim,
            ropeTheta);

        return new LlamaModel(config, weights, state, gguf, ropeDim);
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
    {
        int maxSeq = Config.MaxSequenceLength;
        for (int i = 0; i < positions.Length; i++)
        {
            if ((uint)positions[i] >= (uint)maxSeq)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {positions[i]} at index {i} exceeds max sequence length {maxSeq}.");
        }

        int seqLen = tokenIds.Length;
        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int intermediateSize = Config.IntermediateSize;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;

        _state.EnsureCapacity(seqLen);

        float* hidden = (float*)_state.HiddenState;
        float* residual = (float*)_state.Residual;
        float* normOut = (float*)_state.NormOutput;
        float* q = (float*)_state.Q;
        float* k = (float*)_state.K;
        float* v = (float*)_state.V;
        float* attnOut = (float*)_state.AttnOutput;
        float* ffnGate = (float*)_state.FfnGate;
        float* ffnUp = (float*)_state.FfnUp;
        float* siluOut = (float*)_state.SiluOutput;
        float* logits = (float*)_state.Logits;

        // 1. EMBEDDING LOOKUP
        EmbeddingLookup(tokenIds, hidden, hiddenSize);

        // 2. TRANSFORMER LAYERS
        for (int layer = 0; layer < Config.NumLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];

            // a. Copy hiddenState → residual
            new Span<float>(hidden, seqLen * hiddenSize).CopyTo(new Span<float>(residual, seqLen * hiddenSize));

            // b-k: Process each token through the layer
            for (int t = 0; t < seqLen; t++)
            {
                float* hiddenT = hidden + t * hiddenSize;
                float* residualT = residual + t * hiddenSize;
                float* normOutT = normOut + t * hiddenSize;
                float* qT = q + t * (numHeads * headDim);
                float* kT = k + t * (numKvHeads * headDim);
                float* vT = v + t * (numKvHeads * headDim);

                // b. RMSNorm(hiddenState, attnNormWeight, eps) → normOutput
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(hiddenT, hiddenSize),
                    lw.AttnNormWeight,
                    eps,
                    new Span<float>(normOutT, hiddenSize));

                // c. Q/K/V projections (GEMV per token)
                Gemv(lw.QWeight, lw.QQuantType, normOutT, qT, lw.QOutputDim, lw.QInputDim);
                Gemv(lw.KWeight, lw.KQuantType, normOutT, kT, lw.KOutputDim, lw.KInputDim);
                Gemv(lw.VWeight, lw.VQuantType, normOutT, vT, lw.VOutputDim, lw.VInputDim);
            }

            // d. RoPE (in-place on Q and K for all tokens)
            RoPE.Execute(
                new Span<float>(q, seqLen * numHeads * headDim),
                new Span<float>(k, seqLen * numKvHeads * headDim),
                positions,
                numHeads, numKvHeads, headDim, _ropeDim,
                _state.CosTable, _state.SinTable);

            // e. Attention
            Attention.Execute(
                new ReadOnlySpan<float>(q, seqLen * numHeads * headDim),
                new ReadOnlySpan<float>(k, seqLen * numKvHeads * headDim),
                new ReadOnlySpan<float>(v, seqLen * numKvHeads * headDim),
                new Span<float>(attnOut, seqLen * numHeads * headDim),
                seqLen, seqLen, numHeads, numKvHeads, headDim, 0);

            // f-g. Output projection + residual add (per token)
            for (int t = 0; t < seqLen; t++)
            {
                float* attnOutT = attnOut + t * (numHeads * headDim);
                float* normOutT = normOut + t * hiddenSize;
                float* hiddenT = hidden + t * hiddenSize;
                float* residualT = residual + t * hiddenSize;

                // f. normOutput = Wo @ attnOutput
                Gemv(lw.OWeight, lw.OQuantType, attnOutT, normOutT, lw.OOutputDim, lw.OInputDim);

                // g. hiddenState = residual + normOutput (residual add)
                Add.Execute(
                    new ReadOnlySpan<float>(residualT, hiddenSize),
                    new ReadOnlySpan<float>(normOutT, hiddenSize),
                    new Span<float>(hiddenT, hiddenSize));
            }

            // h. Copy hiddenState → residual
            new Span<float>(hidden, seqLen * hiddenSize).CopyTo(new Span<float>(residual, seqLen * hiddenSize));

            // i-k. FFN (per token)
            for (int t = 0; t < seqLen; t++)
            {
                float* hiddenT = hidden + t * hiddenSize;
                float* residualT = residual + t * hiddenSize;
                float* normOutT = normOut + t * hiddenSize;
                float* gateT = ffnGate + t * intermediateSize;
                float* upT = ffnUp + t * intermediateSize;
                float* siluT = siluOut + t * intermediateSize;

                // i. RMSNorm(hiddenState, ffnNormWeight, eps) → normOutput
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(hiddenT, hiddenSize),
                    lw.FfnNormWeight,
                    eps,
                    new Span<float>(normOutT, hiddenSize));

                // j. SwiGLU FFN
                Gemv(lw.GateWeight, lw.GateQuantType, normOutT, gateT, lw.GateOutputDim, lw.GateInputDim);
                Gemv(lw.UpWeight, lw.UpQuantType, normOutT, upT, lw.UpOutputDim, lw.UpInputDim);

                SiLu.Execute(
                    new ReadOnlySpan<float>(gateT, intermediateSize),
                    new Span<float>(siluT, intermediateSize));

                Multiply.Execute(
                    new ReadOnlySpan<float>(siluT, intermediateSize),
                    new ReadOnlySpan<float>(upT, intermediateSize),
                    new Span<float>(siluT, intermediateSize));

                // Use normOutT as scratch for down projection result (hiddenSize fits)
                Gemv(lw.DownWeight, lw.DownQuantType, siluT, normOutT, lw.DownOutputDim, lw.DownInputDim);

                // k. hiddenState = residual + normOutput (residual add)
                Add.Execute(
                    new ReadOnlySpan<float>(residualT, hiddenSize),
                    new ReadOnlySpan<float>(normOutT, hiddenSize),
                    new Span<float>(hiddenT, hiddenSize));
            }
        }

        // 3. FINAL NORM (in-place: hidden → hidden)
        for (int t = 0; t < seqLen; t++)
        {
            float* hiddenT = hidden + t * hiddenSize;
            // Use normOut as temp so we can copy back
            float* normOutT = normOut + t * hiddenSize;

            RmsNorm.Execute(
                new ReadOnlySpan<float>(hiddenT, hiddenSize),
                _weights.OutputNormWeight,
                eps,
                new Span<float>(normOutT, hiddenSize));

            new Span<float>(normOutT, hiddenSize).CopyTo(new Span<float>(hiddenT, hiddenSize));
        }

        // 4. LM HEAD — only last token
        float* lastHidden = hidden + (seqLen - 1) * hiddenSize;
        Gemv(_weights.OutputWeight, _weights.OutputQuantType,
             lastHidden, logits, _weights.OutputOutputDim, _weights.OutputInputDim);

        // 5. RETURN [1, vocabSize] — copy logits to new tensor (caller owns disposal)
        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
        new Span<float>(logits, vocabSize).CopyTo(
            new Span<float>((void*)result.DataPointer, vocabSize));

        return result;
    }

    /// <summary>
    /// Dispatches to the appropriate GEMV kernel based on quantization type.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Gemv(nint weights, QuantizationType qt, float* x, float* y, int m, int k)
    {
        if (qt == QuantizationType.Q8_0)
            MatMul.GemvQ8_0((byte*)weights, x, y, m, k);
        else if (qt == QuantizationType.F32)
            MatMul.GemvF32((float*)weights, x, y, m, k);
        else if (qt == QuantizationType.F16)
            GemvF16(weights, x, y, m, k);
        else
            throw new NotSupportedException($"Unsupported quantization type for GEMV: {qt}");
    }

    /// <summary>
    /// F16 GEMV: dequantize each row to scratch, then dot product. Uses stackalloc for small rows.
    /// </summary>
    [SkipLocalsInit]
    private static void GemvF16(nint weights, float* x, float* y, int m, int k)
    {
        const int stackThreshold = 2048; // 8KB of floats
        Half* weightsHalf = (Half*)weights;

        if (k <= stackThreshold)
        {
            float* rowBuf = stackalloc float[k];
            for (int row = 0; row < m; row++)
            {
                var srcRow = new ReadOnlySpan<Half>(weightsHalf + row * k, k);
                var destRow = new Span<float>(rowBuf, k);
                System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(srcRow, destRow);
                y[row] = System.Numerics.Tensors.TensorPrimitives.Dot(destRow, new ReadOnlySpan<float>(x, k));
            }
        }
        else
        {
            float[] rented = System.Buffers.ArrayPool<float>.Shared.Rent(k);
            try
            {
                for (int row = 0; row < m; row++)
                {
                    var srcRow = new ReadOnlySpan<Half>(weightsHalf + row * k, k);
                    var destRow = rented.AsSpan(0, k);
                    System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(srcRow, destRow);
                    y[row] = System.Numerics.Tensors.TensorPrimitives.Dot(destRow, new ReadOnlySpan<float>(x, k));
                }
            }
            finally
            {
                System.Buffers.ArrayPool<float>.Shared.Return(rented);
            }
        }
    }

    /// <summary>
    /// Copies or dequantizes one row of the embedding table per token into the hidden state buffer.
    /// </summary>
    private void EmbeddingLookup(ReadOnlySpan<int> tokenIds, float* hidden, int hiddenSize)
    {
        nint embPtr = _weights.TokenEmbedWeight;
        var qt = _weights.TokenEmbedQuantType;

        for (int t = 0; t < tokenIds.Length; t++)
        {
            int tokenId = tokenIds[t];
            if ((uint)tokenId >= (uint)Config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds),
                    $"Token ID {tokenId} at position {t} is out of range [0, {Config.VocabSize}).");

            float* dest = hidden + t * hiddenSize;
            var destSpan = new Span<float>(dest, hiddenSize);

            if (qt == QuantizationType.F32)
            {
                // Direct copy
                float* src = (float*)embPtr + (long)tokenId * hiddenSize;
                new ReadOnlySpan<float>(src, hiddenSize).CopyTo(destSpan);
            }
            else if (qt == QuantizationType.Q8_0)
            {
                // Dequantize one row: each row is hiddenSize elements in Q8_0 blocks
                int blocksPerRow = hiddenSize / Q8_0GroupSize;
                long rowOffset = (long)tokenId * blocksPerRow * Q8_0BlockBytes;
                nint rowPtr = embPtr + (nint)rowOffset;
                Dequantize.ToFloat32(rowPtr, hiddenSize, QuantizationType.Q8_0, destSpan);
            }
            else if (qt == QuantizationType.F16)
            {
                Half* src = (Half*)embPtr + (long)tokenId * hiddenSize;
                System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(
                    new ReadOnlySpan<Half>(src, hiddenSize), destSpan);
            }
            else
            {
                throw new NotSupportedException($"Unsupported embedding quantization type: {qt}");
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _state.Dispose();
        // _weights and _gguf are not owned by us — caller manages GgufFile lifetime.
    }
}
