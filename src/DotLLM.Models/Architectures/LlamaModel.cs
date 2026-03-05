using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
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
    private readonly ComputeThreadPool? _threadPool;
    private readonly bool _ownsThreadPool;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <summary>Total bytes allocated for inference scratch buffers.</summary>
    public long ComputeMemoryBytes => _state.AllocatedBytes;

    private LlamaModel(ModelConfig config, LlamaWeights weights, LlamaForwardState state,
                       GgufFile gguf, int ropeDim, ComputeThreadPool? threadPool, bool ownsPool)
    {
        Config = config;
        _weights = weights;
        _state = state;
        _gguf = gguf;
        _ropeDim = ropeDim;
        _threadPool = threadPool;
        _ownsThreadPool = ownsPool;
    }

    /// <summary>
    /// Loads a Llama model from an opened GGUF file (single-threaded).
    /// The <paramref name="gguf"/> must remain alive for the lifetime of the returned model.
    /// </summary>
    public static LlamaModel LoadFromGguf(GgufFile gguf, ModelConfig config)
        => LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);

    /// <summary>
    /// Loads a Llama model from an opened GGUF file with threading configuration.
    /// When <paramref name="threading"/> is parallel, creates a <see cref="ComputeThreadPool"/>
    /// owned by this model (disposed with the model).
    /// </summary>
    public static LlamaModel LoadFromGguf(GgufFile gguf, ModelConfig config, ThreadingConfig threading)
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

        ComputeThreadPool? pool = threading.IsParallel
            ? new ComputeThreadPool(threading.EffectiveThreadCount)
            : null;

        return new LlamaModel(config, weights, state, gguf, ropeDim, pool, ownsPool: pool is not null);
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <summary>
    /// Runs a forward pass with optional KV-cache. When <paramref name="kvCache"/> is provided,
    /// K/V projections are stored in the cache after RoPE, and attention reads from the full
    /// cached context — enabling O(1) per-token decode instead of O(n) recomputation.
    /// </summary>
    /// <param name="tokenIds">Input token IDs for this step (all prompt tokens for prefill, single token for decode).</param>
    /// <param name="positions">Position indices for each token.</param>
    /// <param name="deviceId">Target device for computation.</param>
    /// <param name="kvCache">Optional KV-cache. When null, behaves identically to the uncached forward pass.</param>
    /// <returns>Logits tensor of shape [1, vocab_size] for the last token.</returns>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
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
        int kvStride = numKvHeads * headDim;
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

            // b. RMSNorm all tokens
            for (int t = 0; t < seqLen; t++)
            {
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                    lw.AttnNormWeight,
                    eps,
                    new Span<float>(normOut + t * hiddenSize, hiddenSize));
            }

            // c. Pre-quantize normOutput once, reuse across Q/K/V projections
            byte* inputQ8Scratch = (byte*)_state.InputQ8Scratch;
            byte* preQuantNorm = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, seqLen, lw.QQuantType);

            // Batched Q/K/V projections (GEMM)
            Gemm(lw.QWeight, lw.QQuantType, normOut, q, lw.QOutputDim, lw.QInputDim, seqLen, preQuantNorm);
            Gemm(lw.KWeight, lw.KQuantType, normOut, k, lw.KOutputDim, lw.KInputDim, seqLen, preQuantNorm);
            Gemm(lw.VWeight, lw.VQuantType, normOut, v, lw.VOutputDim, lw.VInputDim, seqLen, preQuantNorm);

            // d. RoPE (in-place on Q and K for all tokens)
            RoPE.Execute(
                new Span<float>(q, seqLen * numHeads * headDim),
                new Span<float>(k, seqLen * kvStride),
                positions,
                numHeads, numKvHeads, headDim, _ropeDim,
                _state.CosTable, _state.SinTable);

            // e. Attention — with or without KV-cache
            if (kvCache is not null)
            {
                // Store new K/V in cache, then attend over full cached context (zero allocations)
                var kRef = new TensorRef(seqLen, kvStride, DType.Float32, -1, (nint)k);
                var vRef = new TensorRef(seqLen, kvStride, DType.Float32, -1, (nint)v);

                kvCache.Update(kRef, vRef, positions, layer);

                int seqKv = kvCache.CurrentLength;
                var cachedK = kvCache.GetKeysRef(layer);
                var cachedV = kvCache.GetValuesRef(layer);

                Attention.Execute(q, (float*)cachedK.DataPointer, (float*)cachedV.DataPointer, attnOut,
                    seqLen, seqKv, numHeads, numKvHeads, headDim, positions[0], _threadPool);
            }
            else
            {
                Attention.Execute(q, k, v, attnOut,
                    seqLen, seqLen, numHeads, numKvHeads, headDim, 0, _threadPool);
            }

            // f. Batched O projection
            byte* preQuantAttn = QuantizeInput(attnOut, inputQ8Scratch, numHeads * headDim, seqLen, lw.OQuantType);
            Gemm(lw.OWeight, lw.OQuantType, attnOut, normOut, lw.OOutputDim, lw.OInputDim, seqLen, preQuantAttn);

            // g. Residual add (per token)
            for (int t = 0; t < seqLen; t++)
            {
                Add.Execute(
                    new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                    new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                    new Span<float>(hidden + t * hiddenSize, hiddenSize));
            }

            // h. Copy hiddenState → residual
            new Span<float>(hidden, seqLen * hiddenSize).CopyTo(new Span<float>(residual, seqLen * hiddenSize));

            // i. FFN RMSNorm all tokens
            for (int t = 0; t < seqLen; t++)
            {
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                    lw.FfnNormWeight,
                    eps,
                    new Span<float>(normOut + t * hiddenSize, hiddenSize));
            }

            // j. Pre-quantize normOutput once, reuse across Gate/Up projections
            byte* preQuantFfn = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, seqLen, lw.GateQuantType);

            // Batched Gate + Up projections
            Gemm(lw.GateWeight, lw.GateQuantType, normOut, ffnGate, lw.GateOutputDim, lw.GateInputDim, seqLen, preQuantFfn);
            Gemm(lw.UpWeight, lw.UpQuantType, normOut, ffnUp, lw.UpOutputDim, lw.UpInputDim, seqLen, preQuantFfn);

            // SiLU + Multiply (element-wise, per token)
            for (int t = 0; t < seqLen; t++)
            {
                float* gateT = ffnGate + t * intermediateSize;
                float* upT = ffnUp + t * intermediateSize;
                float* siluT = siluOut + t * intermediateSize;

                SiLu.Execute(
                    new ReadOnlySpan<float>(gateT, intermediateSize),
                    new Span<float>(siluT, intermediateSize));

                Multiply.Execute(
                    new ReadOnlySpan<float>(siluT, intermediateSize),
                    new ReadOnlySpan<float>(upT, intermediateSize),
                    new Span<float>(siluT, intermediateSize));
            }

            // Pre-quantize siluOutput for Down projection (different input dim = intermediateSize)
            byte* preQuantSilu = QuantizeInput(siluOut, inputQ8Scratch, intermediateSize, seqLen, lw.DownQuantType);

            // Batched Down projection (output into normOut as scratch)
            Gemm(lw.DownWeight, lw.DownQuantType, siluOut, normOut, lw.DownOutputDim, lw.DownInputDim, seqLen, preQuantSilu);

            // k. Residual add (per token)
            for (int t = 0; t < seqLen; t++)
            {
                Add.Execute(
                    new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                    new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                    new Span<float>(hidden + t * hiddenSize, hiddenSize));
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
    /// Passes <see cref="_threadPool"/> for parallel execution.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Gemv(nint weights, QuantizationType qt, float* x, float* y, int m, int k)
    {
        if (qt == QuantizationType.Q8_0)
            MatMul.GemvQ8_0((byte*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.F32)
            MatMul.GemvF32((float*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.F16)
            MatMul.GemvF16(weights, x, y, m, k, _threadPool);
        else
            throw new NotSupportedException($"Unsupported quantization type for GEMV: {qt}");
    }

    /// <summary>
    /// Dispatches to the appropriate GEMM kernel based on quantization type.
    /// Passes <see cref="_threadPool"/> for parallel execution.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Gemm(nint weights, QuantizationType qt, float* b, float* c,
                      int m, int k, int n, byte* preQuantizedInput = null)
    {
        if (qt == QuantizationType.Q8_0)
            MatMul.GemmQ8_0((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
        else if (qt == QuantizationType.F32)
            MatMul.GemmF32((float*)weights, b, c, m, k, n, _threadPool);
        else if (qt == QuantizationType.F16)
            MatMul.GemmF16(weights, b, c, m, k, n, _threadPool);
        else
            throw new NotSupportedException($"Unsupported quantization type for GEMM: {qt}");
    }

    /// <summary>
    /// Pre-quantizes [seqLen, dim] f32 input to Q8_0 into <see cref="LlamaForwardState.InputQ8Scratch"/>.
    /// Returns the scratch pointer if Q8_0 and seqLen &gt; 1, otherwise null (no benefit).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte* QuantizeInput(float* input, byte* scratch, int dim, int seqLen,
                                       QuantizationType qt)
    {
        if (qt != QuantizationType.Q8_0 || seqLen <= 1)
            return null;

        int blockCount = dim / 32; // Q8_0GroupSize
        int q8RowBytes = blockCount * 34; // Q8_0BlockBytes
        for (int t = 0; t < seqLen; t++)
            MatMul.QuantizeF32ToQ8_0(input + t * dim, scratch + t * q8RowBytes, dim);

        return scratch;
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
        if (_ownsThreadPool)
            _threadPool?.Dispose();
        _state.Dispose();
        // _weights and _gguf are not owned by us — caller manages GgufFile lifetime.
    }
}
