using System.Buffers;
using System.Numerics.Tensors;
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
/// Transformer forward pass: embedding lookup → N × transformer blocks → final norm → LM head → logits.
/// Operates entirely on the CPU using pre-allocated scratch buffers for zero-allocation inference.
/// </summary>
public sealed unsafe class TransformerModel : IModel
{
    /// <summary>Q8_0 block: 2 bytes (Half scale) + 32 bytes (sbyte values).</summary>
    private const int Q8_0BlockBytes = 34;

    /// <summary>Elements per Q8_0 block.</summary>
    private const int Q8_0GroupSize = 32;

    /// <summary>Elements per Q8_1 block.</summary>
    private const int Q8_1GroupSize = 32;

    private readonly TransformerWeights _weights;
    private readonly TransformerForwardState _state;
    private readonly GgufFile _gguf; // prevent premature GC of mmap
    private readonly int _ropeDim;
    private readonly RoPEType _ropeType;
    private readonly int? _slidingWindowSize;
    private readonly ComputeThreadPool? _threadPool;
    private readonly bool _ownsThreadPool;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <summary>Total bytes allocated for inference scratch buffers.</summary>
    public long ComputeMemoryBytes => _state.AllocatedBytes;

    private TransformerModel(ModelConfig config, TransformerWeights weights, TransformerForwardState state,
                       GgufFile gguf, int ropeDim, RoPEType ropeType,
                       ComputeThreadPool? threadPool, bool ownsPool)
    {
        Config = config;
        _weights = weights;
        _state = state;
        _gguf = gguf;
        _ropeDim = ropeDim;
        _ropeType = ropeType;
        _slidingWindowSize = config.SlidingWindowSize;
        _threadPool = threadPool;
        _ownsThreadPool = ownsPool;
    }

    /// <summary>
    /// Loads a transformer model from an opened GGUF file (single-threaded).
    /// The <paramref name="gguf"/> must remain alive for the lifetime of the returned model.
    /// </summary>
    public static TransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config)
        => LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);

    /// <summary>
    /// Loads a transformer model from an opened GGUF file with threading configuration.
    /// When <paramref name="threading"/> is parallel, creates a <see cref="ComputeThreadPool"/>
    /// owned by this model (disposed with the model).
    /// </summary>
    public static TransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, ThreadingConfig threading)
    {
        var weights = TransformerWeights.LoadFromGguf(gguf, config);

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        RoPEType ropeType = config.RoPEConfig?.Type ?? RoPEType.Norm;

        var state = new TransformerForwardState(
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

        return new TransformerModel(config, weights, state, gguf, ropeDim, ropeType, pool, ownsPool: pool is not null);
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

            // Batched Q/K/V projections — fused dispatch for decode (seqLen=1)
            if (seqLen == 1 && _threadPool != null)
            {
                FusedQkvDecode(in lw, normOut, preQuantNorm, q, k, v);
            }
            else
            {
                // Guard: only reuse preQuantNorm when target quant type is in the same family as QQuantType
                Gemm(lw.QWeight, lw.QQuantType, normOut, q, lw.QOutputDim, lw.QInputDim, seqLen, preQuantNorm);
                Gemm(lw.KWeight, lw.KQuantType, normOut, k, lw.KOutputDim, lw.KInputDim, seqLen,
                     IsCompatiblePreQuant(lw.QQuantType, lw.KQuantType) ? preQuantNorm : null);
                Gemm(lw.VWeight, lw.VQuantType, normOut, v, lw.VOutputDim, lw.VInputDim, seqLen,
                     IsCompatiblePreQuant(lw.QQuantType, lw.VQuantType) ? preQuantNorm : null);
            }

            // Optional bias: y = Wx + b (no-op when null)
            AddBias(lw.QBias, q, lw.QOutputDim, seqLen);
            AddBias(lw.KBias, k, lw.KOutputDim, seqLen);
            AddBias(lw.VBias, v, lw.VOutputDim, seqLen);

            // Optional QK-norms (Qwen3-style): per-head RMSNorm on Q/K after projection, before RoPE
            if (lw.QNormWeight is not null)
                ApplyPerHeadNorm(lw.QNormWeight, q, numHeads, headDim, seqLen, eps);
            if (lw.KNormWeight is not null)
                ApplyPerHeadNorm(lw.KNormWeight, k, numKvHeads, headDim, seqLen, eps);

            // d. RoPE (in-place on Q and K for all tokens)
            RoPE.Execute(
                new Span<float>(q, seqLen * numHeads * headDim),
                new Span<float>(k, seqLen * kvStride),
                positions,
                numHeads, numKvHeads, headDim, _ropeDim,
                _state.CosTable, _state.SinTable, _ropeType);

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
                    seqLen, seqKv, numHeads, numKvHeads, headDim, positions[0], _threadPool,
                    _slidingWindowSize);
            }
            else
            {
                Attention.Execute(q, k, v, attnOut,
                    seqLen, seqLen, numHeads, numKvHeads, headDim, 0, _threadPool,
                    _slidingWindowSize);
            }

            // f. Batched O projection
            byte* preQuantAttn = QuantizeInput(attnOut, inputQ8Scratch, numHeads * headDim, seqLen, lw.OQuantType);
            Gemm(lw.OWeight, lw.OQuantType, attnOut, normOut, lw.OOutputDim, lw.OInputDim, seqLen, preQuantAttn);
            AddBias(lw.OBias, normOut, lw.OOutputDim, seqLen);

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

            // Batched Gate + Up projections — fused dispatch for decode (seqLen=1)
            if (seqLen == 1 && _threadPool != null)
            {
                FusedGateUpDecode(in lw, normOut, preQuantFfn, ffnGate, ffnUp);
            }
            else
            {
                Gemm(lw.GateWeight, lw.GateQuantType, normOut, ffnGate, lw.GateOutputDim, lw.GateInputDim, seqLen, preQuantFfn);
                Gemm(lw.UpWeight, lw.UpQuantType, normOut, ffnUp, lw.UpOutputDim, lw.UpInputDim, seqLen,
                     IsCompatiblePreQuant(lw.GateQuantType, lw.UpQuantType) ? preQuantFfn : null);
            }
            AddBias(lw.GateBias, ffnGate, lw.GateOutputDim, seqLen);
            AddBias(lw.UpBias, ffnUp, lw.UpOutputDim, seqLen);

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
            AddBias(lw.DownBias, normOut, lw.DownOutputDim, seqLen);

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
    /// Applies RMSNorm per attention head to a Q or K tensor [seqLen, numHeads * headDim].
    /// Used for QK-norm (Qwen3-style) where each head vector is independently normalized
    /// after projection and before RoPE.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyPerHeadNorm(float[] normWeight, float* qk,
        int numHeads, int headDim, int seqLen, float eps)
    {
        int stride = numHeads * headDim;
        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                float* head = qk + t * stride + h * headDim;
                var input = new ReadOnlySpan<float>(head, headDim);
                var output = new Span<float>(head, headDim);
                RmsNorm.Execute(input, normWeight, eps, output);
            }
        }
    }

    /// <summary>
    /// Adds a bias vector [outputDim] to each row of a [seqLen, outputDim] output buffer.
    /// No-op when <paramref name="bias"/> is null (zero overhead for bias-less models).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddBias(float[]? bias, float* output, int outputDim, int seqLen)
    {
        if (bias is null) return;
        for (int t = 0; t < seqLen; t++)
        {
            var row = new Span<float>(output + t * outputDim, outputDim);
            TensorPrimitives.Add((ReadOnlySpan<float>)row, bias, row);
        }
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
        else if (qt == QuantizationType.Q5_0)
            MatMul.GemvQ5_0((byte*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.Q4_K)
            MatMul.GemvQ4_K((byte*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.Q5_K)
            MatMul.GemvQ5_K((byte*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.Q6_K)
            MatMul.GemvQ6_K((byte*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.F32)
            MatMul.GemvF32((float*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.F16)
            MatMul.GemvF16(weights, x, y, m, k, _threadPool);
        else
            GemvDequantFallback(weights, qt, x, y, m, k);
    }

    /// <summary>
    /// Fallback GEMV for quant types without dedicated vec_dot kernels.
    /// Dequantizes one weight row at a time and computes float dot product.
    /// Correct but slower than fused kernels.
    /// </summary>
    private static void GemvDequantFallback(nint weights, QuantizationType qt, float* x, float* y, int m, int k)
    {
        long rowBytes = Dequantize.RowByteSize(k, qt);
        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            var xSpan = new ReadOnlySpan<float>(x, k);
            for (int i = 0; i < m; i++)
            {
                Dequantize.ToFloat32(weights + i * (nint)rowBytes, k, qt, rowSpan);
                y[i] = TensorPrimitives.Dot(new ReadOnlySpan<float>(rowBuf, 0, k), xSpan);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
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
        else if (qt == QuantizationType.Q5_0)
            MatMul.GemmQ5_0((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
        else if (qt == QuantizationType.Q4_K)
            MatMul.GemmQ4_K((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
        else if (qt == QuantizationType.Q5_K)
            MatMul.GemmQ5_K((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
        else if (qt == QuantizationType.Q6_K)
            MatMul.GemmQ6_K((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
        else if (qt == QuantizationType.F32)
            MatMul.GemmF32((float*)weights, b, c, m, k, n, _threadPool);
        else if (qt == QuantizationType.F16)
            MatMul.GemmF16(weights, b, c, m, k, n, _threadPool);
        else
            GemmDequantFallback(weights, qt, b, c, m, k, n);
    }

    /// <summary>
    /// Fallback GEMM for quant types without dedicated vec_dot kernels.
    /// Iterates per input row, calling <see cref="GemvDequantFallback"/> for each.
    /// </summary>
    private static void GemmDequantFallback(nint weights, QuantizationType qt, float* b, float* c,
                                            int m, int k, int n)
    {
        for (int t = 0; t < n; t++)
        {
            GemvDequantFallback(weights, qt, b + t * k, c + t * m, m, k);
        }
    }

    /// <summary>
    /// Returns true when a pre-quantized buffer produced for <paramref name="preQuantSource"/>
    /// can be safely reused for a GEMM targeting <paramref name="target"/>.
    /// K-quant types share Q8_K layout. Q8_0 and Q5_0 each use different input quantization
    /// (Q8_0 and Q8_1 respectively) and cannot share pre-quantized buffers.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsCompatiblePreQuant(QuantizationType preQuantSource, QuantizationType target)
    {
        if (preQuantSource == target) return true;

        bool sourceIsKQuant = preQuantSource is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K;
        bool targetIsKQuant = target is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K;
        if (sourceIsKQuant && targetIsKQuant) return true;

        // Q8_0 and Q5_0 no longer share input format — Q8_0 uses Q8_0, Q5_0 uses Q8_1.
        return false;
    }

    /// <summary>
    /// Pre-quantizes [seqLen, dim] f32 input for GEMM reuse across Q/K/V or Gate/Up projections.
    /// K-quant types (Q4_K, Q5_K, Q6_K) use Q8_K (float32 scale, 256 elements/block).
    /// Q8_0 uses Q8_0 (Half scale, 32 elements/block).
    /// Q5_0 uses Q8_1 (Half d + Half s, 32 elements/block) with precomputed block sums.
    /// Returns the scratch pointer if quantized, otherwise null.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte* QuantizeInput(float* input, byte* scratch, int dim, int seqLen,
                                       QuantizationType qt)
    {
        if (qt == QuantizationType.Q4_K || qt == QuantizationType.Q5_K || qt == QuantizationType.Q6_K)
        {
            int blockCount = dim / 256; // Q8_K_GroupSize
            int q8kRowBytes = blockCount * MatMul.Q8_K_BlockBytes;
            for (int t = 0; t < seqLen; t++)
                MatMul.QuantizeF32ToQ8_K(input + t * dim, scratch + t * q8kRowBytes, dim);
            return scratch;
        }

        if (qt == QuantizationType.Q5_0)
        {
            int blockCount = dim / Q8_1GroupSize;
            int q8_1RowBytes = blockCount * MatMul.Q8_1BlockBytes;
            for (int t = 0; t < seqLen; t++)
                MatMul.QuantizeF32ToQ8_1(input + t * dim, scratch + t * q8_1RowBytes, dim);
            return scratch;
        }

        if (qt == QuantizationType.Q8_0)
        {
            int blockCount = dim / Q8_0GroupSize;
            int q8RowBytes = blockCount * Q8_0BlockBytes;
            for (int t = 0; t < seqLen; t++)
                MatMul.QuantizeF32ToQ8_0(input + t * dim, scratch + t * q8RowBytes, dim);
            return scratch;
        }

        return null;
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
                // Generic dequant fallback for any supported quant type (Q4_K, Q5_K, Q6_K, Q5_0, etc.)
                long rowBytes = Dequantize.RowByteSize(hiddenSize, qt);
                long rowOffset = (long)tokenId * rowBytes;
                nint rowPtr = embPtr + (nint)rowOffset;
                Dequantize.ToFloat32(rowPtr, hiddenSize, qt, destSpan);
            }
        }
    }

    /// <summary>
    /// Fused Q/K/V decode: dispatches all three projections in a single pool.Dispatch() call
    /// when they share the same quant family, saving 2 dispatch overheads per layer.
    /// Cross-family projections dispatch individually with self-quantizing GEMV.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void FusedQkvDecode(ref readonly TransformerLayerWeights lw,
        float* normOut, byte* preQuantNorm, float* q, float* k, float* v)
    {
        // preQuantNorm was quantized for lw.QQuantType's family.
        // FusedDecodeGemv3 handles cross-family by dispatching those projections
        // individually with self-quantizing GEMV (null preQuant).
        MatMul.FusedDecodeGemv3(
            (byte*)lw.QWeight, lw.QQuantType, q, lw.QOutputDim,
            (byte*)lw.KWeight, lw.KQuantType, k, lw.KOutputDim,
            (byte*)lw.VWeight, lw.VQuantType, v, lw.VOutputDim,
            normOut, preQuantNorm, lw.QInputDim,
            _threadPool!);
    }

    /// <summary>
    /// Fused Gate/Up decode: dispatches both projections in a single pool.Dispatch() call
    /// when they share the same quant family, saving 1 dispatch overhead per layer.
    /// Cross-family projections dispatch individually with self-quantizing GEMV.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void FusedGateUpDecode(ref readonly TransformerLayerWeights lw,
        float* normOut, byte* preQuantFfn, float* ffnGate, float* ffnUp)
    {
        // preQuantFfn was quantized for lw.GateQuantType's family.
        // FusedDecodeGemv2 handles cross-family by dispatching separately.
        MatMul.FusedDecodeGemv2(
            (byte*)lw.GateWeight, lw.GateQuantType, ffnGate, lw.GateOutputDim,
            (byte*)lw.UpWeight, lw.UpQuantType, ffnUp, lw.UpOutputDim,
            normOut, preQuantFfn, lw.GateInputDim,
            _threadPool!);
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
