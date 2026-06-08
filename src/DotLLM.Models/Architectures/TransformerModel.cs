using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;

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
    // Persistent KV cache for MLA layers. Exactly one of these is non-null
    // at any time, selected by Config.MlaConfig.UseLatentCache at first use.
    // Both are lazily constructed on the first MLA forward and reset when
    // the caller signals a fresh sequence via positions[0] == 0. See
    // MlaExpandedKvState / MlaLatentKvState docstrings for the Phase A vs
    // Phase B distinction (correctness oracle vs ~7× memory win).
    private MlaExpandedKvState? _mlaKvState;
    private MlaLatentKvState? _mlaLatentKvState;

    // Lifetime anchor for the underlying mmap-backed weight file. Holds a
    // strong reference so the GC cannot collect the GgufFile / SafetensorsFile
    // while weight pointers are still in use. Not null for any loaded model.
#pragma warning disable IDE0052, CA1823 // field used only as a GC root
    private readonly object _mmapAnchor;
#pragma warning restore IDE0052, CA1823

    // Active LoRA adapter for the current Forward call (when invoked via the
    // 5-arg adapter-aware overload). Cleared back to null in the try/finally
    // surrounding the call. Not thread-safe — TransformerModel as a whole is
    // single-threaded per instance (forward state buffers, MLA caches are
    // also instance-scoped) so this is consistent with existing semantics.
    private ILoraAdapter? _currentAdapter;
    private readonly int _ropeDim;
    private readonly RoPEType _ropeType;
    private readonly int? _slidingWindowSize;
    private readonly ComputeThreadPool? _threadPool;
    private readonly bool _ownsThreadPool;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <summary>Total bytes allocated for inference scratch buffers.</summary>
    public long ComputeMemoryBytes => _state.AllocatedBytes;

    /// <summary>Debug: limit the number of transformer layers processed. 0 = all layers (default). -1 = skip all layers (embedding + LM head only).</summary>
    internal int DebugMaxLayers { get; set; }

    private TransformerModel(ModelConfig config, TransformerWeights weights, TransformerForwardState state,
                       object mmapAnchor, int ropeDim, RoPEType ropeType,
                       ComputeThreadPool? threadPool, bool ownsPool)
    {
        Config = config;
        _weights = weights;
        _state = state;
        _mmapAnchor = mmapAnchor;
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
        weights.RepackWeights();

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

        ComputeThreadPool? pool = null;
        if (threading.IsParallel)
        {
            int effectiveThreads = threading.EffectiveThreadCount;

            if (threading.EnableNumaPinning || threading.EnablePCorePinning)
            {
                var topology = NumaTopology.Detect();

                // If P-core pinning on hybrid, cap threads to P-core count
                if (threading.EnablePCorePinning && topology.IsHybrid)
                    effectiveThreads = Math.Min(effectiveThreads, topology.PerformanceCoreIds.Count);

                pool = new ComputeThreadPool(effectiveThreads, topology, threading);
            }
            else
            {
                pool = new ComputeThreadPool(effectiveThreads, topology: null, threading);
            }
        }

        return new TransformerModel(config, weights, state, gguf, ropeDim, ropeType, pool, ownsPool: pool is not null);
    }

    /// <summary>
    /// Loads a transformer model from an opened HuggingFace-convention
    /// safetensors source (single-threaded). The <paramref name="file"/> must
    /// remain alive for the lifetime of the returned model — internally
    /// anchored to prevent GC, but the caller must still dispose it after
    /// disposing the model.
    /// </summary>
    /// <param name="file">The safetensors source (single-file or multi-shard).</param>
    /// <param name="config">The model configuration.</param>
    /// <returns>The loaded transformer model.</returns>
    public static TransformerModel LoadFromSafetensors(ISafetensorsTensorSource file, ModelConfig config)
        => LoadFromSafetensors(file, config, ThreadingConfig.SingleThreaded);

    /// <summary>
    /// Loads a transformer model from an opened HuggingFace-convention
    /// safetensors source with threading configuration.
    /// </summary>
    /// <param name="file">The safetensors source (single-file or multi-shard).</param>
    /// <param name="config">The model configuration.</param>
    /// <param name="threading">Threading configuration for parallel execution.</param>
    /// <returns>The loaded transformer model.</returns>
    public static TransformerModel LoadFromSafetensors(
        ISafetensorsTensorSource file, ModelConfig config, ThreadingConfig threading)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(config);

        var weights = TransformerWeightsSafetensorsLoader.Load(file, config);
        weights.RepackWeights();

        // For MLA (DeepSeek-V2/V3) RoPE applies only to the decoupled
        // qk_rope_head_dim sub-dimension — NOT the full qk_head_dim carried
        // in ModelConfig.HeadDim. Size the RoPE table accordingly so the MLA
        // kernel's [pos, qk_rope_head_dim / 2] indexing lines up.
        int ropeDim = config.MlaConfig is not null
            ? config.MlaConfig.QkRopeHeadDim
            : (config.RoPEConfig?.DimensionCount ?? config.HeadDim);
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.MlaConfig?.RopeTheta ?? config.RoPEConfig?.Theta ?? 10000.0f;
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

        ComputeThreadPool? pool = null;
        if (threading.IsParallel)
        {
            int effectiveThreads = threading.EffectiveThreadCount;
            if (threading.EnableNumaPinning || threading.EnablePCorePinning)
            {
                var topology = NumaTopology.Detect();
                if (threading.EnablePCorePinning && topology.IsHybrid)
                    effectiveThreads = Math.Min(effectiveThreads, topology.PerformanceCoreIds.Count);
                pool = new ComputeThreadPool(effectiveThreads, topology, threading);
            }
            else
            {
                pool = new ComputeThreadPool(effectiveThreads, topology: null, threading);
            }
        }

        return new TransformerModel(config, weights, state, file, ropeDim, ropeType, pool, ownsPool: pool is not null);
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <summary>
    /// LoRA-aware forward. When <paramref name="adapter"/> is non-null, each
    /// adapted projection adds <c>scale × (x · B) · A</c> on top of the base
    /// projection. When null, this is byte-equivalent to the 4-arg overload.
    /// </summary>
    /// <remarks>
    /// MoE FFN sites are not adapted in this Phase 4a slice — if the model
    /// has any MoE layer AND <paramref name="adapter"/> targets a gate / up /
    /// down projection, the call throws <see cref="NotSupportedException"/>.
    /// MLA-specific projections (DeepSeek-V2/V3 q_a_proj, kv_a_proj_with_mqa,
    /// …) are also out of scope and silently passed through.
    /// </remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache, ILoraAdapter? adapter)
    {
        if (adapter is null)
            return Forward(tokenIds, positions, deviceId, kvCache);

        ValidateAdapterForModel(adapter);

        // Phase 4d.6 — eager transposed-A materialisation. The outer-product
        // stage-2 fast path needs a [rank, outputDim] view of A; building it
        // is O(outputDim × rank) per (layer, proj) — a few ms total for a
        // typical Llama-3.2-1B / rank=16 adapter. PrewarmAdapter is
        // idempotent so the actual cost is paid only on first activation;
        // hoisting it out of the per-Apply lazy path eliminates first-token
        // latency contamination AND smooths low-iteration BDN measurement
        // variance. No-op for rank != 16 or non-AVX-512 hosts.
        LoraStage2.PrewarmAdapter(adapter as LoraAdapter);

        _currentAdapter = adapter;
        try
        {
            return Forward(tokenIds, positions, deviceId, kvCache);
        }
        finally
        {
            _currentAdapter = null;
        }
    }

    /// <summary>
    /// Runs a forward pass with optional KV-cache. When <paramref name="kvCache"/> is provided,
    /// K/V projections are stored in the cache after RoPE, and attention reads from the full
    /// cached context — enabling O(1) per-token decode instead of O(n) recomputation.
    /// </summary>
    /// <param name="tokenIds">Input token IDs for this step (all prompt tokens for prefill, single token for decode).</param>
    /// <param name="positions">Position indices for each token.</param>
    /// <param name="deviceId">Target device for computation.</param>
    /// <param name="kvCache">Optional KV-cache. When null, behaves identically to the uncached forward pass.</param>
    /// <returns>Logits tensor of shape [seqLen, vocab_size] for all input positions.</returns>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        RunLayersAndFinalNormCore(tokenIds, positions, kvCache);
        return RunLmHead(tokenIds.Length, deviceId);
    }

    /// <summary>
    /// Embedding lookup + transformer layer loop + final RMSNorm. Leaves the final
    /// hidden state in <c>_state.HiddenState[0..seqLen*hiddenSize]</c>. Used by both
    /// <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?)"/>
    /// (which then runs the lm_head per call) and <see cref="ForwardBatch"/> (which
    /// invokes this once per sequence, snapshots each result, then runs ONE batched
    /// lm_head GEMM on the stacked snapshot).
    /// </summary>
    private unsafe void RunLayersAndFinalNormCore(
        ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, IKvCache? kvCache)
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

        // Adaptive dispatch mode: spin-wait for decode (short, frequent dispatches),
        // event-based for prefill (long dispatches where kernel transition cost is negligible).
        _threadPool?.SetDispatchMode(seqLen == 1 ? DispatchMode.SpinWait : DispatchMode.EventBased);

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
        var repackedLayers = _weights.RepackedLayers;
        int numLayers = DebugMaxLayers switch
        {
            < 0 => 0,
            0 => Config.NumLayers,
            _ => Math.Min(DebugMaxLayers, Config.NumLayers)
        };

        // MLA cache lifecycle: allocated lazily on the first MLA forward
        // pass, reset when positions[0] == 0 so successive unrelated calls
        // (integration tests, multiple prompts, …) don't reuse stale KV.
        // Phase A (default) uses MlaExpandedKvState; Phase B / Phase C use
        // the smaller MlaLatentKvState. Phase C (UseHybridMlaCache) shares
        // the Phase B cache layout verbatim — the only difference is which
        // kernel consumes it (absorbed decode, expand-then-MHA prefill).
        // UseLatentCache and UseHybridMlaCache are mutually exclusive.
        if (Config.MlaConfig is not null)
        {
            var mla = Config.MlaConfig;
            if (mla.UseLatentCache && mla.UseHybridMlaCache)
                throw new InvalidOperationException(
                    "MlaConfig.UseLatentCache and MlaConfig.UseHybridMlaCache are mutually exclusive.");

            if (mla.UseLatentCache || mla.UseHybridMlaCache)
            {
                if (_mlaLatentKvState is null)
                {
                    _mlaLatentKvState = new MlaLatentKvState(
                        numLayers: Config.NumLayers,
                        maxSeqLen: Config.MaxSequenceLength,
                        kvLoraRank: mla.KvLoraRank,
                        qkRopeHeadDim: mla.QkRopeHeadDim);
                }
                if (positions[0] == 0)
                    _mlaLatentKvState.Reset();
            }
            else
            {
                if (_mlaKvState is null)
                {
                    _mlaKvState = new MlaExpandedKvState(
                        numLayers: Config.NumLayers,
                        maxSeqLen: Config.MaxSequenceLength,
                        numHeads: Config.NumAttentionHeads,
                        qkNopeHeadDim: mla.QkNopeHeadDim,
                        vHeadDim: mla.VHeadDim,
                        qkRopeHeadDim: mla.QkRopeHeadDim);
                }
                if (positions[0] == 0)
                    _mlaKvState.Reset();
            }
        }

        for (int layer = 0; layer < numLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];
            var rl = repackedLayers?[layer];

            // Declared once for the whole layer so both the GQA and MLA
            // paths share the same input-quantisation scratch region.
            byte* inputQ8Scratch = (byte*)_state.InputQ8Scratch;

            // a. Copy hiddenState → residual
            new Span<float>(hidden, seqLen * hiddenSize).CopyTo(new Span<float>(residual, seqLen * hiddenSize));

            // ── MLA branch (DeepSeek-V2/V3) ──────────────────────────────
            // Routes through the standalone MlaAttention kernel: RMSNorm → Q
            // path (LoRA or monolithic) → KV path (LoRA + MQA-shared rope-K)
            // → decoupled RoPE on the rope sub-dim only → per-head
            // scaled-dot-product attention with causal mask → o_proj.
            //
            // Cache: the kernel writes new K_nope / V / K_pe into the
            // persistent per-layer _mlaKvState store at offset
            // currentLength[layer] and attends over all (currentLength +
            // seqLen) tokens. This is the "non-absorbed reference" path per
            // the P2.3 plan — it matches the cacheless kernel numerically
            // and unblocks generation-loop tests on DeepSeek. Phase B
            // (latent compression + W_UK absorption) will layer on top,
            // using this as the correctness oracle. The caller-supplied
            // IKvCache is still ignored for MLA layers (shape-incompatible).
            if (lw.Mla is not null)
            {
                // RMSNorm per token into normOut (MLA kernel consumes the
                // normalised hidden state).
                for (int t = 0; t < seqLen; t++)
                {
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                        lw.AttnNormWeight, eps,
                        new Span<float>(normOut + t * hiddenSize, hiddenSize));
                }

                MlaLayerWeights mlaW = lw.Mla!;
                int qTotalElems = mlaW.NumHeads * (mlaW.QkNopeHeadDim + mlaW.QkRopeHeadDim);
                int kvAElems = mlaW.KvLoraRank + mlaW.QkRopeHeadDim;
                int kvBElems = mlaW.NumHeads * (mlaW.QkNopeHeadDim + mlaW.VHeadDim);
                int oElems = hiddenSize * (mlaW.NumHeads * mlaW.VHeadDim);
                int qAElems = mlaW.QLoraRank > 0 ? mlaW.QLoraRank * hiddenSize : 0;
                int qBElems = mlaW.QLoraRank > 0 ? qTotalElems * mlaW.QLoraRank : 0;
                int qMonoElems = mlaW.QLoraRank > 0 ? 0 : qTotalElems * hiddenSize;

                int ropeHalf = mlaW.QkRopeHeadDim / 2;
                int ropeTableLen = _state.CosTable.Length;

                float mlaScaleMultiplier = Config.MlaConfig!.ComputeYarnSoftmaxScaleMultiplier();
                if (_mlaLatentKvState is not null)
                {
                    // Phase B (pure absorbed) OR Phase C (hybrid
                    // expand-prefill / absorbed-decode) — both share the
                    // latent cache layout; the config flag picks the kernel.
                    bool hybrid = Config.MlaConfig!.UseHybridMlaCache;
                    if (hybrid)
                    {
                        MlaAttention.ExecuteLatentHybrid(
                            hidden: new ReadOnlySpan<float>(normOut, seqLen * hiddenSize),
                            output: new Span<float>(attnOut, seqLen * hiddenSize),
                            seqLen: seqLen,
                            positionOffset: positions[0],
                            hiddenSize: hiddenSize,
                            numHeads: mlaW.NumHeads,
                            qkNopeHeadDim: mlaW.QkNopeHeadDim,
                            qkRopeHeadDim: mlaW.QkRopeHeadDim,
                            vHeadDim: mlaW.VHeadDim,
                            qLoraRank: mlaW.QLoraRank,
                            kvLoraRank: mlaW.KvLoraRank,
                            rmsNormEps: eps,
                            ropeCosTable: _state.CosTable.AsSpan(0, ropeTableLen),
                            ropeSinTable: _state.SinTable.AsSpan(0, ropeTableLen),
                            qAProj: qAElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QAProj, qAElems) : ReadOnlySpan<float>.Empty,
                            qALayernormWeight: mlaW.QALayernormWeight ?? (ReadOnlySpan<float>)ReadOnlySpan<float>.Empty,
                            qBProj: qBElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QBProj, qBElems) : ReadOnlySpan<float>.Empty,
                            qProj: qMonoElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QProj, qMonoElems) : ReadOnlySpan<float>.Empty,
                            kvAProjWithMqa: new ReadOnlySpan<float>((void*)mlaW.KvAProjWithMqa, kvAElems * hiddenSize),
                            kvALayernormWeight: mlaW.KvALayernormWeight,
                            kvBProj: new ReadOnlySpan<float>((void*)mlaW.KvBProj, kvBElems * mlaW.KvLoraRank),
                            oProj: new ReadOnlySpan<float>((void*)lw.OWeight, oElems),
                            cachedLatent: _mlaLatentKvState.GetLatentPointer(layer),
                            cachedKPe: _mlaLatentKvState.GetKPePointer(layer),
                            cachedLength: _mlaLatentKvState.GetCurrentLength(layer),
                            attnScaleMultiplier: mlaScaleMultiplier);
                    }
                    else
                    {
                        MlaAttention.ExecuteLatent(
                            hidden: new ReadOnlySpan<float>(normOut, seqLen * hiddenSize),
                            output: new Span<float>(attnOut, seqLen * hiddenSize),
                            seqLen: seqLen,
                            positionOffset: positions[0],
                            hiddenSize: hiddenSize,
                            numHeads: mlaW.NumHeads,
                            qkNopeHeadDim: mlaW.QkNopeHeadDim,
                            qkRopeHeadDim: mlaW.QkRopeHeadDim,
                            vHeadDim: mlaW.VHeadDim,
                            qLoraRank: mlaW.QLoraRank,
                            kvLoraRank: mlaW.KvLoraRank,
                            rmsNormEps: eps,
                            ropeCosTable: _state.CosTable.AsSpan(0, ropeTableLen),
                            ropeSinTable: _state.SinTable.AsSpan(0, ropeTableLen),
                            qAProj: qAElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QAProj, qAElems) : ReadOnlySpan<float>.Empty,
                            qALayernormWeight: mlaW.QALayernormWeight ?? (ReadOnlySpan<float>)ReadOnlySpan<float>.Empty,
                            qBProj: qBElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QBProj, qBElems) : ReadOnlySpan<float>.Empty,
                            qProj: qMonoElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QProj, qMonoElems) : ReadOnlySpan<float>.Empty,
                            kvAProjWithMqa: new ReadOnlySpan<float>((void*)mlaW.KvAProjWithMqa, kvAElems * hiddenSize),
                            kvALayernormWeight: mlaW.KvALayernormWeight,
                            kvBProj: new ReadOnlySpan<float>((void*)mlaW.KvBProj, kvBElems * mlaW.KvLoraRank),
                            oProj: new ReadOnlySpan<float>((void*)lw.OWeight, oElems),
                            cachedLatent: _mlaLatentKvState.GetLatentPointer(layer),
                            cachedKPe: _mlaLatentKvState.GetKPePointer(layer),
                            cachedLength: _mlaLatentKvState.GetCurrentLength(layer),
                            attnScaleMultiplier: mlaScaleMultiplier);
                    }
                    _mlaLatentKvState.Advance(layer, seqLen);
                }
                else
                {
                    // Phase A — expanded cache + standard per-head attention.
                    MlaAttention.Execute(
                        hidden: new ReadOnlySpan<float>(normOut, seqLen * hiddenSize),
                        output: new Span<float>(attnOut, seqLen * hiddenSize),
                        seqLen: seqLen,
                        positionOffset: positions[0],
                        hiddenSize: hiddenSize,
                        numHeads: mlaW.NumHeads,
                        qkNopeHeadDim: mlaW.QkNopeHeadDim,
                        qkRopeHeadDim: mlaW.QkRopeHeadDim,
                        vHeadDim: mlaW.VHeadDim,
                        qLoraRank: mlaW.QLoraRank,
                        kvLoraRank: mlaW.KvLoraRank,
                        rmsNormEps: eps,
                        ropeCosTable: _state.CosTable.AsSpan(0, ropeTableLen),
                        ropeSinTable: _state.SinTable.AsSpan(0, ropeTableLen),
                        qAProj: qAElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QAProj, qAElems) : ReadOnlySpan<float>.Empty,
                        qALayernormWeight: mlaW.QALayernormWeight ?? (ReadOnlySpan<float>)ReadOnlySpan<float>.Empty,
                        qBProj: qBElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QBProj, qBElems) : ReadOnlySpan<float>.Empty,
                        qProj: qMonoElems > 0 ? new ReadOnlySpan<float>((void*)mlaW.QProj, qMonoElems) : ReadOnlySpan<float>.Empty,
                        kvAProjWithMqa: new ReadOnlySpan<float>((void*)mlaW.KvAProjWithMqa, kvAElems * hiddenSize),
                        kvALayernormWeight: mlaW.KvALayernormWeight,
                        kvBProj: new ReadOnlySpan<float>((void*)mlaW.KvBProj, kvBElems * mlaW.KvLoraRank),
                        oProj: new ReadOnlySpan<float>((void*)lw.OWeight, oElems),
                        attnScaleMultiplier: mlaScaleMultiplier,
                        cachedKNope: _mlaKvState!.GetKNopePointer(layer),
                        cachedV: _mlaKvState.GetVPointer(layer),
                        cachedKPe: _mlaKvState.GetKPePointer(layer),
                        cachedLength: _mlaKvState.GetCurrentLength(layer),
                        loraAdapter: _currentAdapter,
                        loraLayer: layer);
                    _mlaKvState.Advance(layer, seqLen);
                }

                // Bias on o_proj (rare — DeepSeek doesn't ship one by default).
                AddBias(lw.OBias, attnOut, hiddenSize, seqLen);

                // Residual add: attnOut + residual → hidden
                for (int t = 0; t < seqLen; t++)
                {
                    Add.Execute(
                        new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                        new ReadOnlySpan<float>(attnOut + t * hiddenSize, hiddenSize),
                        new Span<float>(hidden + t * hiddenSize, hiddenSize));
                }

                // Prepare residual for FFN.
                new Span<float>(hidden, seqLen * hiddenSize).CopyTo(new Span<float>(residual, seqLen * hiddenSize));

                // Fall through to the standard FFN branch (dense OR MoE,
                // decided by lw.Moe). Keep the original code path below by
                // goto-less control: set a flag and skip the GQA attention
                // code.
                goto FfnBranch;
            }

            // b. RMSNorm + Pre-quantize + Q/K/V projections
            // When a LoRA adapter is active we need the F32 normalised
            // hidden state (normOut) to feed LoraDelta — the fused
            // RmsNormQuantize decode path skips that intermediate. Force
            // the unfused path in that case.
            bool adapterActive = _currentAdapter is not null;
            // Phase 4d.5 / Gap 2: hoist preQuantNorm out of the decode/prefill
            // sub-branches so the LoRA delta call site (Q8_0-B fast path) can
            // re-use the buffer for stage 1. Pre-LoRA-Q8_0 this was scoped
            // inside each sub-branch.
            byte* preQuantNormQkv = null;
            if (seqLen == 1 && _threadPool != null && !adapterActive)
            {
                // Decode path: try fused RmsNorm+Quantize (skips normOut intermediate)
                byte* preQuantNorm = null;
                if (IsCompatiblePreQuant(lw.QQuantType, lw.KQuantType)
                    && IsCompatiblePreQuant(lw.QQuantType, lw.VQuantType))
                {
                    preQuantNorm = FusedOps.RmsNormQuantize(hidden, lw.AttnNormWeight, eps,
                        inputQ8Scratch, hiddenSize, lw.QQuantType);
                }

                if (preQuantNorm == null)
                {
                    // Fallback: unfused (F32/F16 weights or cross-family projections)
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(hidden, hiddenSize),
                        lw.AttnNormWeight, eps,
                        new Span<float>(normOut, hiddenSize));
                    preQuantNorm = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, 1, lw.QQuantType);
                }

                FusedQkvDecode(in lw, normOut, preQuantNorm, q, k, v);
                preQuantNormQkv = preQuantNorm;
            }
            else
            {
                // Prefill path: unfused RmsNorm + Quantize + individual projections
                for (int t = 0; t < seqLen; t++)
                {
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                        lw.AttnNormWeight, eps,
                        new Span<float>(normOut + t * hiddenSize, hiddenSize));
                }

                byte* preQuantNorm = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, seqLen, lw.QQuantType);

                var rwQ = rl?.Q ?? default;
                var rwK = rl?.K ?? default;
                var rwV = rl?.V ?? default;
                GemmInterleaved(lw.QWeight, lw.QQuantType, normOut, q, lw.QOutputDim, lw.QInputDim, seqLen,
                    preQuantNorm, in rwQ);
                GemmInterleaved(lw.KWeight, lw.KQuantType, normOut, k, lw.KOutputDim, lw.KInputDim, seqLen,
                    IsCompatiblePreQuant(lw.QQuantType, lw.KQuantType) ? preQuantNorm : null, in rwK);
                GemmInterleaved(lw.VWeight, lw.VQuantType, normOut, v, lw.VOutputDim, lw.VInputDim, seqLen,
                    IsCompatiblePreQuant(lw.QQuantType, lw.VQuantType) ? preQuantNorm : null, in rwV);
                preQuantNormQkv = preQuantNorm;
            }

            // Optional bias: y = Wx + b (no-op when null)
            AddBias(lw.QBias, q, lw.QOutputDim, seqLen);
            AddBias(lw.KBias, k, lw.KOutputDim, seqLen);
            AddBias(lw.VBias, v, lw.VOutputDim, seqLen);

            // LoRA delta (q/k/v): y += scale * (normOut · B) · A. No-op when
            // no adapter is active. Applied AFTER bias and BEFORE QK-norm /
            // RoPE so the delta contributes to the same downstream pipeline
            // as the base projection. F32 normOut is guaranteed materialised
            // here (we forced the unfused path above when adapter is active).
            //
            // Phase 4d.5 / Gap 2: when the base projection is Q8_0 the
            // `preQuantNormQkv` buffer is the Q8_0-encoded F32 input. We hand
            // that to ApplyLoraDelta so a Q8_0-B adapter's stage 1 can re-use
            // the buffer via `GemmQ8_0(preQuantizedInput=preQuantNormQkv)`,
            // skipping the activation quantise step that Phase 4d.4 had to
            // pay per-projection. Re-quantised path (`QuantizeInput` returning
            // null) drops through to the F32 / dequant-once fallback as before.
            if (_currentAdapter is not null)
            {
                // preQuantNormQkv is only valid for k/v when K/V quant types
                // are compatible with Q (same IsCompatiblePreQuant check the
                // base GEMM uses for the shared-input optimisation).
                byte* preQ_q = preQuantNormQkv;
                byte* preQ_k = (preQ_q is not null && IsCompatiblePreQuant(lw.QQuantType, lw.KQuantType)) ? preQ_q : null;
                byte* preQ_v = (preQ_q is not null && IsCompatiblePreQuant(lw.QQuantType, lw.VQuantType)) ? preQ_q : null;
                ApplyLoraDelta(layer, "q_proj", normOut, q, seqLen, lw.QInputDim, lw.QOutputDim,
                               preQ_q, lw.QQuantType);
                ApplyLoraDelta(layer, "k_proj", normOut, k, seqLen, lw.KInputDim, lw.KOutputDim,
                               preQ_k, lw.KQuantType);
                ApplyLoraDelta(layer, "v_proj", normOut, v, seqLen, lw.VInputDim, lw.VOutputDim,
                               preQ_v, lw.VQuantType);
            }

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

                if (kvCache is IQuantizedKvCache qkvCache)
                {
                    // Quantized path: dequantize KV tiles on-the-fly during attention
                    Attention.Execute(q, qkvCache, layer, attnOut,
                        seqLen, seqKv, numHeads, numKvHeads, headDim, positions[0], _threadPool,
                        _slidingWindowSize);
                }
                else
                {
                    var cachedK = kvCache.GetKeysRef(layer);
                    var cachedV = kvCache.GetValuesRef(layer);

                    Attention.Execute(q, (float*)cachedK.DataPointer, (float*)cachedV.DataPointer, attnOut,
                        seqLen, seqKv, numHeads, numKvHeads, headDim, positions[0], _threadPool,
                        _slidingWindowSize);
                }
            }
            else
            {
                Attention.Execute(q, k, v, attnOut,
                    seqLen, seqLen, numHeads, numKvHeads, headDim, 0, _threadPool,
                    _slidingWindowSize);
            }

            // f. Batched O projection
            byte* preQuantAttn = QuantizeInput(attnOut, inputQ8Scratch, numHeads * headDim, seqLen, lw.OQuantType);
            var rwO = rl?.O ?? default;
            GemmInterleaved(lw.OWeight, lw.OQuantType, attnOut, normOut, lw.OOutputDim, lw.OInputDim, seqLen,
                preQuantAttn, in rwO);
            AddBias(lw.OBias, normOut, lw.OOutputDim, seqLen);

            // LoRA delta (o_proj): y += scale * (attnOut · B) · A.
            // Phase 4d.5 / Gap 2: pass preQuantAttn so Q8_0-B adapter stage 1
            // re-uses the activation Q8_0 buffer.
            if (_currentAdapter is not null)
            {
                ApplyLoraDelta(layer, "o_proj", attnOut, normOut, seqLen, lw.OInputDim, lw.OOutputDim,
                               preQuantAttn, lw.OQuantType);
            }

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

            FfnBranch:
            // ── MoE branch ──────────────────────────────────────────────
            // Mixtral-convention top-k dense routing replaces the dense FFN
            // block entirely. Takes post-attn hidden + FFN RMSNorm weight,
            // runs router + top-k experts, writes into normOut, then residual
            // adds into hidden and continues to the next layer. No R4 repack
            // (expert GEMMs are tiny), no pre-quantise (experts are F32).
            if (lw.Moe is not null)
            {
                // FFN RMSNorm per token into normOut.
                for (int t = 0; t < seqLen; t++)
                {
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                        lw.FfnNormWeight, eps,
                        new Span<float>(normOut + t * hiddenSize, hiddenSize));
                }

                MoeLayerWeights moe = lw.Moe!;
                // Route through the shared-expert-aware overload iff we need
                // shared-expert addition OR the raw-softmax (non-renormalised)
                // Qwen1.5-MoE gating. The simple Mixtral path stays the call
                // target for the common case.
                if (moe.HasSharedExpert || !moe.NormTopKProb)
                {
                    ReadOnlySpan<float> sharedGateSpan = moe.SharedExpertGate is not null
                        ? moe.SharedExpertGate.AsSpan()
                        : ReadOnlySpan<float>.Empty;
                    MoeSwiGluMlp.ExecuteWithSharedExpert(
                        hidden: new ReadOnlySpan<float>(normOut, seqLen * hiddenSize),
                        gateWeights: moe.Gate,
                        expertsW1: moe.W1,
                        expertsW2: moe.W2,
                        expertsW3: moe.W3,
                        output: new Span<float>(normOut, seqLen * hiddenSize),
                        numExperts: moe.NumExperts,
                        numExpertsPerTok: moe.NumExpertsPerTok,
                        hiddenSize: hiddenSize,
                        intermediateSize: moe.IntermediateSize,
                        seqLen: seqLen,
                        normTopKProb: moe.NormTopKProb,
                        sharedGateProj: moe.SharedGateProj,
                        sharedUpProj: moe.SharedUpProj,
                        sharedDownProj: moe.SharedDownProj,
                        sharedIntermediateSize: moe.SharedIntermediateSize,
                        sharedExpertGate: sharedGateSpan,
                        loraAdapter: _currentAdapter,
                        loraLayer: layer);
                }
                else
                {
                    MoeSwiGluMlp.Execute(
                        hidden: new ReadOnlySpan<float>(normOut, seqLen * hiddenSize),
                        gateWeights: moe.Gate,
                        expertsW1: moe.W1,
                        expertsW2: moe.W2,
                        expertsW3: moe.W3,
                        output: new Span<float>(normOut, seqLen * hiddenSize),
                        numExperts: moe.NumExperts,
                        numExpertsPerTok: moe.NumExpertsPerTok,
                        hiddenSize: hiddenSize,
                        intermediateSize: moe.IntermediateSize,
                        seqLen: seqLen,
                        loraAdapter: _currentAdapter,
                        loraLayer: layer);
                }

                // Residual add (per token) → hidden. Same as dense path.
                for (int t = 0; t < seqLen; t++)
                {
                    Add.Execute(
                        new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                        new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                        new Span<float>(hidden + t * hiddenSize, hiddenSize));
                }
                continue;
            }

            // i. FFN RMSNorm + Pre-quantize + Gate/Up projections
            // When a LoRA adapter is active we need F32 normOut for delta —
            // skip the fused decode path so it materialises (same trick as Q/K/V).
            bool ffnAdapterActive = _currentAdapter is not null;
            // Phase 4d.5 / Gap 2: hoist preQuantFfn out of both sub-branches
            // so the LoRA delta call site can reuse the activation Q8_0
            // buffer for stage 1.
            byte* preQuantFfnHoisted = null;
            if (seqLen == 1 && _threadPool != null && !ffnAdapterActive)
            {
                // Decode path: try fused RmsNorm+Quantize (skips normOut intermediate)
                byte* preQuantFfn = null;
                if (IsCompatiblePreQuant(lw.GateQuantType, lw.UpQuantType))
                {
                    preQuantFfn = FusedOps.RmsNormQuantize(hidden, lw.FfnNormWeight, eps,
                        inputQ8Scratch, hiddenSize, lw.GateQuantType);
                }

                if (preQuantFfn == null)
                {
                    // Fallback: unfused (F32/F16 weights or cross-family projections)
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(hidden, hiddenSize),
                        lw.FfnNormWeight, eps,
                        new Span<float>(normOut, hiddenSize));
                    preQuantFfn = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, 1, lw.GateQuantType);
                }

                FusedGateUpDecode(in lw, normOut, preQuantFfn, ffnGate, ffnUp);
                preQuantFfnHoisted = preQuantFfn;
            }
            else
            {
                // Prefill path: unfused RmsNorm + Quantize + individual projections
                for (int t = 0; t < seqLen; t++)
                {
                    RmsNorm.Execute(
                        new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                        lw.FfnNormWeight, eps,
                        new Span<float>(normOut + t * hiddenSize, hiddenSize));
                }

                byte* preQuantFfn = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, seqLen, lw.GateQuantType);

                var rwGate = rl?.Gate ?? default;
                var rwUp = rl?.Up ?? default;
                GemmInterleaved(lw.GateWeight, lw.GateQuantType, normOut, ffnGate, lw.GateOutputDim, lw.GateInputDim, seqLen,
                    preQuantFfn, in rwGate);
                GemmInterleaved(lw.UpWeight, lw.UpQuantType, normOut, ffnUp, lw.UpOutputDim, lw.UpInputDim, seqLen,
                    IsCompatiblePreQuant(lw.GateQuantType, lw.UpQuantType) ? preQuantFfn : null, in rwUp);
                preQuantFfnHoisted = preQuantFfn;
            }
            AddBias(lw.GateBias, ffnGate, lw.GateOutputDim, seqLen);
            AddBias(lw.UpBias, ffnUp, lw.UpOutputDim, seqLen);

            // LoRA delta (gate/up): y += scale * (normOut · B) · A.
            // Phase 4d.5 / Gap 2: pass the hoisted preQuantFfn so the Q8_0-B
            // adapter stage 1 re-uses the activation Q8_0 buffer.
            if (_currentAdapter is not null)
            {
                byte* preQ_gate = preQuantFfnHoisted;
                byte* preQ_up = (preQ_gate is not null && IsCompatiblePreQuant(lw.GateQuantType, lw.UpQuantType)) ? preQ_gate : null;
                ApplyLoraDelta(layer, "gate_proj", normOut, ffnGate, seqLen, lw.GateInputDim, lw.GateOutputDim,
                               preQ_gate, lw.GateQuantType);
                ApplyLoraDelta(layer, "up_proj", normOut, ffnUp, seqLen, lw.UpInputDim, lw.UpOutputDim,
                               preQ_up, lw.UpQuantType);
            }

            // Fused SwiGLU: SiLU(gate) * up in a single tiled pass (per token)
            for (int t = 0; t < seqLen; t++)
            {
                float* gateT = ffnGate + t * intermediateSize;
                float* upT = ffnUp + t * intermediateSize;
                float* siluT = siluOut + t * intermediateSize;

                FusedOps.SwiGLU(
                    new ReadOnlySpan<float>(gateT, intermediateSize),
                    new ReadOnlySpan<float>(upT, intermediateSize),
                    new Span<float>(siluT, intermediateSize));
            }

            // Pre-quantize siluOutput for Down projection (different input dim = intermediateSize)
            byte* preQuantSilu = QuantizeInput(siluOut, inputQ8Scratch, intermediateSize, seqLen, lw.DownQuantType);

            // Batched Down projection (output into normOut as scratch)
            var rwDown = rl?.Down ?? default;
            GemmInterleaved(lw.DownWeight, lw.DownQuantType, siluOut, normOut, lw.DownOutputDim, lw.DownInputDim, seqLen,
                preQuantSilu, in rwDown);
            AddBias(lw.DownBias, normOut, lw.DownOutputDim, seqLen);

            // LoRA delta (down_proj): y += scale * (siluOut · B) · A.
            // Input is post-SwiGLU (siluOut), not normOut. The base GEMM
            // already wrote into normOut, so we accumulate delta in place.
            // Phase 4d.5 / Gap 2: pass preQuantSilu so Q8_0-B adapter stage 1
            // re-uses the activation Q8_0 buffer.
            if (_currentAdapter is not null)
            {
                ApplyLoraDelta(layer, "down_proj", siluOut, normOut, seqLen, lw.DownInputDim, lw.DownOutputDim,
                               preQuantSilu, lw.DownQuantType);
            }

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
    }

    /// <summary>
    /// LM head GEMM at <paramref name="seqLen"/> rows. Reads the final hidden state
    /// from <c>_state.HiddenState[0..seqLen*hiddenSize]</c> (left there by
    /// <see cref="RunLayersAndFinalNormCore"/>), writes logits into
    /// <c>_state.Logits</c>, allocates a freshly-owned tensor and copies the logits
    /// into it. Caller disposes the tensor.
    /// </summary>
    private unsafe ITensor RunLmHead(int seqLen, int deviceId)
    {
        int vocabSize = Config.VocabSize;
        float* hidden = (float*)_state.HiddenState;
        float* logits = (float*)_state.Logits;

        var rwOutput = _weights.RepackedOutput ?? default;
        GemmInterleaved(_weights.OutputWeight, _weights.OutputQuantType,
            hidden, logits, _weights.OutputOutputDim, _weights.OutputInputDim, seqLen,
            null, in rwOutput);

        var shape = new TensorShape(seqLen, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
        new Span<float>(logits, seqLen * vocabSize).CopyTo(
            new Span<float>((void*)result.DataPointer, seqLen * vocabSize));
        return result;
    }

    /// <summary>
    /// Fused forward across multiple in-flight sequences. Sequences are partitioned
    /// into a SIMPLE subgroup (GQA / MHA / MQA, no MLA, no MoE, no adapter) and a
    /// COMPLEX subgroup (any of those features present). The simple subgroup runs
    /// through <see cref="RunLayersAndFinalNormBatched"/>, which fuses the per-layer
    /// Q/K/V/O/gate/up/down GEMMs across sequences (one big <c>[Σ N_i, hidden] × W</c>
    /// dispatch instead of N small ones — the matmul-fusion win this method exists for).
    /// Complex sequences fall back to a per-seq <see cref="RunLayersAndFinalNormCore"/>
    /// loop. The lm_head GEMM is fused across the union of both subgroups.
    /// </summary>
    /// <remarks>
    /// <para>Phase 5a fused the lm_head only. Phase 5b adds intra-block matmul fusion
    /// for the simple subgroup. Attention still runs per-seq (each sequence has its
    /// own KV cache, positions, and position offset) — only the GEMMs at the seam
    /// of the attention block are fused.</para>
    /// <para>Parity contract: byte-identical per-element logits vs the per-seq
    /// <see cref="Forward(System.ReadOnlySpan{int},System.ReadOnlySpan{int},int,IKvCache?)"/>
    /// loop. Each batched-GEMM output element is an independent dot product over a
    /// fixed-length contraction axis, so per-row results don't depend on the batched
    /// row count.</para>
    /// </remarks>
    public IReadOnlyList<ITensor> ForwardBatch(
        IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();
        if (requests.Count == 1)
        {
            var r0 = requests[0];
            return new[] { Forward(r0.TokenIds.Span, r0.Positions.Span,
                                   deviceId, r0.KvCache, r0.Adapter) };
        }

        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        int totalTokens = 0;
        foreach (var r in requests) totalTokens += r.TokenIds.Length;

        _state.EnsureCapacity(totalTokens);

        // Partition into simple (matmul-fused) vs complex (per-seq fallback)
        // subgroups. The model-level "has complex layer" check is one-shot — if
        // ANY layer is MLA or MoE, the batched path can't fuse safely (the per-
        // layer branch executes for every sequence in the batch, so even a
        // simple-looking sequence would hit the MLA/MoE branch). LoRA adapters
        // are per-sequence, so each request is judged individually.
        bool modelHasComplexLayer = ModelHasMlaOrMoeLayer();
        Span<int> simpleIdxs = requests.Count <= 256
            ? stackalloc int[requests.Count]
            : new int[requests.Count];
        Span<int> complexIdxs = requests.Count <= 256
            ? stackalloc int[requests.Count]
            : new int[requests.Count];
        int simpleCount = 0;
        int complexCount = 0;
        int simpleTotalTokens = 0;
        for (int i = 0; i < requests.Count; i++)
        {
            var r = requests[i];
            bool seqComplex = modelHasComplexLayer || r.Adapter is not null;
            if (seqComplex)
            {
                complexIdxs[complexCount++] = i;
            }
            else
            {
                simpleIdxs[simpleCount++] = i;
                simpleTotalTokens += r.TokenIds.Length;
            }
        }

        // Per-batch snapshot buffer: each per-seq RunLayersAndFinalNormCore call
        // and the batched simple-subgroup pass all write final hidden states into
        // _state.HiddenState (overlapping). We copy each seq's slice OUT to its
        // index-ordered offset in `batched` immediately after producing it, then
        // copy the whole thing BACK into _state.HiddenState for the batched
        // lm_head dispatch. The total snapshot footprint is the same as Phase 5a
        // (totalTokens * hidden * 4 bytes).
        var pool = ArrayPool<float>.Shared;
        float[] batched = pool.Rent(totalTokens * hiddenSize);
        try
        {
            // Per-seq token offsets in the original (caller-supplied) request
            // order — drives the lm_head logits-split-back step and the per-seq
            // copy destination in `batched`.
            Span<int> tokOffsets = requests.Count <= 256
                ? stackalloc int[requests.Count]
                : new int[requests.Count];
            int running = 0;
            for (int i = 0; i < requests.Count; i++)
            {
                tokOffsets[i] = running;
                running += requests[i].TokenIds.Length;
            }

            // ── Simple subgroup: batched matmul path ────────────────────────
            // Writes its sequences' final hidden states into _state.HiddenState
            // packed in the order of `simpleIdxs[0..simpleCount]`. We snapshot
            // each one out into its original-index offset in `batched`.
            if (simpleCount > 0)
            {
                RunLayersAndFinalNormBatched(requests, simpleIdxs[..simpleCount], simpleTotalTokens);

                int packedOff = 0;
                float* hidden = (float*)_state.HiddenState;
                for (int s = 0; s < simpleCount; s++)
                {
                    int origIdx = simpleIdxs[s];
                    int n = requests[origIdx].TokenIds.Length;
                    new Span<float>(hidden + packedOff * hiddenSize, n * hiddenSize)
                        .CopyTo(batched.AsSpan(tokOffsets[origIdx] * hiddenSize, n * hiddenSize));
                    packedOff += n;
                }
            }

            // ── Complex subgroup: per-seq fallback (Phase 5a behaviour) ─────
            for (int c = 0; c < complexCount; c++)
            {
                int origIdx = complexIdxs[c];
                var r = requests[origIdx];
                int n = r.TokenIds.Length;
                if (r.Adapter is not null)
                {
                    ValidateAdapterForModel(r.Adapter);
                    LoraStage2.PrewarmAdapter(r.Adapter as LoraAdapter);
                    _currentAdapter = r.Adapter;
                }
                try
                {
                    RunLayersAndFinalNormCore(r.TokenIds.Span, r.Positions.Span, r.KvCache);
                    new Span<float>((float*)_state.HiddenState, n * hiddenSize)
                        .CopyTo(batched.AsSpan(tokOffsets[origIdx] * hiddenSize, n * hiddenSize));
                }
                finally
                {
                    if (r.Adapter is not null) _currentAdapter = null;
                }
            }

            // Stack the per-seq snapshots back into _state.HiddenState in original
            // request order, then run one batched lm_head dispatch at seqLen = Σ N_i.
            batched.AsSpan(0, totalTokens * hiddenSize)
                .CopyTo(new Span<float>((float*)_state.HiddenState, totalTokens * hiddenSize));

            float* logitsPtr = (float*)_state.Logits;
            var rwOutput = _weights.RepackedOutput ?? default;
            GemmInterleaved(_weights.OutputWeight, _weights.OutputQuantType,
                (float*)_state.HiddenState, logitsPtr,
                _weights.OutputOutputDim, _weights.OutputInputDim, totalTokens,
                null, in rwOutput);

            // Split logits per-seq.
            var results = new ITensor[requests.Count];
            for (int i = 0; i < requests.Count; i++)
            {
                int n = requests[i].TokenIds.Length;
                int srcOff = tokOffsets[i];
                var shape = new TensorShape(n, vocabSize);
                var tensor = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
                new Span<float>(logitsPtr + (long)srcOff * vocabSize, n * vocabSize).CopyTo(
                    new Span<float>((void*)tensor.DataPointer, n * vocabSize));
                results[i] = tensor;
            }
            return results;
        }
        finally
        {
            pool.Return(batched);
        }
    }

    /// <summary>
    /// Returns true when any layer of this model uses MLA (DeepSeek-V2/V3) or
    /// MoE (Mixtral / Qwen-MoE / DeepSeek-V2/V3). Such layers carry per-layer
    /// kernels that aren't trivially batchable across sequences in the Phase 5b
    /// matmul-fused path, so the entire batch falls back to per-seq when this
    /// returns true.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private bool ModelHasMlaOrMoeLayer()
    {
        var layers = _weights.Layers;
        for (int i = 0; i < layers.Length; i++)
        {
            if (layers[i].Mla is not null || layers[i].Moe is not null) return true;
        }
        return false;
    }

    /// <summary>
    /// Phase 5b matmul-fused layer loop for the SIMPLE subgroup (GQA / MHA / MQA,
    /// no MLA / no MoE / no LoRA adapter). For each transformer layer:
    /// <list type="number">
    /// <item>Concat per-seq hidden states into a single <c>[Σ N_i, hidden]</c>
    ///   batched buffer (residual copy already does this via the packed layout
    ///   — sequences are stored contiguously in <c>_state.HiddenState</c>).</item>
    /// <item>One batched RMSNorm over <c>Σ N_i</c> rows.</item>
    /// <item>One batched QuantizeInput.</item>
    /// <item>One batched Q/K/V GEMM at <c>[Σ N_i, hidden] × [hidden, dim]</c>.</item>
    /// <item>Q/K/V outputs are sliced per-seq for RoPE + attention (each seq has
    ///   independent positions / position offset / KV cache).</item>
    /// <item>One batched O projection + residual.</item>
    /// <item>Same pattern for the FFN block (RMSNorm + gate/up GEMM + SwiGLU +
    ///   down GEMM + residual).</item>
    /// </list>
    /// At return, each simple sequence's final hidden state is packed contiguously
    /// in <c>_state.HiddenState</c> in <paramref name="simpleIdxs"/> order, having
    /// passed through the final RMSNorm.
    /// </summary>
    /// <remarks>
    /// Parity contract with <see cref="RunLayersAndFinalNormCore"/>: byte-identical
    /// per-element output. Each batched-GEMM output element is an independent dot
    /// product over a fixed-length contraction axis, so the FP accumulation order
    /// (and therefore the per-row result) does NOT depend on whether the GEMM
    /// processes 1 or <c>Σ N_i</c> rows.
    /// </remarks>
    [SkipLocalsInit]
    private unsafe void RunLayersAndFinalNormBatched(
        IReadOnlyList<SequenceForwardRequest> requests,
        ReadOnlySpan<int> simpleIdxs,
        int simpleTotalTokens)
    {
        int maxSeq = Config.MaxSequenceLength;
        // Validate positions per-seq (mirrors the per-seq core).
        for (int s = 0; s < simpleIdxs.Length; s++)
        {
            var positions = requests[simpleIdxs[s]].Positions.Span;
            for (int i = 0; i < positions.Length; i++)
            {
                if ((uint)positions[i] >= (uint)maxSeq)
                    throw new ArgumentOutOfRangeException(nameof(requests),
                        $"Position {positions[i]} at index {i} of sequence {simpleIdxs[s]} exceeds max sequence length {maxSeq}.");
            }
        }

        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int intermediateSize = Config.IntermediateSize;
        int kvStride = numKvHeads * headDim;
        int qStride = numHeads * headDim;
        float eps = Config.NormEpsilon;

        // Total tokens across simple seqs. The caller has already called
        // EnsureCapacity on _state for at least this much.
        int total = simpleTotalTokens;

        // EventBased: batched is by definition multi-token (we early-return at
        // requests.Count==1 above, so simpleCount + complexCount ≥ 2; even with
        // 4× decode the batched matmul is "prefill-shaped" relative to a 1-token
        // dispatch). The per-seq fallback path may flip back to SpinWait inside
        // RunLayersAndFinalNormCore — that's fine, both modes are independent.
        _threadPool?.SetDispatchMode(DispatchMode.EventBased);

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

        // Packed per-seq token offsets (into the batched [total, *] buffers).
        // simpleIdxs[s] gives the caller-supplied request index for sub-seq s,
        // packedOffsets[s] gives where that seq starts in the batched buffers.
        Span<int> packedOffsets = simpleIdxs.Length <= 256
            ? stackalloc int[simpleIdxs.Length]
            : new int[simpleIdxs.Length];
        int run = 0;
        for (int s = 0; s < simpleIdxs.Length; s++)
        {
            packedOffsets[s] = run;
            run += requests[simpleIdxs[s]].TokenIds.Length;
        }

        // 1. EMBEDDING LOOKUP — pack per-seq directly into the batched buffer.
        for (int s = 0; s < simpleIdxs.Length; s++)
        {
            var r = requests[simpleIdxs[s]];
            int n = r.TokenIds.Length;
            EmbeddingLookup(r.TokenIds.Span, hidden + (long)packedOffsets[s] * hiddenSize, hiddenSize);
        }

        // 2. TRANSFORMER LAYERS
        var repackedLayers = _weights.RepackedLayers;
        int numLayers = DebugMaxLayers switch
        {
            < 0 => 0,
            0 => Config.NumLayers,
            _ => Math.Min(DebugMaxLayers, Config.NumLayers)
        };

        for (int layer = 0; layer < numLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];
            var rl = repackedLayers?[layer];

            byte* inputQ8Scratch = (byte*)_state.InputQ8Scratch;

            // a. Copy hidden → residual (whole packed buffer).
            new Span<float>(hidden, total * hiddenSize).CopyTo(new Span<float>(residual, total * hiddenSize));

            // b. Batched RMSNorm: same per-row math as the prefill path of the
            // unfused loop (each row is an independent normalisation). Loop is
            // identical to RunLayersAndFinalNormCore's prefill RMSNorm.
            for (int t = 0; t < total; t++)
            {
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                    lw.AttnNormWeight, eps,
                    new Span<float>(normOut + t * hiddenSize, hiddenSize));
            }

            // c. Batched QuantizeInput + Q/K/V projections at n=total.
            byte* preQuantNorm = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, total, lw.QQuantType);

            var rwQ = rl?.Q ?? default;
            var rwK = rl?.K ?? default;
            var rwV = rl?.V ?? default;
            GemmInterleaved(lw.QWeight, lw.QQuantType, normOut, q, lw.QOutputDim, lw.QInputDim, total,
                preQuantNorm, in rwQ);
            GemmInterleaved(lw.KWeight, lw.KQuantType, normOut, k, lw.KOutputDim, lw.KInputDim, total,
                IsCompatiblePreQuant(lw.QQuantType, lw.KQuantType) ? preQuantNorm : null, in rwK);
            GemmInterleaved(lw.VWeight, lw.VQuantType, normOut, v, lw.VOutputDim, lw.VInputDim, total,
                IsCompatiblePreQuant(lw.QQuantType, lw.VQuantType) ? preQuantNorm : null, in rwV);

            // Optional bias (operates over all batched rows uniformly).
            AddBias(lw.QBias, q, lw.QOutputDim, total);
            AddBias(lw.KBias, k, lw.KOutputDim, total);
            AddBias(lw.VBias, v, lw.VOutputDim, total);

            // Optional QK-norms (Qwen3-style) — independently applied per row.
            if (lw.QNormWeight is not null)
                ApplyPerHeadNorm(lw.QNormWeight, q, numHeads, headDim, total, eps);
            if (lw.KNormWeight is not null)
                ApplyPerHeadNorm(lw.KNormWeight, k, numKvHeads, headDim, total, eps);

            // d/e. Per-sequence RoPE + Attention + KV-cache update. The Q/K/V
            // slices live at packedOffsets[s] in the batched buffers; we hand
            // each slice and the seq's own positions to the per-seq kernels.
            // Attention writes its output back into attnOut at the same offset,
            // re-stacking the post-attention tokens into a single packed buffer
            // ready for the next batched GEMM (O projection).
            // NOTE: original commit guarded RoPE with !Config.IsNoRopeLayer(layer)
            // (NoPE support from #208). #208 is not in this base, and the per-seq
            // Forward path on this base applies RoPE unconditionally, so we do the
            // same here to preserve byte-identical parity. When #208 lands, restore
            // the guard.
            for (int s = 0; s < simpleIdxs.Length; s++)
            {
                int origIdx = simpleIdxs[s];
                var r = requests[origIdx];
                int n = r.TokenIds.Length;
                int off = packedOffsets[s];
                var positions = r.Positions.Span;

                float* qSlice = q + (long)off * qStride;
                float* kSlice = k + (long)off * kvStride;
                float* vSlice = v + (long)off * kvStride;
                float* aSlice = attnOut + (long)off * qStride;

                RoPE.Execute(
                    new Span<float>(qSlice, n * qStride),
                    new Span<float>(kSlice, n * kvStride),
                    positions,
                    numHeads, numKvHeads, headDim, _ropeDim,
                    _state.CosTable, _state.SinTable, _ropeType);

                IKvCache kvCache = r.KvCache;
                // KV cache is required on the request — write new K/V then attend
                // over the cached range.
                var kRef = new TensorRef(n, kvStride, DType.Float32, -1, (nint)kSlice);
                var vRef = new TensorRef(n, kvStride, DType.Float32, -1, (nint)vSlice);
                kvCache.Update(kRef, vRef, positions, layer);

                int seqKv = kvCache.CurrentLength;
                if (kvCache is IQuantizedKvCache qkvCache)
                {
                    Attention.Execute(qSlice, qkvCache, layer, aSlice,
                        n, seqKv, numHeads, numKvHeads, headDim, positions[0], _threadPool,
                        _slidingWindowSize);
                }
                else
                {
                    var cachedK = kvCache.GetKeysRef(layer);
                    var cachedV = kvCache.GetValuesRef(layer);
                    Attention.Execute(qSlice, (float*)cachedK.DataPointer, (float*)cachedV.DataPointer, aSlice,
                        n, seqKv, numHeads, numKvHeads, headDim, positions[0], _threadPool,
                        _slidingWindowSize);
                }
            }

            // f. Batched O projection: [total, qStride] × [qStride, hidden] → [total, hidden] into normOut.
            byte* preQuantAttn = QuantizeInput(attnOut, inputQ8Scratch, qStride, total, lw.OQuantType);
            var rwO = rl?.O ?? default;
            GemmInterleaved(lw.OWeight, lw.OQuantType, attnOut, normOut, lw.OOutputDim, lw.OInputDim, total,
                preQuantAttn, in rwO);
            AddBias(lw.OBias, normOut, lw.OOutputDim, total);

            // g. Residual add: hidden ← residual + normOut (all batched rows).
            for (int t = 0; t < total; t++)
            {
                Add.Execute(
                    new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                    new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                    new Span<float>(hidden + t * hiddenSize, hiddenSize));
            }

            // h. Copy hidden → residual (snapshot for FFN block).
            new Span<float>(hidden, total * hiddenSize).CopyTo(new Span<float>(residual, total * hiddenSize));

            // i. Batched FFN RMSNorm + Gate/Up + SwiGLU + Down.
            for (int t = 0; t < total; t++)
            {
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                    lw.FfnNormWeight, eps,
                    new Span<float>(normOut + t * hiddenSize, hiddenSize));
            }

            byte* preQuantFfn = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, total, lw.GateQuantType);

            var rwGate = rl?.Gate ?? default;
            var rwUp = rl?.Up ?? default;
            GemmInterleaved(lw.GateWeight, lw.GateQuantType, normOut, ffnGate, lw.GateOutputDim, lw.GateInputDim, total,
                preQuantFfn, in rwGate);
            GemmInterleaved(lw.UpWeight, lw.UpQuantType, normOut, ffnUp, lw.UpOutputDim, lw.UpInputDim, total,
                IsCompatiblePreQuant(lw.GateQuantType, lw.UpQuantType) ? preQuantFfn : null, in rwUp);

            AddBias(lw.GateBias, ffnGate, lw.GateOutputDim, total);
            AddBias(lw.UpBias, ffnUp, lw.UpOutputDim, total);

            // Fused SwiGLU per row.
            for (int t = 0; t < total; t++)
            {
                float* gateT = ffnGate + t * intermediateSize;
                float* upT = ffnUp + t * intermediateSize;
                float* siluT = siluOut + t * intermediateSize;

                FusedOps.SwiGLU(
                    new ReadOnlySpan<float>(gateT, intermediateSize),
                    new ReadOnlySpan<float>(upT, intermediateSize),
                    new Span<float>(siluT, intermediateSize));
            }

            // Batched Down projection: [total, intermediate] × [intermediate, hidden] → [total, hidden] into normOut.
            byte* preQuantSilu = QuantizeInput(siluOut, inputQ8Scratch, intermediateSize, total, lw.DownQuantType);
            var rwDown = rl?.Down ?? default;
            GemmInterleaved(lw.DownWeight, lw.DownQuantType, siluOut, normOut, lw.DownOutputDim, lw.DownInputDim, total,
                preQuantSilu, in rwDown);
            AddBias(lw.DownBias, normOut, lw.DownOutputDim, total);

            // k. Final residual add.
            for (int t = 0; t < total; t++)
            {
                Add.Execute(
                    new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                    new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                    new Span<float>(hidden + t * hiddenSize, hiddenSize));
            }
        }

        // 3. FINAL NORM (in-place: hidden → hidden) over all batched rows.
        for (int t = 0; t < total; t++)
        {
            float* hiddenT = hidden + t * hiddenSize;
            float* normOutT = normOut + t * hiddenSize;

            RmsNorm.Execute(
                new ReadOnlySpan<float>(hiddenT, hiddenSize),
                _weights.OutputNormWeight,
                eps,
                new Span<float>(normOutT, hiddenSize));

            new Span<float>(normOutT, hiddenSize).CopyTo(new Span<float>(hiddenT, hiddenSize));
        }
    }

    /// <summary>
    /// Validates that <paramref name="adapter"/> is compatible with this
    /// model and that its targeted projections do not collide with
    /// out-of-scope MLA / MoE structures. Called once per LoRA-aware Forward.
    /// </summary>
    private void ValidateAdapterForModel(ILoraAdapter adapter)
    {
        if (!adapter.IsCompatible(Config))
            throw new InvalidOperationException(
                $"LoRA adapter '{adapter.Name}' is not compatible with the loaded model "
                + "(layer count, hidden size, or per-projection dimensions mismatch).");

        // Phase 4d.2: MLA / MoE rejections are lifted. The standard
        // ApplyLoraDelta call sites are only reached on non-MLA / dense FFN
        // layers (the MLA branch in Forward routes through MlaAttention which
        // has its own LoRA hooks; MoE routes through MoeSwiGluMlp). Adapters
        // that target standard q/k/v/o or gate/up/down on MLA / MoE layers
        // therefore pass through silently — applying the delta requires the
        // MLA-specific (q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj)
        // or per-expert (mlp.experts.{j}.{...}) projection names which the
        // PEFT loader will eventually emit. Until those code paths are wired,
        // a non-applicable target is a no-op rather than an error.
    }

    /// <summary>
    /// Applies the LoRA delta for <paramref name="projName"/> at
    /// <paramref name="layer"/> if the active adapter targets that site.
    /// No-op when there is no active adapter or no entry for this projection.
    /// <para>
    /// Phase 4d.5 / Gap 2 — when the caller has already quantised
    /// <paramref name="x"/> for the base projection's GEMM
    /// (<see cref="QuantizeInput"/>) and passes the resulting buffer as
    /// <paramref name="preQuantX"/>, AND <paramref name="preQuantXType"/> is
    /// <see cref="QuantizationType.Q8_0"/>, AND the adapter's B factor is
    /// <see cref="LoraWeightDType.Q8_0"/>, the LoRA stage-1 GEMM re-uses
    /// the pre-quantised buffer via
    /// <see cref="LoraDelta.ApplyQ8_0BWithPreQuantX"/> instead of dequanting B
    /// to F32 and running an F32 GEMM. This closes the residual −16% prefill
    /// regression the Phase 4d.4 dequant-once path left on the table on a
    /// Q8_0 base (Strix Halo / Llama-3.2-1B). The default arguments give the
    /// legacy F32 / dequant-once behaviour.
    /// </para>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ApplyLoraDelta(int layer, string projName,
                                float* x, float* y, int seqLen, int inputDim, int outputDim,
                                byte* preQuantX = null,
                                QuantizationType preQuantXType = QuantizationType.F32)
    {
        var adapter = _currentAdapter;
        if (adapter is null) return;
        var lora = adapter.GetLayerWeights(layer, projName);
        if (lora is not { } w) return;

        // Defensive shape check — IsCompatible already validated dims, but
        // we re-check at the call site so a bug in dim plumbing surfaces
        // here rather than as a silent OOB into x/y buffers.
        if (w.InputDim != inputDim || w.OutputDim != outputDim)
            throw new InvalidOperationException(
                $"LoRA adapter '{adapter.Name}' layer={layer} proj='{projName}' shape "
                + $"({w.InputDim}x{w.OutputDim}) does not match base projection "
                + $"({inputDim}x{outputDim}).");

        float scale = adapter.Alpha / adapter.Rank;

        // Phase 4d.6 — outer-product stage-2 fast path. At rank=16 with
        // AVX-512 present, the per-token GEMV-then-MultiplyAdd stage 2
        // (~outputDim short Dot calls per token, ~1M total at outputDim=2048
        // / seqLen=512) is replaced by an outer-product kernel that
        // collapses to ~seqLen × outputDim/16 tile FMAs (~3-4× faster on
        // Strix Halo). The kernel consumes a [rank, outputDim] transposed-A
        // buffer; we lazy-build + cache it on the adapter the first time we
        // dispatch a (layer, proj) pair through this path. The cache also
        // covers F16 / BF16 / Q8_0-B adapters — the dequant-and-transpose
        // happens once at first use.
        nint aTransposedHandle = LoraStage2.EnsureATransposedF32(
            adapter as LoraAdapter, layer, projName, in w, adapter.Rank);

        // Phase 4d.5 / Gap 2 — fast-path plumbing: when both base and adapter
        // B are Q8_0 AND the caller pre-quantised x, we can route stage 1
        // through `MatMul.GemmQ8_0(preQuantizedInput=preQuantX)` and skip the
        // activation-quant cost. The original Phase 4d.5 spike gated this
        // behind `DOTLLM_LORA_FORCE_Q8_PREQUANT=1` because kernel-level
        // probing showed the Q8_0 GEMM at M=rank=16 was ~1.7× slower than
        // the dequant-once F32 path. Phase 4d.6 keeps the env-var gate —
        // independent of the stage-2 outer-product fix below — until a
        // tiny-M Q8_0 stage-1 kernel can win at this geometry.
        if (preQuantX is not null
            && preQuantXType == QuantizationType.Q8_0
            && w.WeightDType == LoraWeightDType.Q8_0
            && (inputDim & 31) == 0
            && Environment.GetEnvironmentVariable("DOTLLM_LORA_FORCE_Q8_PREQUANT") == "1")
        {
            LoraDelta.ApplyQ8_0BWithPreQuantX(
                preQuantX, (byte*)w.BHandle, (void*)w.AHandle, y,
                seqLen, inputDim, outputDim, adapter.Rank, scale,
                w.ResolvedAWeightDType, _threadPool, aTransposedHandle);
            return;
        }

        LoraDelta.Apply((float*)x, (void*)w.BHandle, (void*)w.AHandle, (float*)y,
                        seqLen, inputDim, outputDim, adapter.Rank, scale,
                        w.WeightDType, w.ResolvedAWeightDType, aTransposedHandle);
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
    /// Minimum row byte stride for R4 interleaving to be beneficial.
    /// Below this, 4 rows span &lt; 4KB (1 page) and the hardware prefetcher
    /// handles the original stride efficiently. Above this, R4 contiguity
    /// avoids cross-page TLB misses and prefetcher stride-limit failures.
    /// </summary>
    private const int InterleavedMinRowBytes = 1024;

    /// <summary>
    /// GEMV using R4-interleaved repacked weights for improved cache locality.
    /// Falls back to original Gemv when repacked weight is default (Ptr == 0)
    /// or when row stride is too small for interleaving to help.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void GemvInterleaved(nint origWeights, QuantizationType qt, float* x, float* y,
                                 int m, int k, in WeightRepacking.RepackedWeight rw)
    {
        if (rw.Ptr == 0 || rw.RowBytes < InterleavedMinRowBytes)
        {
            Gemv(origWeights, qt, x, y, m, k);
            return;
        }

        // Quantize input for the interleaved ComputeRows variant
        byte* inputQ8Scratch = (byte*)_state.InputQ8Scratch;
        if (qt == QuantizationType.Q8_0)
        {
            int blockCount = k / Q8_0GroupSize;
            int xQ8Bytes = blockCount * Q8_0BlockBytes;
            byte* xQ8 = (byte*)(_threadPool?.GetWorkerScratch(0, xQ8Bytes) ?? (nint)inputQ8Scratch);
            MatMul.QuantizeF32ToQ8_0(x, xQ8, k);
            MatMul.ComputeRowsQ8_0Interleaved((byte*)rw.Ptr, xQ8, y, rw.FullGroupCount, rw.TailRows, blockCount, _threadPool);
        }
        else if (qt == QuantizationType.Q5_0)
        {
            int blockCount = k / Q8_0GroupSize;
            int xQ8Bytes = blockCount * MatMul.Q8_1BlockBytes;
            byte* xQ8 = (byte*)(_threadPool?.GetWorkerScratch(0, xQ8Bytes) ?? (nint)inputQ8Scratch);
            MatMul.QuantizeF32ToQ8_1(x, xQ8, k);
            MatMul.ComputeRowsQ5_0Interleaved((byte*)rw.Ptr, xQ8, y, rw.FullGroupCount, rw.TailRows, blockCount, _threadPool);
        }
        else if (qt is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K)
        {
            int superBlockCount = k / 256;
            int xQ8KBytes = superBlockCount * MatMul.Q8_K_BlockBytes;
            byte* xQ8K = (byte*)(_threadPool?.GetWorkerScratch(0, xQ8KBytes) ?? (nint)inputQ8Scratch);
            MatMul.QuantizeF32ToQ8_K(x, xQ8K, k);
            if (qt == QuantizationType.Q4_K)
                MatMul.ComputeRowsQ4_KInterleaved((byte*)rw.Ptr, xQ8K, y, rw.FullGroupCount, rw.TailRows, superBlockCount, _threadPool);
            else if (qt == QuantizationType.Q5_K)
                MatMul.ComputeRowsQ5_KInterleaved((byte*)rw.Ptr, xQ8K, y, rw.FullGroupCount, rw.TailRows, superBlockCount, _threadPool);
            else
                MatMul.ComputeRowsQ6_KInterleaved((byte*)rw.Ptr, xQ8K, y, rw.FullGroupCount, rw.TailRows, superBlockCount, _threadPool);
        }
        else
        {
            Gemv(origWeights, qt, x, y, m, k);
        }
    }

    /// <summary>
    /// GEMM using R4-interleaved repacked weights. For single-token (n=1) uses interleaved ComputeRows.
    /// Multi-token (n&gt;1) falls back to original Gemm — outer-product microkernels don't win on AVX2
    /// due to RyuJIT register pressure (12 YMM accumulators spill with only 16 registers available).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void GemmInterleaved(nint origWeights, QuantizationType qt, float* b, float* c,
                                 int m, int k, int n, byte* preQuantizedInput,
                                 in WeightRepacking.RepackedWeight rw)
    {
        if (rw.Ptr == 0 || n > 1 || rw.RowBytes < InterleavedMinRowBytes)
        {
            // Multi-token or small row stride: use original tiled GEMM path
            Gemm(origWeights, qt, b, c, m, k, n, preQuantizedInput);
            return;
        }

        // Single-token with pre-quantized input: use interleaved ComputeRows directly
        if (preQuantizedInput != null)
        {
            DispatchInterleavedComputeRows(qt, (byte*)rw.Ptr, preQuantizedInput, c,
                rw.FullGroupCount, rw.TailRows, k);
            return;
        }

        // Single-token without pre-quantized: quantize + interleaved dispatch
        GemvInterleaved(origWeights, qt, b, c, m, k, in rw);
    }

    /// <summary>
    /// Dispatches interleaved ComputeRows for a given quant type with pre-quantized input.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void DispatchInterleavedComputeRows(QuantizationType qt, byte* repackedWeights,
        byte* preQuantInput, float* result, int fullGroups, int tailRows, int k)
    {
        if (qt == QuantizationType.Q8_0)
            MatMul.ComputeRowsQ8_0Interleaved(repackedWeights, preQuantInput, result,
                fullGroups, tailRows, k / 32, _threadPool);
        else if (qt == QuantizationType.Q5_0)
            MatMul.ComputeRowsQ5_0Interleaved(repackedWeights, preQuantInput, result,
                fullGroups, tailRows, k / 32, _threadPool);
        else if (qt == QuantizationType.Q4_K)
            MatMul.ComputeRowsQ4_KInterleaved(repackedWeights, preQuantInput, result,
                fullGroups, tailRows, k / 256, _threadPool);
        else if (qt == QuantizationType.Q5_K)
            MatMul.ComputeRowsQ5_KInterleaved(repackedWeights, preQuantInput, result,
                fullGroups, tailRows, k / 256, _threadPool);
        else if (qt == QuantizationType.Q6_K)
            MatMul.ComputeRowsQ6_KInterleaved(repackedWeights, preQuantInput, result,
                fullGroups, tailRows, k / 256, _threadPool);
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
        _mlaKvState?.Dispose();
        _mlaLatentKvState?.Dispose();
        _weights.Dispose(); // free R4-interleaved weight buffers and any owned bf16→F32 scratch
        // _mmapAnchor is not owned by us — caller disposes the GgufFile / SafetensorsFile.
    }
}
