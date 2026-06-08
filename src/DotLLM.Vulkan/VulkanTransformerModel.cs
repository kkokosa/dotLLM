using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// End-to-end F32 Vulkan forward pass for Llama-family transformer models.
/// Implements <see cref="IModel"/> using only the six wave-1/wave-2 Vulkan
/// compute kernels: <see cref="MatMulF32Kernel"/>, <see cref="RmsNormF32Kernel"/>,
/// <see cref="RopeF32Kernel"/>, <see cref="AttentionF32Kernel"/>,
/// <see cref="SwiGluF32Kernel"/>, plus <see cref="AddKernel"/> for residuals.
/// </summary>
/// <remarks>
/// <para>
/// Scope: F32-only. Quantised weights are dequantised to FP32 at
/// construction time via <see cref="VulkanWeights.Upload"/> and uploaded
/// to device-local (VRAM) memory. The model assumes a pure-Transformer
/// Llama-family architecture — MLA, MoE, and SSM layers are rejected at
/// load time.
/// </para>
/// <para>
/// Forward pass is fence-pipelined: a single persistent command buffer
/// records every kernel dispatch + inter-kernel pipeline barrier for the
/// whole forward, submits once per forward, and waits on a single fence
/// before downloading logits. Legacy synchronous kernel launches (one
/// <c>vkQueueWaitIdle</c> per kernel) are only used by the standalone
/// unit tests.
/// </para>
/// <para>
/// Architectural parallel with <c>DotLLM.Cuda.CudaTransformerModel</c>:
/// upload weights once at construction, reuse a single
/// <see cref="VulkanForwardState"/> for scratch, and drive every linear
/// projection through one <c>matmul_f32</c> call — no prefill / decode
/// split because there is no quantised GEMV kernel yet. Logits come back
/// as a single <see cref="UnmanagedTensor"/> of shape <c>[1, vocabSize]</c>
/// matching the CUDA return convention.
/// </para>
/// </remarks>
public sealed class VulkanTransformerModel : IModel
{
    private readonly VulkanDevice _device;
    private readonly VulkanWeights _weights;
    private readonly VulkanForwardState _state;

    // Kernels — one instance each, pipelines are reused across all launches.
    private readonly MatMulF32Kernel _matmul;
    private readonly RmsNormF32Kernel _rmsnorm;
    private readonly RopeF32Kernel _rope;
    private readonly AttentionF32Kernel _attention;
    private readonly SwiGluF32Kernel _swiglu;
    private readonly AddKernel _add;

    // Persistent command buffer + fence used by Forward. One SubmitContext
    // per model — reset+begin at the start of each forward, submit+wait at
    // the end. Bias host-side steps split the forward into multiple submits
    // but each submit still batches many dispatches behind one fence.
    private readonly VulkanDevice.SubmitContext _submit;

    private readonly TransformerWeights _cpuWeights; // retained for embedding lookup
    private readonly GgufFile? _gguf;
    private readonly float _ropeTheta;
    private readonly int _ropeDim;
    private readonly RopeF32Kernel.Variant _ropeVariant;
    private readonly int _slidingWindow;
    private readonly bool _ownsDevice;

    // LoRA (Phase 4b) — device-side cache of uploaded adapters keyed by
    // ILoraAdapter reference identity. Lazy: zero VRAM when no LoRA Forward
    // is ever invoked. _currentLora is set/cleared in the try/finally
    // surrounding the inner Forward and is checked at every projection
    // site in RecordMatmulWithLora to decide whether to dispatch the
    // LoRA delta on top of the base projection.
    private readonly VulkanLoraAdapterCache _loraCache;
    private VulkanLoraAdapter? _currentLora;

    // Fused LoRA delta-GEMV (single dispatch in place of the four-step
    // matmul(B) → matmul(A) → add → vkCmdCopyBuffer chain). Null when the
    // .spv is missing (older builds); router falls back to the un-fused
    // path. Used only when the adapter's rank ≤ LoraDeltaGemvFusedF32Kernel.MaxRank.
    private readonly LoraDeltaGemvFusedF32Kernel? _loraDeltaGemvFused;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes + _weights.AllocatedBytes;

    /// <summary>Creates a <see cref="VulkanKvCache"/> sized for this model.</summary>
    public VulkanKvCache CreateKvCache(int maxSeqLen)
        => new(_device, Config.NumLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen);

    private VulkanTransformerModel(
        VulkanDevice device, bool ownsDevice,
        ModelConfig config, VulkanWeights weights, TransformerWeights cpuWeights,
        VulkanForwardState state,
        MatMulF32Kernel matmul, RmsNormF32Kernel rmsnorm, RopeF32Kernel rope,
        AttentionF32Kernel attention, SwiGluF32Kernel swiglu, AddKernel add,
        LoraDeltaGemvFusedF32Kernel? loraDeltaGemvFused,
        VulkanDevice.SubmitContext submit,
        GgufFile? gguf,
        float ropeTheta, int ropeDim, RopeF32Kernel.Variant ropeVariant, int slidingWindow)
    {
        _device = device;
        _ownsDevice = ownsDevice;
        Config = config;
        _weights = weights;
        _cpuWeights = cpuWeights;
        _state = state;
        _matmul = matmul;
        _rmsnorm = rmsnorm;
        _rope = rope;
        _attention = attention;
        _swiglu = swiglu;
        _add = add;
        _loraDeltaGemvFused = loraDeltaGemvFused;
        _submit = submit;
        _gguf = gguf;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        _ropeVariant = ropeVariant;
        _slidingWindow = slidingWindow;
        _loraCache = new VulkanLoraAdapterCache(device);
    }

    /// <summary>
    /// Loads a model from an opened GGUF file onto a new Vulkan device.
    /// The caller owns the returned model; disposing it tears down the
    /// device, pipelines, and weight buffers.
    /// </summary>
    /// <param name="gguf">Opened GGUF file. Must remain alive for the model's lifetime.</param>
    /// <param name="config">Model configuration extracted from the GGUF metadata.</param>
    /// <param name="spvDir">
    /// Directory containing the compiled Vulkan SPIR-V blobs. When null,
    /// falls back to <c>spv/</c> next to the running assembly (matches the
    /// MSBuild <c>Content</c> copy pattern used by the Vulkan project).
    /// </param>
    public static VulkanTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);

        RejectUnsupportedArchitecture(config);

        var device = VulkanDevice.Create();
        try
        {
            spvDir ??= Path.Combine(AppContext.BaseDirectory, "spv");
            var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);
            return BuildModel(device, ownsDevice: true, config, cpuWeights, spvDir, gguf);
        }
        catch
        {
            device.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Loads a model onto an existing <see cref="VulkanDevice"/>. The device
    /// is NOT disposed when the model is disposed — the caller retains
    /// ownership. Useful when the device is shared with other Vulkan
    /// components (e.g. a diagnostic hook that wants to launch its own
    /// kernels on the same queue).
    /// </summary>
    public static VulkanTransformerModel LoadFromGguf(
        VulkanDevice device, GgufFile gguf, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);

        RejectUnsupportedArchitecture(config);

        spvDir ??= Path.Combine(AppContext.BaseDirectory, "spv");
        var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);
        return BuildModel(device, ownsDevice: false, config, cpuWeights, spvDir, gguf);
    }

    /// <summary>
    /// Loads a model from a HuggingFace-convention safetensors file onto a
    /// new Vulkan device. Mirrors <see cref="LoadFromGguf(GgufFile, ModelConfig, string?)"/>
    /// but reads weights via <see cref="TransformerWeightsSafetensorsLoader"/>.
    /// </summary>
    public static VulkanTransformerModel LoadFromSafetensors(
        SafetensorsFile file, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(config);

        RejectUnsupportedArchitecture(config);

        var device = VulkanDevice.Create();
        try
        {
            spvDir ??= Path.Combine(AppContext.BaseDirectory, "spv");
            var cpuWeights = TransformerWeightsSafetensorsLoader.Load(file, config);
            return BuildModel(device, ownsDevice: true, config, cpuWeights, spvDir, gguf: null);
        }
        catch
        {
            device.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Loads a model from a safetensors file onto an existing
    /// <see cref="VulkanDevice"/>. The device is NOT disposed when the model
    /// is disposed.
    /// </summary>
    public static VulkanTransformerModel LoadFromSafetensors(
        VulkanDevice device, SafetensorsFile file, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(config);

        RejectUnsupportedArchitecture(config);

        spvDir ??= Path.Combine(AppContext.BaseDirectory, "spv");
        var cpuWeights = TransformerWeightsSafetensorsLoader.Load(file, config);
        return BuildModel(device, ownsDevice: false, config, cpuWeights, spvDir, gguf: null);
    }

    private static VulkanTransformerModel BuildModel(
        VulkanDevice device, bool ownsDevice, ModelConfig config,
        TransformerWeights cpuWeights, string spvDir, GgufFile? gguf)
    {
        var weights = VulkanWeights.Upload(device, cpuWeights, config.NumLayers);

        var state = new VulkanForwardState(device,
            config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
            config.HeadDim, config.IntermediateSize, config.VocabSize,
            initialSeqLen: 1);

        var matmul = MatMulF32Kernel.Create(device, spvDir);
        var rmsnorm = RmsNormF32Kernel.Create(device, spvDir);
        var rope = RopeF32Kernel.Create(device, spvDir);
        var attention = AttentionF32Kernel.Create(device, spvDir);
        var swiglu = SwiGluF32Kernel.Create(device, spvDir);
        var add = AddKernel.Create(device, spvDir);

        // Optional fused LoRA delta-GEMV — TryCreate so older builds without
        // the .spv blob fall back to the un-fused 4-dispatch path. Always
        // attempted (no MoE/MLA gating) because LoRA can target any standard
        // q/k/v/o + gate/up/down projection on the dense path.
        LoraDeltaGemvFusedF32Kernel? loraDeltaGemvFused =
            LoraDeltaGemvFusedF32Kernel.TryCreate(device, spvDir);

        var submit = device.CreateSubmitContext();

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        RoPEType ropeType = config.RoPEConfig?.Type ?? RoPEType.Norm;
        var ropeVariant = ropeType == RoPEType.NeoX ? RopeF32Kernel.Variant.NeoX : RopeF32Kernel.Variant.Norm;

        int slidingWindow = config.SlidingWindowSize ?? 0;

        return new VulkanTransformerModel(
            device, ownsDevice,
            config, weights, cpuWeights, state,
            matmul, rmsnorm, rope, attention, swiglu, add,
            loraDeltaGemvFused,
            submit,
            gguf,
            ropeTheta, ropeDim, ropeVariant, slidingWindow);
    }

    private static void RejectUnsupportedArchitecture(ModelConfig config)
    {
        if (config.MlaConfig is not null)
            throw new NotSupportedException("MLA (DeepSeek-V2/V3) is not supported on the Vulkan backend yet.");

        // MoE / HybridLayout / SsmConfig / Mamba3Config guards live with the
        // chains that introduce those ModelConfig properties — they will be
        // wired into RejectUnsupportedArchitecture by those chains' Vulkan
        // follow-up PRs. F32-dense routing is the only path supported here.
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <summary>
    /// LoRA-aware forward. When <paramref name="adapter"/> is non-null, each
    /// adapted projection (q/k/v/o + gate/up/down on the standard transformer
    /// path) adds <c>scale × (x · B) · A</c> on top of the base projection.
    /// When null, this is byte-equivalent to the 4-arg overload.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Mirrors the CPU <c>TransformerModel.Forward</c> 5-arg overload: a
    /// per-call <see cref="_currentLora"/> field is set/cleared via
    /// try/finally around the inner forward; <see cref="MaybeApplyLoraDelta"/>
    /// at every standard projection site in the inner forward checks the
    /// field and applies the LoRA delta as an extra dispatch chain.
    /// </para>
    /// <para>
    /// MLA-attention (DeepSeek-V2/V3) and MoE-FFN adapter targets are
    /// rejected at validation time — they are deferred follow-ups.
    /// </para>
    /// </remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache, ILoraAdapter? adapter)
    {
        if (adapter is null)
            return Forward(tokenIds, positions, deviceId, kvCache);

        ValidateAdapterForModel(adapter);

        // Resolve / lazy-upload device-side LoRA buffers. Subsequent forwards
        // with the same adapter hit the cache and pay zero upload cost.
        var vkLora = _loraCache.GetOrAdd(adapter);

        // Size LoRA scratch for this adapter's largest output dim. The inner
        // Forward also calls EnsureCapacity, so we run it first ourselves to
        // ensure the LoRA scratch is sized at the current seqLen capacity
        // before any LoRA-active dispatch.
        int seqLen = tokenIds.Length;
        if (seqLen == 0) throw new ArgumentException("tokenIds must be non-empty.", nameof(tokenIds));
        _state.EnsureCapacity(seqLen);
        _state.EnsureLoraScratch(vkLora.Rank, vkLora.MaxOutputDim);

        _currentLora = vkLora;
        try
        {
            return Forward(tokenIds, positions, deviceId, kvCache);
        }
        finally
        {
            _currentLora = null;
        }
    }

    /// <summary>
    /// Validates that <paramref name="adapter"/> is compatible with this
    /// model and that its targeted projections do not collide with
    /// out-of-scope MLA / MoE structures. Mirrors the CPU
    /// <c>TransformerModel.ValidateAdapterForModel</c>.
    /// </summary>
    private void ValidateAdapterForModel(ILoraAdapter adapter)
    {
        if (!adapter.IsCompatible(Config))
            throw new InvalidOperationException(
                $"LoRA adapter '{adapter.Name}' is not compatible with the loaded model "
                + "(layer count, hidden size, or per-projection dimensions mismatch).");

        if (Config.MlaConfig is not null)
        {
            string[] mlaUnsupported = ["q_proj", "k_proj", "v_proj", "o_proj"];
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                foreach (var name in mlaUnsupported)
                {
                    if (adapter.GetLayerWeights(layer, name) is not null)
                        throw new NotSupportedException(
                            $"LoRA adapter '{adapter.Name}' targets MLA-attention projection "
                            + $"'{name}' at layer {layer}. MLA-LoRA support is a follow-up "
                            + "(Phase 4b covers standard q/k/v/o + gate/up/down projections only).");
                }
            }
        }

        // NOTE: MoE LoRA validation is deferred — MoE config is not yet on
        // this base. When MoE lands on the Vulkan backend, mirror the MLA
        // guard above to reject gate/up/down adapter targets on MoE layers
        // until a MoE-LoRA follow-up wires per-expert delta dispatch.
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
    {
        if (tokenIds.Length != positions.Length)
            throw new ArgumentException("tokenIds and positions must have the same length.");

        int seqLen = tokenIds.Length;
        if (seqLen == 0) throw new ArgumentException("tokenIds must be non-empty.", nameof(tokenIds));

        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int intermediateSize = Config.IntermediateSize;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;

        bool scratchResized = _state.EnsureCapacity(seqLen);

        // Descriptor sets cache buffer handles. When scratch is re-allocated
        // every cached set becomes stale and must be dropped — otherwise the
        // next dispatch binds a dangling VkBuffer. In steady-state decode
        // (seqLen = 1 after the initial prefill) scratch never grows, so the
        // cache stays warm across forwards.
        if (scratchResized)
            InvalidateKernelCaches();

        // 1. Host-side upload of per-token embedding rows + positions. Both
        //    land in host-visible host-coherent buffers; a HOST→COMPUTE
        //    barrier at the start of the recorded command buffer makes the
        //    writes visible to the first compute kernel without an explicit
        //    vkQueueWaitIdle.
        UploadEmbeddings(tokenIds);
        UploadPositions(positions);

        // 2. Begin the single per-forward command buffer and record the
        //    whole transformer. Bias-add host steps split the forward into
        //    multiple submits (one per distinct set of biases we need to
        //    pause for); everything else stays inside the pipelined path.
        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);

        for (int layer = 0; layer < Config.NumLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];
            ref readonly var cpuLw = ref _cpuWeights.Layers[layer];

            // Residual snapshot (pre-attention): HiddenState → Residual.
            RecordCopyBuffer(cmdBuf, _state.HiddenState, _state.Residual, (long)seqLen * hiddenSize * sizeof(float));
            KernelSupport.ComputeToComputeBarrier(cmdBuf); // TRANSFER→COMPUTE would be tighter; COMPUTE→COMPUTE covers both paths

            // Attn RMSNorm
            _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.AttnNormWeight, _state.NormOutput,
                rowCount: seqLen, n: hiddenSize, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Q/K/V projections
            _matmul.Record(cmdBuf, lw.Q, _state.NormOutput, _state.Q, lw.QOutputDim, lw.QInputDim, seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            _matmul.Record(cmdBuf, lw.K, _state.NormOutput, _state.K, lw.KOutputDim, lw.KInputDim, seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            _matmul.Record(cmdBuf, lw.V, _state.NormOutput, _state.V, lw.VOutputDim, lw.VInputDim, seqLen);

            // Optional QKV biases — host path. Submit, wait, write, re-begin.
            if (cpuLw.QBias is not null || cpuLw.KBias is not null || cpuLw.VBias is not null)
            {
                KernelSupport.ComputeToHostBarrier(cmdBuf);
                _submit.SubmitAndWait();
                if (cpuLw.QBias is { } qb) AddBiasRows(_state.Q, qb, lw.QOutputDim, seqLen);
                if (cpuLw.KBias is { } kb) AddBiasRows(_state.K, kb, lw.KOutputDim, seqLen);
                if (cpuLw.VBias is { } vb) AddBiasRows(_state.V, vb, lw.VOutputDim, seqLen);
                _submit.Begin();
                cmdBuf = _submit.CommandBuffer;
                KernelSupport.HostToComputeBarrier(cmdBuf);
            }
            else
            {
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // LoRA delta (q/k/v) — applied AFTER bias and BEFORE RoPE so the
            // delta contributes to the same downstream pipeline as the base
            // projection. The matmul input (NormOutput) is still live here.
            if (_currentLora is not null)
            {
                MaybeApplyLoraDelta(cmdBuf, layer, "q_proj", _state.NormOutput, _state.Q,
                    seqLen, lw.QInputDim, lw.QOutputDim);
                MaybeApplyLoraDelta(cmdBuf, layer, "k_proj", _state.NormOutput, _state.K,
                    seqLen, lw.KInputDim, lw.KOutputDim);
                MaybeApplyLoraDelta(cmdBuf, layer, "v_proj", _state.NormOutput, _state.V,
                    seqLen, lw.VInputDim, lw.VOutputDim);
            }

            // RoPE on Q and K
            _rope.Record(cmdBuf, _state.Q, _state.K, _state.PositionsBuffer,
                seqLen: seqLen, numHeads: numHeads, numKvHeads: numKvHeads,
                headDim: headDim, ropeDim: _ropeDim, theta: _ropeTheta,
                variant: _ropeVariant);

            // Attention input buffers: either the uncached K/V window or the full KV cache.
            VulkanDevice.Buffer kSrc, vSrc;
            int seqKv;
            int positionOffset;
            if (kvCache is VulkanKvCache vkCache)
            {
                // RoPE writes K; attention (via the cache buffers) reads K.
                // Barrier the RoPE → KV copy, then the KV copy → attention.
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                vkCache.RecordUpdate(cmdBuf, _state.K, _state.V, positions, seqLen, layer);
                KernelSupport.TransferToComputeBarrier(cmdBuf);
                kSrc = vkCache.GetKeysBuffer(layer);
                vSrc = vkCache.GetValuesBuffer(layer);
                seqKv = vkCache.CurrentLength;
                positionOffset = positions[0];
            }
            else
            {
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                kSrc = _state.K;
                vSrc = _state.V;
                seqKv = seqLen;
                positionOffset = 0;
            }

            _attention.Record(cmdBuf, _state.Q, kSrc, vSrc, _state.AttnOutput,
                seqQ: seqLen, seqKv: seqKv,
                numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
                positionOffset: positionOffset, slidingWindow: _slidingWindow);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Output projection → NormOutput (reuse slot).
            _matmul.Record(cmdBuf, lw.O, _state.AttnOutput, _state.NormOutput,
                lw.OOutputDim, lw.OInputDim, seqLen);

            if (cpuLw.OBias is { } ob)
            {
                KernelSupport.ComputeToHostBarrier(cmdBuf);
                _submit.SubmitAndWait();
                AddBiasRows(_state.NormOutput, ob, lw.OOutputDim, seqLen);
                _submit.Begin();
                cmdBuf = _submit.CommandBuffer;
                KernelSupport.HostToComputeBarrier(cmdBuf);
            }
            else
            {
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // LoRA delta (o_proj): y += scale * (attnOut · B) · A. Applied after
            // bias and before the residual add so the delta participates in the
            // residual stream.
            if (_currentLora is not null)
            {
                MaybeApplyLoraDelta(cmdBuf, layer, "o_proj", _state.AttnOutput, _state.NormOutput,
                    seqLen, lw.OInputDim, lw.OOutputDim);
            }

            // Residual add #1: AddScratch = Residual + NormOutput; then AddScratch → HiddenState.
            _add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch, seqLen * hiddenSize);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            RecordCopyBuffer(cmdBuf, _state.AddScratch, _state.HiddenState, (long)seqLen * hiddenSize * sizeof(float));
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Residual snapshot (pre-FFN): HiddenState → Residual.
            RecordCopyBuffer(cmdBuf, _state.HiddenState, _state.Residual, (long)seqLen * hiddenSize * sizeof(float));
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // FFN RMSNorm
            _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.FfnNormWeight, _state.NormOutput,
                rowCount: seqLen, n: hiddenSize, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Gate/Up projections
            _matmul.Record(cmdBuf, lw.Gate, _state.NormOutput, _state.FfnGate,
                lw.GateOutputDim, lw.GateInputDim, seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            _matmul.Record(cmdBuf, lw.Up, _state.NormOutput, _state.FfnUp,
                lw.UpOutputDim, lw.UpInputDim, seqLen);

            if (cpuLw.GateBias is not null || cpuLw.UpBias is not null)
            {
                KernelSupport.ComputeToHostBarrier(cmdBuf);
                _submit.SubmitAndWait();
                if (cpuLw.GateBias is { } gb) AddBiasRows(_state.FfnGate, gb, lw.GateOutputDim, seqLen);
                if (cpuLw.UpBias is { } ub) AddBiasRows(_state.FfnUp, ub, lw.UpOutputDim, seqLen);
                _submit.Begin();
                cmdBuf = _submit.CommandBuffer;
                KernelSupport.HostToComputeBarrier(cmdBuf);
            }
            else
            {
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // LoRA delta (gate/up): y += scale * (normOut · B) · A. Applied
            // after bias and before SwiGLU so the delta is fused into the
            // nonlinearity input.
            if (_currentLora is not null)
            {
                MaybeApplyLoraDelta(cmdBuf, layer, "gate_proj", _state.NormOutput, _state.FfnGate,
                    seqLen, lw.GateInputDim, lw.GateOutputDim);
                MaybeApplyLoraDelta(cmdBuf, layer, "up_proj", _state.NormOutput, _state.FfnUp,
                    seqLen, lw.UpInputDim, lw.UpOutputDim);
            }

            // SwiGLU
            _swiglu.Record(cmdBuf, _state.FfnGate, _state.FfnUp, _state.SiluOutput, seqLen * intermediateSize);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Down projection
            _matmul.Record(cmdBuf, lw.Down, _state.SiluOutput, _state.NormOutput,
                lw.DownOutputDim, lw.DownInputDim, seqLen);

            if (cpuLw.DownBias is { } db)
            {
                KernelSupport.ComputeToHostBarrier(cmdBuf);
                _submit.SubmitAndWait();
                AddBiasRows(_state.NormOutput, db, lw.DownOutputDim, seqLen);
                _submit.Begin();
                cmdBuf = _submit.CommandBuffer;
                KernelSupport.HostToComputeBarrier(cmdBuf);
            }
            else
            {
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // LoRA delta (down_proj): y += scale * (siluOut · B) · A.
            // Input is post-SwiGLU (siluOut), not normOut. The base GEMM
            // already wrote into normOut, so we accumulate delta in place.
            if (_currentLora is not null)
            {
                MaybeApplyLoraDelta(cmdBuf, layer, "down_proj", _state.SiluOutput, _state.NormOutput,
                    seqLen, lw.DownInputDim, lw.DownOutputDim);
            }

            // Residual add #2: AddScratch = Residual + NormOutput; then AddScratch → HiddenState.
            _add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch, seqLen * hiddenSize);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            RecordCopyBuffer(cmdBuf, _state.AddScratch, _state.HiddenState, (long)seqLen * hiddenSize * sizeof(float));

            // COMPUTE→COMPUTE between layers — next iteration's first op is the HiddenState→Residual copy.
            if (layer < Config.NumLayers - 1)
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }

        // 3. Final RMSNorm on the last token only, then LM head.
        long rowBytes = (long)hiddenSize * sizeof(float);
        long lastRowOffset = (long)(seqLen - 1) * rowBytes;
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.NormOutput,
            srcOffset: (ulong)lastRowOffset, dstOffset: 0, size: (ulong)rowBytes);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _rmsnorm.Record(cmdBuf, _state.NormOutput, _weights.OutputNormWeight, _state.NormOutput,
            rowCount: 1, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _matmul.Record(cmdBuf, _weights.OutputWeight, _state.NormOutput, _state.Logits,
            _weights.OutputOutputDim, _weights.OutputInputDim, 1);

        // 4. COMPUTE→HOST barrier for the vocab-row download that follows, submit, wait.
        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();

        // 5. Return logits as a host-resident UnmanagedTensor [1, vocabSize].
        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        unsafe
        {
            var dest = new Span<float>((void*)result.DataPointer, vocabSize);
            _device.Download(_state.Logits, dest);
        }
        return result;
    }

    private void InvalidateKernelCaches()
    {
        _matmul.InvalidateDescriptorCache();
        _rmsnorm.InvalidateDescriptorCache();
        _rope.InvalidateDescriptorCache();
        _attention.InvalidateDescriptorCache();
        _swiglu.InvalidateDescriptorCache();
        _add.InvalidateDescriptorCache();
        _loraDeltaGemvFused?.InvalidateDescriptorCache();
    }

    /// <summary>
    /// Records a device-to-device <c>vkCmdCopyBuffer</c> of
    /// <paramref name="byteCount"/> bytes from the start of <paramref name="src"/>
    /// to the start of <paramref name="dst"/>. Replaces the scaffold's
    /// host-mapped memcpy which required a submit boundary on every call.
    /// </summary>
    private static void RecordCopyBuffer(nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, long byteCount)
        => RecordCopyBufferRange(cmdBuf, src, dst, srcOffset: 0, dstOffset: 0, size: (ulong)byteCount);

    /// <summary>
    /// Dispatches the LoRA delta for <paramref name="projName"/> at
    /// <paramref name="layer"/> when an adapter is active and targets that
    /// site. No-op when there is no active adapter or no entry.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Fast path (rank ≤ <see cref="LoraDeltaGemvFusedF32Kernel.MaxRank"/>
    /// and the fused .spv blob is present): a single dispatch of
    /// <see cref="LoraDeltaGemvFusedF32Kernel"/> performs
    /// <c>y[t, m] += sum_r A[m, r] · dot(B[r, :], x[t, :])</c> in place.
    /// One workgroup per token row keeps the rank-sized inner reduction in
    /// shared memory and reuses it across the full output dim.
    /// </para>
    /// <para>
    /// Fallback path (rank &gt; 32 or older builds without the fused .spv):
    /// the original 4-dispatch chain
    /// <list type="number">
    ///   <item><c>tmp[seqLen, rank] = matmul_f32(B_scaled, x)</c> via <see cref="MatMulF32Kernel"/>.</item>
    ///   <item><c>delta[seqLen, outputDim] = matmul_f32(A, tmp)</c> via <see cref="MatMulF32Kernel"/>.</item>
    ///   <item><c>deltaSum[seqLen, outputDim] = AddKernel(y, delta)</c> via <see cref="AddKernel"/>.</item>
    ///   <item><c>vkCmdCopyBuffer(deltaSum -> y)</c>.</item>
    /// </list>
    /// </para>
    /// <para>
    /// The <c>scale = alpha / rank</c> factor is folded into <c>B</c> at
    /// upload time (see <see cref="VulkanLoraAdapter.Upload"/>), so neither
    /// path needs a separate scale parameter.
    /// </para>
    /// </remarks>
    private void MaybeApplyLoraDelta(
        nint cmdBuf, int layer, string projName,
        VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int seqLen, int inputDim, int outputDim)
    {
        var lora = _currentLora;
        if (lora is null) return;
        var lb = lora.Get(layer, projName);
        if (lb is not { } w) return;

        if (w.InputDim != inputDim || w.OutputDim != outputDim)
            throw new InvalidOperationException(
                $"LoRA adapter '{lora.Source.Name}' layer={layer} proj='{projName}' shape "
                + $"({w.InputDim}x{w.OutputDim}) does not match base projection ({inputDim}x{outputDim}).");

        var tmp = _state.LoraTmp ?? throw new InvalidOperationException(
            "LoraTmp scratch is null — EnsureLoraScratch was not called before a LoRA-active Forward.");

        // Fused fast path: two dispatches (B-reduce + A-accumulate-in-place)
        // in place of the original four. Gated by SPV availability + rank cap.
        if (_loraDeltaGemvFused is not null && w.Rank <= LoraDeltaGemvFusedF32Kernel.MaxRank
            && Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_FUSED_LORA_DELTA") != "1")
        {
            _loraDeltaGemvFused.Record(cmdBuf, x, w.B, w.A, y, tmp,
                seqLen: seqLen, inputDim: inputDim, outputDim: outputDim, rank: w.Rank);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            return;
        }

        var delta = _state.LoraDelta ?? throw new InvalidOperationException("LoraDelta scratch is null.");
        var deltaSum = _state.LoraDeltaSum ?? throw new InvalidOperationException("LoraDeltaSum scratch is null.");

        _matmul.Record(cmdBuf, w.B, x, tmp, m: w.Rank, k: inputDim, n: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _matmul.Record(cmdBuf, w.A, tmp, delta, m: outputDim, k: w.Rank, n: seqLen);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _add.Record(cmdBuf, y, delta, deltaSum, seqLen * outputDim);
        // COMPUTE→TRANSFER would be tighter; COMPUTE→COMPUTE covers it and matches
        // the convention used by the other RecordCopyBuffer sites in this file.
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        var region = new VkBufferCopy
        {
            srcOffset = 0,
            dstOffset = 0,
            size = (ulong)((long)seqLen * outputDim * sizeof(float)),
        };
        VulkanApi.vkCmdCopyBuffer(cmdBuf, deltaSum.Handle, y.Handle, 1, region);
        KernelSupport.TransferToComputeBarrier(cmdBuf);
    }

    private static void RecordCopyBufferRange(
        nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst,
        ulong srcOffset, ulong dstOffset, ulong size)
    {
        var region = new VkBufferCopy { srcOffset = srcOffset, dstOffset = dstOffset, size = size };
        VulkanApi.vkCmdCopyBuffer(cmdBuf, src.Handle, dst.Handle, 1, region);
    }

    /// <summary>
    /// Adds a per-feature bias vector to every row of a
    /// <c>[seqLen, outputDim]</c> FP32 output buffer. Implemented in-place on
    /// the host via mapped memory — biases are tiny (hidden_size scale), and
    /// adding a dedicated "bias_add" compute kernel is out of scope for the
    /// correctness wave.
    /// </summary>
    private unsafe void AddBiasRows(VulkanDevice.Buffer output, float[] bias, int outputDim, int seqLen)
    {
        long biasBytes = (long)outputDim * sizeof(float);
        long outBytes = biasBytes * seqLen;

        VulkanApi.vkMapMemory(_device.Handle, output.Memory, 0, (ulong)outBytes, 0, out nint outMapped)
            .ThrowOnError("vkMapMemory AddBiasRows.output");
        try
        {
            float* o = (float*)outMapped;
            fixed (float* b = bias)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    for (int i = 0; i < outputDim; i++)
                        o[t * outputDim + i] += b[i];
                }
            }
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device.Handle, output.Memory);
        }
    }

    /// <summary>
    /// Resolves each token ID into its FP32 embedding row and packs the
    /// result into <see cref="VulkanForwardState.HiddenState"/>. Does a
    /// row-by-row dequant when the table was Q8_0 / F16 / other (GGUF often
    /// quantises the embedding table alongside the weights).
    /// </summary>
    private unsafe void UploadEmbeddings(ReadOnlySpan<int> tokenIds)
    {
        int hiddenSize = Config.HiddenSize;
        int vocab = Config.VocabSize;
        int seqLen = tokenIds.Length;
        var qt = _cpuWeights.TokenEmbedQuantType;

        long rowBytes = (long)hiddenSize * sizeof(float);
        VulkanApi.vkMapMemory(_device.Handle, _state.HiddenState.Memory, 0, (ulong)(seqLen * rowBytes), 0, out nint mapped)
            .ThrowOnError("vkMapMemory UploadEmbeddings");
        try
        {
            float* dst = (float*)mapped;

            if (qt == QuantizationType.F32)
            {
                // Direct memcpy from mmap.
                float* src = (float*)_cpuWeights.TokenEmbedWeight;
                for (int t = 0; t < seqLen; t++)
                {
                    int id = tokenIds[t];
                    if ((uint)id >= (uint)vocab)
                        throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {id} is out of range");
                    new ReadOnlySpan<float>(src + (long)id * hiddenSize, hiddenSize)
                        .CopyTo(new Span<float>(dst + (long)t * hiddenSize, hiddenSize));
                }
            }
            else
            {
                // Dequantize one row per token into mapped hidden-state region.
                long tableRowBytes = Dequantize.RowByteSize(hiddenSize, qt);
                for (int t = 0; t < seqLen; t++)
                {
                    int id = tokenIds[t];
                    if ((uint)id >= (uint)vocab)
                        throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {id} is out of range");
                    nint rowPtr = _cpuWeights.TokenEmbedWeight + (nint)(id * tableRowBytes);
                    Dequantize.ToFloat32(rowPtr, hiddenSize, qt,
                        new Span<float>(dst + (long)t * hiddenSize, hiddenSize));
                }
            }
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device.Handle, _state.HiddenState.Memory);
        }
    }

    private unsafe void UploadPositions(ReadOnlySpan<int> positions)
    {
        // The Allocate in EnsureCapacity already sized PositionsBuffer for seqLen;
        // delegate the mapped copy to device.Upload via a raw byte span.
        var posBytes = MemoryMarshal.AsBytes(positions);
        _device.Upload(posBytes, _state.PositionsBuffer);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _submit.Dispose();

        // Drop the device-side LoRA cache before tearing down the device —
        // each VulkanLoraAdapter owns VkBuffers that must be freed before
        // the device is disposed.
        _loraCache.Dispose();

        _state.Dispose();
        _weights.Dispose();

        _loraDeltaGemvFused?.Dispose();
        _add.Dispose();
        _swiglu.Dispose();
        _attention.Dispose();
        _rope.Dispose();
        _rmsnorm.Dispose();
        _matmul.Dispose();

        _cpuWeights.Dispose();
        if (_ownsDevice)
            _device.Dispose();
    }
}
