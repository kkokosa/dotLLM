namespace DotLLM.Vulkan;

/// <summary>
/// Owns all per-forward-pass scratch buffers on the Vulkan device.
/// Sized for the maximum <c>seqLen</c> the caller has used so far; grows monotonically.
/// Mirrors <c>DotLLM.Cuda.CudaForwardState</c> but for FP32 storage on a Vulkan device.
/// </summary>
/// <remarks>
/// All buffers are host-visible host-coherent. The Vulkan scaffold does not
/// have a staging path yet; <see cref="VulkanDevice.Allocate"/> returns memory
/// that is mappable from both host and GPU, which is slower than device-local
/// memory for real kernels but keeps weight upload / result download trivial.
/// Optimising this is explicitly out of scope for the end-to-end wave.
/// </remarks>
internal sealed class VulkanForwardState : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly int _vocabSize;
    private int _capacitySeqLen;

    // ── Transformer layer scratch (all FP32) ──────────────────────────
    public VulkanDevice.Buffer HiddenState { get; private set; } = null!;
    public VulkanDevice.Buffer Residual { get; private set; } = null!;
    public VulkanDevice.Buffer NormOutput { get; private set; } = null!;
    public VulkanDevice.Buffer AddScratch { get; private set; } = null!;
    public VulkanDevice.Buffer Q { get; private set; } = null!;
    public VulkanDevice.Buffer K { get; private set; } = null!;
    public VulkanDevice.Buffer V { get; private set; } = null!;
    public VulkanDevice.Buffer AttnOutput { get; private set; } = null!;
    public VulkanDevice.Buffer FfnGate { get; private set; } = null!;
    public VulkanDevice.Buffer FfnUp { get; private set; } = null!;
    public VulkanDevice.Buffer SiluOutput { get; private set; } = null!;

    // ── Logits (last token only) ──────────────────────────────────────
    public VulkanDevice.Buffer Logits { get; private set; }

    // ── Host → device transfer scratch (tokens + positions) ──────────
    public VulkanDevice.Buffer PositionsBuffer { get; private set; }

    // ── LoRA delta scratch (Phase 4b) ─────────────────────────────────
    // Allocated lazily on first LoRA-aware forward via EnsureLoraScratch
    // (otherwise null — forward pass with no adapter pays no extra VRAM).
    // Sized for [seqLen, max(rank)] / [seqLen, max(outputDim)] so we can
    // dispatch the two-stage LoRA delta as:
    //   LoraTmp[seqLen, rank] = matmul_f32(B_scaled, x)
    //   LoraDelta[seqLen, outputDim] = matmul_f32(A, LoraTmp)
    //   LoraDeltaSum[seqLen, outputDim] = AddKernel(y, LoraDelta)
    //   vkCmdCopyBuffer(LoraDeltaSum -> y)
    // The third buffer is needed because AddKernel writes to a separate
    // output (read-only A, write-only C); we copy the sum back into y.
    private int _loraCapacityRank;
    private int _loraCapacityOutputDim;
    public VulkanDevice.Buffer? LoraTmp { get; private set; }       // [seqLen, rank]
    public VulkanDevice.Buffer? LoraDelta { get; private set; }     // [seqLen, outputDim]
    public VulkanDevice.Buffer? LoraDeltaSum { get; private set; }  // [seqLen, outputDim]

    private bool _disposed;

    public long AllocatedBytes { get; private set; }

    public VulkanForwardState(
        VulkanDevice device,
        int hiddenSize, int numHeads, int numKvHeads, int headDim,
        int intermediateSize, int vocabSize, int initialSeqLen)
    {
        _device = device;
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _intermediateSize = intermediateSize;
        _vocabSize = vocabSize;

        // LM-head logits are always one token (last). Positions buffer sized for some reasonable
        // default; grows with EnsureCapacity.
        Logits = device.Allocate((long)vocabSize * sizeof(float));
        PositionsBuffer = device.Allocate(Math.Max(1, initialSeqLen) * sizeof(int));

        AllocateForCapacity(Math.Max(1, initialSeqLen));
    }

    /// <summary>
    /// Ensures all scratch buffers are large enough to host <paramref name="seqLen"/> tokens.
    /// Grows monotonically; never shrinks.
    /// </summary>
    /// <returns><c>true</c> when scratch was re-allocated (so every cached VkBuffer handle
    /// is now stale); <c>false</c> when existing capacity was already sufficient.</returns>
    public bool EnsureCapacity(int seqLen)
    {
        if (seqLen <= _capacitySeqLen) return false;

        ReleaseLayerScratch();
        AllocateForCapacity(seqLen);
        return true;
    }

    /// <summary>
    /// Ensures the LoRA scratch buffers are sized for at least
    /// <paramref name="rank"/> × <paramref name="outputDim"/> at the current
    /// <see cref="EnsureCapacity"/>-honoured seqLen capacity. Allocated
    /// lazily (so non-LoRA forwards never pay this VRAM cost) and grows
    /// monotonically — multiple adapters with different ranks share one
    /// scratch sized to the largest seen so far.
    /// </summary>
    /// <returns>
    /// <c>true</c> when scratch was re-allocated (so cached descriptor sets
    /// pointing at the old <see cref="LoraTmp"/> / <see cref="LoraDelta"/> /
    /// <see cref="LoraDeltaSum"/> handles are now stale); <c>false</c> when
    /// existing capacity was sufficient.
    /// </returns>
    public bool EnsureLoraScratch(int rank, int outputDim)
    {
        if (rank <= 0) throw new ArgumentOutOfRangeException(nameof(rank));
        if (outputDim <= 0) throw new ArgumentOutOfRangeException(nameof(outputDim));

        bool needRealloc =
            LoraTmp is null || LoraDelta is null || LoraDeltaSum is null
            || rank > _loraCapacityRank
            || outputDim > _loraCapacityOutputDim;
        if (!needRealloc) return false;

        // Grow to the max ever requested (monotonic — small adapters
        // benefit from a previous larger allocation; large adapters force
        // a one-shot resize).
        int newRank = Math.Max(_loraCapacityRank, rank);
        int newOutputDim = Math.Max(_loraCapacityOutputDim, outputDim);

        LoraTmp?.Dispose();
        LoraDelta?.Dispose();
        LoraDeltaSum?.Dispose();

        long tmpBytes = (long)_capacitySeqLen * newRank * sizeof(float);
        long deltaBytes = (long)_capacitySeqLen * newOutputDim * sizeof(float);

        // Host-visible host-coherent (matches every other scratch buffer on
        // this scaffold path — see VulkanDevice.Allocate). A real Vulkan
        // perf pass would migrate these to device-local + staging, but the
        // scaffold has not yet introduced AllocateDeviceLocal.
        LoraTmp = _device.Allocate(tmpBytes);
        LoraDelta = _device.Allocate(deltaBytes);
        LoraDeltaSum = _device.Allocate(deltaBytes);
        _loraCapacityRank = newRank;
        _loraCapacityOutputDim = newOutputDim;
        return true;
    }

    private void AllocateForCapacity(int seqLen)
    {
        long hiddenBytes = (long)seqLen * _hiddenSize * sizeof(float);
        long qBytes = (long)seqLen * _numHeads * _headDim * sizeof(float);
        long kvBytes = (long)seqLen * _numKvHeads * _headDim * sizeof(float);
        long ffnBytes = (long)seqLen * _intermediateSize * sizeof(float);

        // Buffers never read or written from the host go device-local — the
        // driver can pick its tiled / swizzled native layout instead of the
        // host-coherent linear layout that Allocate() returns. On UMA (Strix
        // Halo iGPU) this lets the compute path use the GPU's preferred
        // memory access pattern; on dGPU it keeps the bytes off the
        // PCIe-host-coherent path.
        //
        // Bias-add receiver buffers (Q, K, V, NormOutput, FfnGate, FfnUp)
        // stay host-visible because AddBiasRows host-maps them when a bias
        // tensor is present. SmolLM-135M has no biases so the host-map
        // never fires, but Phi-3 / Qwen3 / DeepSeek-V2 do — moving them
        // requires a bias_add_f32 compute kernel (issue #7).
        HiddenState = _device.AllocateDeviceLocal(hiddenBytes);
        Residual = _device.AllocateDeviceLocal(hiddenBytes);
        NormOutput = _device.Allocate(hiddenBytes);
        AddScratch = _device.Allocate(hiddenBytes);

        Q = _device.Allocate(qBytes);
        K = _device.Allocate(kvBytes);
        V = _device.Allocate(kvBytes);
        AttnOutput = _device.AllocateDeviceLocal(qBytes);

        FfnGate = _device.Allocate(ffnBytes);
        FfnUp = _device.Allocate(ffnBytes);
        SiluOutput = _device.AllocateDeviceLocal(ffnBytes);

        // Resize positions buffer — host writes positions per forward.
        PositionsBuffer.Dispose();
        PositionsBuffer = _device.Allocate((long)seqLen * sizeof(int));

        _capacitySeqLen = seqLen;

        AllocatedBytes = hiddenBytes * 4 + qBytes * 2 + kvBytes * 2 + ffnBytes * 3
                       + (long)_vocabSize * sizeof(float) + (long)seqLen * sizeof(int);
    }

    private void ReleaseLayerScratch()
    {
        HiddenState?.Dispose();
        Residual?.Dispose();
        NormOutput?.Dispose();
        AddScratch?.Dispose();
        Q?.Dispose();
        K?.Dispose();
        V?.Dispose();
        AttnOutput?.Dispose();
        FfnGate?.Dispose();
        FfnUp?.Dispose();
        SiluOutput?.Dispose();

        // LoRA scratch (Phase 4b) — sized in seqLen × rank / outputDim, so
        // it grows alongside the main scratch on EnsureCapacity. Reset the
        // capacity counters so the next EnsureLoraScratch call re-allocates
        // at the new seqLen.
        LoraTmp?.Dispose(); LoraTmp = null;
        LoraDelta?.Dispose(); LoraDelta = null;
        LoraDeltaSum?.Dispose(); LoraDeltaSum = null;
        _loraCapacityRank = 0;
        _loraCapacityOutputDim = 0;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        ReleaseLayerScratch();
        Logits?.Dispose();
        PositionsBuffer?.Dispose();
    }
}
