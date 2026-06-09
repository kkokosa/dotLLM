using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Two-dispatch fused LoRA delta:
/// <list type="number">
///   <item><c>tmp[t, r] = dot(B[r, :], x[t, :])</c> (cooperative WG-wide reduction).</item>
///   <item><c>y[t, m] += sum_r A[m, r] * tmp[t, r]</c> (in-place accumulate).</item>
/// </list>
/// Replaces the un-fused 4-dispatch chain
/// (matmul B → matmul A → add → vkCmdCopyBuffer) used by
/// <c>VulkanTransformerModel.MaybeApplyLoraDelta</c>: same math, half the
/// dispatches, no scratch round-trip (the A-stage writes <c>y</c> in place).
/// </summary>
/// <remarks>
/// <para>
/// A "single-shader fused" variant was tried first (one workgroup per token,
/// per-thread B reduction) but the per-tile recomputation of B made it 1.4–2.7×
/// slower than the un-fused chain at rank 16/32 on Strix Halo. The two-shader
/// split keeps the B reduction global (one workgroup per <c>(t, r)</c>),
/// matching the un-fused step's compute exactly while still saving 2 dispatches
/// per delta site.
/// </para>
/// <para>
/// Bounded to <see cref="MaxRank"/> = 32 — covers the common PEFT defaults
/// (4 / 8 / 16) and every TinyLlama / Llama-3 adapter checked in. Callers
/// route ranks &gt; 32 through the un-fused path.
/// </para>
/// <para>
/// <c>B</c> is expected to be pre-scaled by <c>alpha / rank</c> at upload
/// time (see <see cref="VulkanLoraAdapter.Upload"/>); the A-stage shader is
/// scale-agnostic.
/// </para>
/// </remarks>
public sealed class LoraDeltaGemvFusedF32Kernel : IDisposable
{
    /// <summary>Maximum LoRA rank the fused shader supports.</summary>
    public const int MaxRank = 32;

    private const int WorkgroupTile = 64;
    private const int BPushConstantBytes = 3 * sizeof(uint); // inputDim, rank, seqLen
    private const int APushConstantBytes = 3 * sizeof(uint); // outputDim, rank, seqLen

    private readonly VulkanDevice _device;
    private readonly VulkanModule _moduleB;
    private readonly VulkanModule _moduleA;
    private readonly ComputePipeline _pipelineB;
    private readonly ComputePipeline _pipelineA;
    private readonly nint _descriptorPoolB;
    private readonly nint _descriptorPoolA;
    private readonly DescriptorSetCache _descriptorCacheB;
    private readonly DescriptorSetCache _descriptorCacheA;
    private bool _disposed;

    private LoraDeltaGemvFusedF32Kernel(
        VulkanDevice device,
        VulkanModule moduleB, ComputePipeline pipelineB, nint poolB,
        VulkanModule moduleA, ComputePipeline pipelineA, nint poolA)
    {
        _device = device;
        _moduleB = moduleB;
        _pipelineB = pipelineB;
        _descriptorPoolB = poolB;
        _descriptorCacheB = new DescriptorSetCache(device, poolB, pipelineB.DescriptorSetLayout, buffersPerSet: 3);
        _moduleA = moduleA;
        _pipelineA = pipelineA;
        _descriptorPoolA = poolA;
        _descriptorCacheA = new DescriptorSetCache(device, poolA, pipelineA.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>
    /// Loads <c>lora_delta_b_reduce_f32.spv</c> + <c>lora_delta_gemv_fused_f32.spv</c>
    /// from <paramref name="spvDir"/> and creates both pipelines.
    /// </summary>
    public static LoraDeltaGemvFusedF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string pathB = Path.Combine(spvDir, "lora_delta_b_reduce_f32.spv");
        string pathA = Path.Combine(spvDir, "lora_delta_gemv_fused_f32.spv");
        if (!File.Exists(pathB))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {pathB}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");
        if (!File.Exists(pathA))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {pathA}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule moduleB = VulkanModule.LoadFromFile(device, pathB);
        ComputePipeline pipelineB;
        nint poolB = 0;
        VulkanModule? moduleA = null;
        ComputePipeline? pipelineA = null;
        nint poolA = 0;
        try
        {
            Span<VkDescriptorBinding> bindingsB = stackalloc VkDescriptorBinding[3];
            bindingsB[0] = new VkDescriptorBinding(0);
            bindingsB[1] = new VkDescriptorBinding(1);
            bindingsB[2] = new VkDescriptorBinding(2);
            pipelineB = moduleB.CreateComputePipeline(
                entryPoint: "main",
                bindings: bindingsB,
                pushConstantBytes: BPushConstantBytes);
            poolB = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);

            moduleA = VulkanModule.LoadFromFile(device, pathA);
            Span<VkDescriptorBinding> bindingsA = stackalloc VkDescriptorBinding[3];
            bindingsA[0] = new VkDescriptorBinding(0);
            bindingsA[1] = new VkDescriptorBinding(1);
            bindingsA[2] = new VkDescriptorBinding(2);
            pipelineA = moduleA.CreateComputePipeline(
                entryPoint: "main",
                bindings: bindingsA,
                pushConstantBytes: APushConstantBytes);
            poolA = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        }
        catch
        {
            moduleA?.Dispose();
            moduleB.Dispose();
            if (poolB != 0) VulkanApi.vkDestroyDescriptorPool(device.Handle, poolB, 0);
            if (poolA != 0) VulkanApi.vkDestroyDescriptorPool(device.Handle, poolA, 0);
            throw;
        }

        return new LoraDeltaGemvFusedF32Kernel(device, moduleB, pipelineB, poolB, moduleA, pipelineA, poolA);
    }

    /// <summary>
    /// Optional creator that returns <c>null</c> when either SPIR-V blob is
    /// missing (older builds). Lets the caller fall back to the un-fused
    /// 4-dispatch path without throwing.
    /// </summary>
    public static LoraDeltaGemvFusedF32Kernel? TryCreate(VulkanDevice device, string spvDir)
    {
        string pathB = Path.Combine(spvDir, "lora_delta_b_reduce_f32.spv");
        string pathA = Path.Combine(spvDir, "lora_delta_gemv_fused_f32.spv");
        if (!File.Exists(pathB) || !File.Exists(pathA)) return null;
        return Create(device, spvDir);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache()
    {
        _descriptorCacheB.Reset();
        _descriptorCacheA.Reset();
    }

    /// <summary>
    /// Synchronous launch — wraps <see cref="Record"/>; used by unit tests.
    /// Caller must allocate the rank-sized scratch buffer
    /// (<c>seqLen × rank × sizeof(float)</c>).
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer x, VulkanDevice.Buffer bWeight, VulkanDevice.Buffer aWeight,
        VulkanDevice.Buffer y, VulkanDevice.Buffer tmp,
        int seqLen, int inputDim, int outputDim, int rank)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, x, bWeight, aWeight, y, tmp, seqLen, inputDim, outputDim, rank);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the two-dispatch fused LoRA delta. <paramref name="tmp"/> must
    /// be at least <c>seqLen × rank × sizeof(float)</c> bytes; its contents
    /// are overwritten and not read after this call returns.
    /// </summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer x, VulkanDevice.Buffer bWeight, VulkanDevice.Buffer aWeight,
        VulkanDevice.Buffer y, VulkanDevice.Buffer tmp,
        int seqLen, int inputDim, int outputDim, int rank)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
        if (outputDim <= 0) throw new ArgumentOutOfRangeException(nameof(outputDim));
        if (rank <= 0) throw new ArgumentOutOfRangeException(nameof(rank));
        if (rank > MaxRank)
            throw new ArgumentOutOfRangeException(
                nameof(rank), $"Rank {rank} exceeds fused-shader cap {MaxRank}; route through the un-fused path.");

        long xBytes = (long)seqLen * inputDim * sizeof(float);
        long bBytes = (long)rank * inputDim * sizeof(float);
        long aBytes = (long)outputDim * rank * sizeof(float);
        long yBytes = (long)seqLen * outputDim * sizeof(float);
        long tmpBytes = (long)seqLen * rank * sizeof(float);
        if (x.Size < xBytes) throw new ArgumentException("x buffer too small.", nameof(x));
        if (bWeight.Size < bBytes) throw new ArgumentException("bWeight buffer too small.", nameof(bWeight));
        if (aWeight.Size < aBytes) throw new ArgumentException("aWeight buffer too small.", nameof(aWeight));
        if (y.Size < yBytes) throw new ArgumentException("y buffer too small.", nameof(y));
        if (tmp.Size < tmpBytes) throw new ArgumentException("tmp buffer too small.", nameof(tmp));

        // Stage B: tmp[t, r] = dot(B[r, :], x[t, :]).
        Span<nint> buffersB = stackalloc nint[3] { x.Handle, bWeight.Handle, tmp.Handle };
        nint setB = _descriptorCacheB.GetOrCreate(buffersB);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipelineB.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipelineB.Layout,
            0, 1, setB, 0, 0);

        Span<uint> pcB = stackalloc uint[3] { (uint)inputDim, (uint)rank, (uint)seqLen };
        fixed (uint* pcPtr = pcB)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipelineB.Layout, VkShaderStageFlags.Compute,
                0, BPushConstantBytes, (nint)pcPtr);
        }
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)rank, (uint)seqLen, 1);

        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        // Stage A: y[t, m] += sum_r A[m, r] * tmp[t, r], in place.
        Span<nint> buffersA = stackalloc nint[3] { tmp.Handle, aWeight.Handle, y.Handle };
        nint setA = _descriptorCacheA.GetOrCreate(buffersA);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipelineA.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipelineA.Layout,
            0, 1, setA, 0, 0);

        Span<uint> pcA = stackalloc uint[3] { (uint)outputDim, (uint)rank, (uint)seqLen };
        fixed (uint* pcPtr = pcA)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipelineA.Layout, VkShaderStageFlags.Compute,
                0, APushConstantBytes, (nint)pcPtr);
        }

        uint groupsX = (uint)((outputDim + WorkgroupTile - 1) / WorkgroupTile);
        VulkanApi.vkCmdDispatch(cmdBuf, groupsX, (uint)seqLen, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_descriptorPoolB != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPoolB, 0);
        if (_descriptorPoolA != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPoolA, 0);
        _pipelineB.Dispose();
        _pipelineA.Dispose();
        _moduleB.Dispose();
        _moduleA.Dispose();
    }
}
