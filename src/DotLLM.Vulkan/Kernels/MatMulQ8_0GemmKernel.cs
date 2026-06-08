using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q8_0 prefill-path batched GEMM: <c>C[N, M] = B[N, K] @ W_q8[M, K]^T</c>.
/// </summary>
/// <remarks>
/// <para>
/// Semantic parity with <c>DotLLM.Cpu.Kernels.MatMul.GemmQ8_0</c>:
/// <list type="bullet">
///   <item><c>W_q8</c> is a row-major <c>[M, K]</c> weight matrix stored as
///     <c>(K / 32)</c> Q8_0 blocks per row (34 bytes each: fp16 scale +
///     32 int8 values).</item>
///   <item><c>B</c> is row-major <c>[N, K]</c> FP32 input, one row per token.</item>
///   <item><c>C</c> is row-major <c>[N, M]</c> FP32 output;
///     <c>C[t, m] = dot(W[m, :], B[t, :])</c>.</item>
/// </list>
/// </para>
/// <para>
/// Companion to <see cref="MatMulQ8_0Kernel"/> (the decode-path GEMV). The GEMV
/// path dispatches one workgroup per output row which is bandwidth-bound for
/// large <c>M</c> and leaves weight reuse on the table when multiple tokens
/// share the same weight matrix. This kernel instead tiles the output: one
/// 16×16 cell of <c>C</c> per workgroup, with the 16-row weight tile
/// dequantized once per K-chunk into shared memory and reused across 16
/// tokens.
/// </para>
/// <para>
/// Dispatch: 2-D grid, workgroup <c>(16, 16, 1)</c>. No subgroup or
/// cooperative-matrix intrinsics yet — broadest driver portability and
/// correctness first. A follow-up subgroup-tiled variant is the intended next
/// step if the CUDA perf gap remains large.
/// </para>
/// </remarks>
public sealed class MatMulQ8_0GemmKernel : IDisposable
{
    /// <summary>Q8_0 block: 2 bytes fp16 scale + 32 signed int8 values.</summary>
    public const int Q8_0BlockBytes = 34;

    /// <summary>Elements per Q8_0 block.</summary>
    public const int Q8_0GroupSize = 32;

    private const int TileM = 16;
    private const int TileN = 16;
    private const int PushConstantBytes = 5 * sizeof(uint); // M, K, N, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ8_0GemmKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>matmul_q8_0_gemm.spv</c> from the given directory and creates the pipeline.</summary>
    public static MatMulQ8_0GemmKernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "matmul_q8_0_gemm.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[3];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main",
                bindings: bindings,
                pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new MatMulQ8_0GemmKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches the batched GEMM: <c>C[N, M] = B[N, K] @ W_q8[M, K]^T</c>.
    /// Synchronous — returns after <c>vkQueueWaitIdle</c>. Legacy wrapper around
    /// <see cref="Record"/>.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ8, inputB, outputC, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the Q8_0 GEMM into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="cmdBuf">Open Vulkan command buffer to append commands to.</param>
    /// <param name="weightsQ8">
    /// Raw Q8_0 blob of <c>M * (K / 32) * 34</c> bytes, rows contiguous.
    /// </param>
    /// <param name="inputB">FP32 input <c>[N, K]</c> row-major.</param>
    /// <param name="outputC">FP32 output <c>[N, M]</c> row-major.</param>
    /// <param name="m">Output dimension (number of weight rows).</param>
    /// <param name="k">Contraction dimension (must be a multiple of 32).</param>
    /// <param name="n">Batch size (number of input tokens).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
        int m, int k, int n)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if ((k % Q8_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q8_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q8_0BlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsQ8.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ8.Size}.",
                nameof(weightsQ8));
        long bMin = (long)n * k * sizeof(float);
        long cMin = (long)n * m * sizeof(float);
        if (inputB.Size < bMin) throw new ArgumentException("Input buffer too small.", nameof(inputB));
        if (outputC.Size < cMin) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        Span<nint> buffers = stackalloc nint[3] { weightsQ8.Handle, inputB.Handle, outputC.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[5]
        {
            (uint)m,
            (uint)k,
            (uint)n,
            (uint)blocksPerRow,
            (uint)rowUints,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint groupsX = (uint)((m + TileM - 1) / TileM);
        uint groupsY = (uint)((n + TileN - 1) / TileN);
        VulkanApi.vkCmdDispatch(cmdBuf, groupsX, groupsY, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_descriptorPool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
