namespace DotLLM.Core.Backends;

/// <summary>
/// Coarse-grained hints about which phase of inference a compute backend is best at,
/// plus a crossover token count at which the bias flips. Used by hybrid execution
/// strategies (e.g. CPU prefill + iGPU decode) to route work between backends that
/// share an address space.
/// </summary>
/// <remarks>
/// <para>
/// This is a hint, not a contract. Strategies still allocate KV-caches per backend
/// and perform an explicit handoff at the boundary. The capability struct only
/// answers "would this backend prefer to do prefill / decode for a prompt of length N?".
/// </para>
/// <para>
/// The CPU bias is towards short prefills (AVX-512 + Q4 dequant is competitive with
/// an iGPU when the matmul is too small to amortise pipeline-cache / descriptor-
/// binding warmup). The iGPU bias is towards decode (memory-bandwidth-bound, where
/// the iGPU's wide LPDDR5X / DDR5 bus wins regardless of compute peak). The
/// crossover threshold encodes this: for prompts shorter than
/// <see cref="PrefillCrossoverTokens"/>, prefer CPU prefill; longer, prefer iGPU prefill.
/// </para>
/// <para>
/// Empirically derived defaults (Strix Halo, 2026-05-14 GAIA/lemonade-server research):
/// CPU pp512 ≈ 294 tok/s, Vulkan iGPU pp512 ≈ 884 tok/s, but the iGPU pays ~30-80 ms
/// of warm-up tax per cold prefill (pipeline-cache lookup, descriptor binding) that
/// dominates a 64-token prefill. The break-even on TinyLlama-class Q8_0 weights is
/// around 256 tokens.
/// </para>
/// </remarks>
public readonly record struct BackendCapabilities
{
    /// <summary>
    /// Whether this backend is biased towards <em>prefill</em> (compute-bound, batched GEMM).
    /// True for: high-FLOP CPUs (AVX-512), iGPUs/dGPUs on long prompts. False for: NPUs
    /// without batched GEMM support, decode-optimised inference engines.
    /// </summary>
    public bool PrefersPrefill { get; init; }

    /// <summary>
    /// Whether this backend is biased towards <em>decode</em> (memory-bandwidth-bound, GEMV).
    /// True for: GPUs with high memory bandwidth, iGPUs on unified-memory APUs. False for:
    /// CPUs (memory bandwidth is shared with the OS and limits decode throughput).
    /// </summary>
    public bool PrefersDecode { get; init; }

    /// <summary>
    /// Prompt-length crossover above which the iGPU prefill wins over CPU prefill
    /// (i.e. for <c>promptTokens &lt; PrefillCrossoverTokens</c>, prefer CPU; for
    /// <c>&gt;= PrefillCrossoverTokens</c>, prefer this backend). Only meaningful on
    /// backends where <see cref="PrefersPrefill"/> is true. Default 256 — empirically
    /// validated on Strix Halo with Q8_0 weights.
    /// </summary>
    public int PrefillCrossoverTokens { get; init; }

    /// <summary>
    /// Whether weights can be shared zero-copy with other backends that mmap the same
    /// source file. True for: CPU (reads directly from the mmap'd GGUF view), Vulkan
    /// on UMA APUs when <c>VK_KHR_external_memory_host</c> is supported. False for:
    /// CUDA (must <c>cudaMemcpy</c> into VRAM), Vulkan on dGPUs.
    /// </summary>
    /// <remarks>
    /// Hybrid strategies use this to decide whether to load weights into both backends
    /// (current default, ~2× weight memory) or to share a single mmap'd view (future
    /// optimisation — recommendation H3 in the GAIA/lemonade-server research note).
    /// </remarks>
    public bool SupportsZeroCopyMmap { get; init; }

    /// <summary>
    /// Capability profile for the CPU SIMD backend (<c>DotLLM.Cpu.CpuBackend</c>).
    /// Prefers prefill below the crossover, never prefers decode, and supports
    /// zero-copy mmap by definition.
    /// </summary>
    public static BackendCapabilities Cpu { get; } = new()
    {
        PrefersPrefill = true,
        PrefersDecode = false,
        PrefillCrossoverTokens = DefaultPrefillCrossoverTokens,
        SupportsZeroCopyMmap = true,
    };

    /// <summary>
    /// Capability profile for the Vulkan iGPU backend on a unified-memory APU
    /// (Strix Halo / Strix Point / mobile Intel Arc). Prefers prefill above the
    /// crossover and prefers decode unconditionally. Mmap-sharing potential is
    /// declared as false today because <c>VK_KHR_external_memory_host</c> wiring
    /// is a follow-up.
    /// </summary>
    public static BackendCapabilities VulkanIgpu { get; } = new()
    {
        PrefersPrefill = true,
        PrefersDecode = true,
        PrefillCrossoverTokens = DefaultPrefillCrossoverTokens,
        SupportsZeroCopyMmap = false,
    };

    /// <summary>
    /// Capability profile for a discrete CUDA GPU. Prefers prefill (high FP16/BF16
    /// FLOPs + cuBLAS) and decode (HBM bandwidth). Cannot share weights zero-copy
    /// because device memory is physically separate.
    /// </summary>
    public static BackendCapabilities CudaDiscrete { get; } = new()
    {
        PrefersPrefill = true,
        PrefersDecode = true,
        PrefillCrossoverTokens = 1, // crossover effectively never — always prefer GPU
        SupportsZeroCopyMmap = false,
    };

    /// <summary>
    /// Default crossover threshold (tokens). Configurable per-strategy and via the
    /// <c>DOTLLM_HYBRID_PREFILL_CROSSOVER</c> environment variable.
    /// </summary>
    public const int DefaultPrefillCrossoverTokens = 256;

    /// <summary>
    /// Reads the <c>DOTLLM_HYBRID_PREFILL_CROSSOVER</c> environment variable. When set
    /// to a positive integer, returns that value; otherwise returns
    /// <see cref="DefaultPrefillCrossoverTokens"/>. Used by
    /// <c>HybridPrefillDecodeStrategy</c> to let operators tune the crossover without
    /// recompiling.
    /// </summary>
    public static int ReadCrossoverFromEnvironment()
    {
        string? raw = System.Environment.GetEnvironmentVariable("DOTLLM_HYBRID_PREFILL_CROSSOVER");
        if (!string.IsNullOrEmpty(raw)
            && int.TryParse(raw, out int parsed)
            && parsed > 0)
        {
            return parsed;
        }
        return DefaultPrefillCrossoverTokens;
    }
}
