using System.Buffers;
using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Backends;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;

namespace DotLLM.Engine.Strategies;

/// <summary>
/// Callback that copies CPU prefill KV state into a decode-side
/// (typically Vulkan / GPU) KV cache. Invoked once after CPU prefill
/// completes, with the populated host-side cache and the freshly
/// allocated decode-side cache as arguments. The callback owns
/// concerns specific to the target backend (staging buffer allocation,
/// per-layer copy ordering, fence waiting); the engine merely orchestrates
/// the call.
/// </summary>
/// <remarks>
/// Engine deliberately does not reference <c>DotLLM.Vulkan</c> or any
/// other backend — the handoff is supplied by the caller wiring up the
/// strategy (typically in <c>DotLLM.Cli</c> or a sample). The default
/// wiring for Vulkan iterates layers, reading
/// <c>SimpleKvCache.KeysSpan</c> / <c>SimpleKvCache.ValuesSpan</c>
/// and calling <c>VulkanKvCache.IngestFromHost</c>.
/// </remarks>
/// <param name="prefillCache">Host-resident KV state populated by CPU prefill.</param>
/// <param name="decodeCache">Decode-side KV cache, freshly allocated and empty.</param>
public delegate void HybridKvHandoff(SimpleKvCache prefillCache, IKvCache decodeCache);

/// <summary>
/// Coordinated CPU-prefill / GPU-decode strategy: runs the prompt's prefill
/// on a CPU-resident <see cref="IModel"/>, transfers the KV state to a
/// GPU-resident sibling model, then defers the rest of generation to the
/// caller's normal decode loop (typically a <see cref="TextGenerator"/>
/// backed by the decode model).
/// </summary>
/// <remarks>
/// <para>
/// <b>Why.</b> On unified-memory APUs (Strix Halo) the iGPU pays a cold
/// pipeline-cache / descriptor-binding warm-up tax per prefill that
/// dominates short (&lt;256 token) prompts. Meanwhile a Zen 5 CPU with
/// AVX-512 can prefill at ~300 tok/s — competitive when the warm-up tax is
/// accounted for. Decode is memory-bandwidth-bound and the iGPU keeps the
/// lead regardless of prompt length. Switching at the prefill/decode boundary
/// captures both biases. See <c>.planning/notes/gaia-lemonade-research.md</c>
/// section 6 H4.
/// </para>
/// <para>
/// <b>Why dotLLM can do this and lemonade-server cannot.</b> Lemonade routes
/// each backend (llama.cpp Vulkan, llama.cpp CPU, ryzenai-server, vLLM ROCm)
/// to a separate subprocess (<c>WrappedServer</c> + <c>ProcessManager</c>).
/// The KV state never crosses the process boundary, so there is no in-process
/// handoff point. dotLLM is a single .NET process: every <see cref="IModel"/>
/// lives in the same address space and can share the same mmap'd GGUF view,
/// and copying KV bytes between heaps is a memcpy.
/// </para>
/// <para>
/// <b>What this class does</b>. <see cref="RunPrefill"/> takes the prompt
/// token ids and runs a single Forward on the CPU model against an internal
/// <see cref="SimpleKvCache"/>; the cache is populated for positions
/// <c>[0, promptLen)</c>. The result is a <see cref="PrefillHandoff"/>
/// carrying the host-side cache, the first generated token's logits, and
/// the per-phase timings. The caller then invokes <see cref="Handoff"/> to
/// transfer the cache into the decode-side <see cref="IKvCache"/> and
/// continues decoding via its preferred path.
/// </para>
/// <para>
/// <b>What this class does <em>not</em> do</b>. It does not run the decode
/// loop itself — that stays in <see cref="TextGenerator"/>. It does not
/// own model loading or tokenization — the caller wires those up. It does
/// not enable hybrid mode automatically; the caller checks
/// <see cref="ShouldRunHybrid"/> against the prompt length.
/// </para>
/// </remarks>
public sealed class HybridPrefillDecodeStrategy
{
    private readonly IModel _prefillModel;
    private readonly IModel _decodeModel;
    private readonly HybridKvHandoff _handoff;
    private readonly int _crossoverTokens;

    /// <summary>
    /// Capability profile assumed for the prefill backend (typically CPU).
    /// </summary>
    public BackendCapabilities PrefillCapabilities { get; }

    /// <summary>
    /// Capability profile assumed for the decode backend (typically Vulkan iGPU).
    /// </summary>
    public BackendCapabilities DecodeCapabilities { get; }

    /// <summary>
    /// Prompt length below which hybrid mode is chosen (CPU prefill + GPU decode).
    /// At or above this length, the caller should prefer pure-GPU prefill+decode.
    /// </summary>
    public int CrossoverTokens => _crossoverTokens;

    /// <summary>The prefill-side model (typically CPU).</summary>
    public IModel PrefillModel => _prefillModel;

    /// <summary>The decode-side model (typically Vulkan iGPU).</summary>
    public IModel DecodeModel => _decodeModel;

    /// <summary>
    /// Creates the strategy. Both <paramref name="prefillModel"/> and
    /// <paramref name="decodeModel"/> must describe the same architecture —
    /// in practice they are loaded from the same GGUF file by two backends
    /// (e.g. <c>TransformerModel.LoadFromGguf</c> for CPU and
    /// <c>VulkanTransformerModel.LoadFromGguf</c> for Vulkan).
    /// </summary>
    /// <param name="prefillModel">Model that runs the prompt's forward pass. Typically CPU.</param>
    /// <param name="decodeModel">Model that runs the autoregressive decode loop. Typically Vulkan iGPU.</param>
    /// <param name="handoff">Callback transferring CPU prefill KV state into the decode-side cache.</param>
    /// <param name="prefillCapabilities">Optional capability profile for the prefill backend; defaults to <see cref="BackendCapabilities.Cpu"/>.</param>
    /// <param name="decodeCapabilities">Optional capability profile for the decode backend; defaults to <see cref="BackendCapabilities.VulkanIgpu"/>.</param>
    /// <param name="crossoverTokens">Optional override for the crossover threshold. Defaults to <see cref="BackendCapabilities.ReadCrossoverFromEnvironment"/>.</param>
    public HybridPrefillDecodeStrategy(
        IModel prefillModel,
        IModel decodeModel,
        HybridKvHandoff handoff,
        BackendCapabilities? prefillCapabilities = null,
        BackendCapabilities? decodeCapabilities = null,
        int? crossoverTokens = null)
    {
        ArgumentNullException.ThrowIfNull(prefillModel);
        ArgumentNullException.ThrowIfNull(decodeModel);
        ArgumentNullException.ThrowIfNull(handoff);

        if (prefillModel.Config.NumLayers != decodeModel.Config.NumLayers
            || prefillModel.Config.NumKvHeads != decodeModel.Config.NumKvHeads
            || prefillModel.Config.HeadDim != decodeModel.Config.HeadDim
            || prefillModel.Config.HiddenSize != decodeModel.Config.HiddenSize
            || prefillModel.Config.VocabSize != decodeModel.Config.VocabSize)
        {
            throw new ArgumentException(
                "Prefill and decode models must describe the same architecture "
                + "(layers, KV heads, head dim, hidden size, vocab). Got: "
                + $"prefill L={prefillModel.Config.NumLayers} KV={prefillModel.Config.NumKvHeads}×{prefillModel.Config.HeadDim} "
                + $"vs decode L={decodeModel.Config.NumLayers} KV={decodeModel.Config.NumKvHeads}×{decodeModel.Config.HeadDim}.");
        }

        _prefillModel = prefillModel;
        _decodeModel = decodeModel;
        _handoff = handoff;
        PrefillCapabilities = prefillCapabilities ?? BackendCapabilities.Cpu;
        DecodeCapabilities = decodeCapabilities ?? BackendCapabilities.VulkanIgpu;
        _crossoverTokens = crossoverTokens ?? BackendCapabilities.ReadCrossoverFromEnvironment();
    }

    /// <summary>
    /// Whether hybrid mode is preferred for the given prompt length. Returns
    /// <c>true</c> when <paramref name="promptTokens"/> is strictly less than
    /// <see cref="CrossoverTokens"/>; <c>false</c> at or above the threshold
    /// (where the decode backend's pure-prefill path is expected to win).
    /// </summary>
    public bool ShouldRunHybrid(int promptTokens) =>
        promptTokens > 0 && promptTokens < _crossoverTokens;

    /// <summary>
    /// Runs CPU prefill on <paramref name="promptIds"/> against an internal
    /// <see cref="SimpleKvCache"/> sized for the full <paramref name="cacheSize"/>
    /// (prompt + generation). Returns the populated KV state and the logits row
    /// for the last prompt position — the caller samples this to get the first
    /// generated token.
    /// </summary>
    /// <remarks>
    /// The returned <see cref="PrefillHandoff.HostCache"/> is owned by the
    /// caller and must be disposed (typically after the handoff completes).
    /// The returned <see cref="PrefillHandoff.LastLogits"/> is heap-allocated
    /// (one row of vocabSize FP32) and may be safely returned to the caller —
    /// it is detached from the model's per-forward scratch.
    /// </remarks>
    /// <param name="promptIds">Tokenised prompt to prefill.</param>
    /// <param name="cacheSize">Total KV-cache slots to allocate (prompt + max generation).</param>
    /// <returns>The populated KV state plus the last-position logits row.</returns>
    public PrefillHandoff RunPrefill(ReadOnlySpan<int> promptIds, int cacheSize)
    {
        if (promptIds.Length == 0)
            throw new ArgumentException("promptIds must be non-empty.", nameof(promptIds));
        if (cacheSize < promptIds.Length)
            throw new ArgumentException("cacheSize must be at least promptIds.Length.", nameof(cacheSize));

        var cfg = _prefillModel.Config;
        var hostCache = new SimpleKvCache(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, cacheSize);

        int promptLen = promptIds.Length;
        int[] positionsArray = ArrayPool<int>.Shared.Rent(promptLen);
        float[] logitsRow;
        long ticks;
        try
        {
            Span<int> positions = positionsArray.AsSpan(0, promptLen);
            for (int i = 0; i < promptLen; i++) positions[i] = i;

            long t0 = Stopwatch.GetTimestamp();
            using (ITensor prefillLogits = _prefillModel.Forward(promptIds, positions, deviceId: -1, hostCache))
            {
                ticks = Stopwatch.GetTimestamp() - t0;

                // Extract last-position logits as a managed array so the caller
                // can sample without holding the model's per-forward scratch
                // tensor alive (it would otherwise be reused on the next call).
                int vocab = cfg.VocabSize;
                int rows = prefillLogits.Shape[0];
                logitsRow = new float[vocab];
                unsafe
                {
                    float* src = (float*)prefillLogits.DataPointer + (long)(rows - 1) * vocab;
                    new ReadOnlySpan<float>(src, vocab).CopyTo(logitsRow);
                }
            }
        }
        catch
        {
            hostCache.Dispose();
            throw;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(positionsArray);
        }

        return new PrefillHandoff(hostCache, logitsRow, ticks);
    }

    /// <summary>
    /// Transfers prefill KV state from <paramref name="hostCache"/> into
    /// <paramref name="decodeCache"/> using the configured handoff callback.
    /// After this returns, <paramref name="decodeCache"/> contains positions
    /// <c>[0, hostCache.CurrentLength)</c> populated and ready for decode.
    /// </summary>
    /// <remarks>
    /// Does <em>not</em> dispose <paramref name="hostCache"/> — the caller
    /// retains ownership and may reuse / dispose it as appropriate.
    /// </remarks>
    public void Handoff(SimpleKvCache hostCache, IKvCache decodeCache)
    {
        ArgumentNullException.ThrowIfNull(hostCache);
        ArgumentNullException.ThrowIfNull(decodeCache);
        if (hostCache.CurrentLength == 0)
            throw new ArgumentException("hostCache is empty; nothing to hand off.", nameof(hostCache));
        if (decodeCache.MaxLength < hostCache.CurrentLength)
            throw new ArgumentException(
                $"decodeCache.MaxLength {decodeCache.MaxLength} < prefill length {hostCache.CurrentLength}.",
                nameof(decodeCache));
        _handoff(hostCache, decodeCache);
    }
}

/// <summary>
/// Output of <see cref="HybridPrefillDecodeStrategy.RunPrefill"/>: the populated
/// host-side KV cache, the last-position logits row, and the prefill duration in
/// <see cref="Stopwatch"/> ticks.
/// </summary>
/// <param name="HostCache">Host-resident <see cref="SimpleKvCache"/> populated for positions <c>[0, promptLen)</c>. Caller owns and disposes.</param>
/// <param name="LastLogits">Logits row at the last prompt position, length <c>vocabSize</c>. Heap-allocated; safe to sample from after the prefill model's tensor is disposed.</param>
/// <param name="PrefillTicks"><see cref="Stopwatch"/> ticks elapsed during prefill (Forward only, excluding KV allocation / handoff).</param>
public readonly record struct PrefillHandoff(
    SimpleKvCache HostCache,
    float[] LastLogits,
    long PrefillTicks);
