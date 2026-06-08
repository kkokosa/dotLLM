using System.Runtime.InteropServices;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Persistent KV cache for MLA (Multi-head Latent Attention) layers. Stores
/// expanded per-head <c>K_nope</c>, per-head <c>V</c>, and the shared
/// <c>K_pe</c> (MQA-style decoupled rope K) for each layer across calls.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this is not an <see cref="DotLLM.Core.Attention.IKvCache"/>.</b> The
/// existing <c>IKvCache</c> contract returns K and V tensors that share a
/// uniform head dimension — it's designed for GQA/MHA where
/// <c>qk_head_dim == v_head_dim</c>. MLA deliberately decouples them
/// (V2-Lite: qk=192, v=128) and adds a shared K_pe that is broadcast across
/// heads. Shoehorning that into <c>IKvCache</c> would require either
/// redundant per-head K_pe storage or an interface change that leaks MLA
/// specifics. A dedicated holder stays honest — and keeps the door open
/// for a future latent cache (store the uncompressed
/// <c>[kv_lora_rank + qk_rope_head_dim]</c> per token, ~8× smaller)
/// without disturbing the GQA/MHA cache path.
/// </para>
/// <para>
/// <b>Layout.</b> All buffers 64-byte aligned via
/// <see cref="NativeMemory.AlignedAlloc(nuint,nuint)"/>. Matches the MLA
/// kernel's scratch-buffer layout one-for-one so the kernel can memcpy new
/// rows in with no shape translation. Per layer:
/// </para>
/// <list type="bullet">
///   <item><c>KNope[layer]</c> : <c>[maxSeqLen, numHeads * qkNopeHeadDim]</c>
///     — per-head non-rope K (the dominant stored term).</item>
///   <item><c>V[layer]</c> : <c>[maxSeqLen, numHeads * vHeadDim]</c> — per-head V.</item>
///   <item><c>KPe[layer]</c> : <c>[maxSeqLen, qkRopeHeadDim]</c> — the single
///     MQA rope-K broadcast across heads, stored once per token per layer
///     (already RoPE-applied — we cache the post-rotation value).</item>
/// </list>
/// <para>
/// <b>Lifecycle.</b> Owned by the <see cref="TransformerModel"/> instance.
/// When <c>positions[0] == 0</c> at the start of a forward pass, the caller
/// resets via <see cref="Reset"/>. Each successful layer call advances
/// <see cref="GetCurrentLength"/> by <c>seqLen</c>.
/// </para>
/// <para>
/// <b>Not re-entrant / not thread-safe.</b> Single-stream only. Batching or
/// beam search needs a per-sequence instance.
/// </para>
/// </remarks>
internal sealed unsafe class MlaExpandedKvState : IDisposable
{
    private readonly int _numLayers;
    private readonly int _maxSeqLen;
    private readonly int _numHeads;
    private readonly int _qkNopeHeadDim;
    private readonly int _vHeadDim;
    private readonly int _qkRopeHeadDim;

    private readonly nint[] _kNopeBuffers;   // _numLayers entries
    private readonly nint[] _vBuffers;
    private readonly nint[] _kPeBuffers;
    private readonly int[] _currentLengths;

    /// <summary>
    /// Total bytes held across K_nope + V + K_pe for all layers at the
    /// configured max sequence length. Useful for diagnostics / memory
    /// reporting.
    /// </summary>
    public long AllocatedBytes
    {
        get
        {
            long perTokenKBytes = (long)_numHeads * _qkNopeHeadDim * sizeof(float);
            long perTokenVBytes = (long)_numHeads * _vHeadDim * sizeof(float);
            long perTokenKPeBytes = (long)_qkRopeHeadDim * sizeof(float);
            return _numLayers * _maxSeqLen * (perTokenKBytes + perTokenVBytes + perTokenKPeBytes);
        }
    }

    public int MaxSeqLen => _maxSeqLen;
    public int NumLayers => _numLayers;

    public MlaExpandedKvState(
        int numLayers, int maxSeqLen,
        int numHeads, int qkNopeHeadDim, int vHeadDim, int qkRopeHeadDim)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));

        _numLayers = numLayers;
        _maxSeqLen = maxSeqLen;
        _numHeads = numHeads;
        _qkNopeHeadDim = qkNopeHeadDim;
        _vHeadDim = vHeadDim;
        _qkRopeHeadDim = qkRopeHeadDim;

        _kNopeBuffers = new nint[numLayers];
        _vBuffers = new nint[numLayers];
        _kPeBuffers = new nint[numLayers];
        _currentLengths = new int[numLayers];

        long kFloatsPerLayer = (long)maxSeqLen * numHeads * qkNopeHeadDim;
        long vFloatsPerLayer = (long)maxSeqLen * numHeads * vHeadDim;
        long kPeFloatsPerLayer = (long)maxSeqLen * qkRopeHeadDim;

        for (int i = 0; i < numLayers; i++)
        {
            _kNopeBuffers[i] = AllocFloats(kFloatsPerLayer);
            _vBuffers[i] = AllocFloats(vFloatsPerLayer);
            _kPeBuffers[i] = AllocFloats(kPeFloatsPerLayer);
        }
    }

    /// <summary>
    /// Resets the current length on every layer to 0, invalidating cached
    /// K/V/K_pe. The allocated buffers are retained and overwritten on the
    /// next <see cref="Advance"/>. Call at the start of a fresh sequence
    /// (i.e., when <c>positions[0] == 0</c>).
    /// </summary>
    public void Reset()
    {
        Array.Clear(_currentLengths);
    }

    /// <summary>
    /// Current number of cached tokens in the given layer. Expected to be the
    /// same across layers unless layers have been skipped, but each is
    /// tracked independently for correctness.
    /// </summary>
    public int GetCurrentLength(int layerIndex) => _currentLengths[layerIndex];

    /// <summary>
    /// Advances the cached length for <paramref name="layerIndex"/> by
    /// <paramref name="tokensAdded"/>. Called by the MLA forward path after
    /// successfully writing the new K/V/K_pe into the cache at
    /// <c>[currentLength..currentLength + tokensAdded)</c>.
    /// </summary>
    public void Advance(int layerIndex, int tokensAdded)
    {
        int newLen = _currentLengths[layerIndex] + tokensAdded;
        if (newLen > _maxSeqLen)
            throw new InvalidOperationException(
                $"MLA cache overflow on layer {layerIndex}: {newLen} > maxSeqLen={_maxSeqLen}.");
        _currentLengths[layerIndex] = newLen;
    }

    /// <summary>
    /// Native pointer to the <c>[maxSeqLen, numHeads * qkNopeHeadDim]</c>
    /// K_nope buffer for the given layer.
    /// </summary>
    public nint GetKNopePointer(int layerIndex) => _kNopeBuffers[layerIndex];

    /// <summary>
    /// Native pointer to the <c>[maxSeqLen, numHeads * vHeadDim]</c> V buffer
    /// for the given layer.
    /// </summary>
    public nint GetVPointer(int layerIndex) => _vBuffers[layerIndex];

    /// <summary>
    /// Native pointer to the <c>[maxSeqLen, qkRopeHeadDim]</c> shared K_pe
    /// buffer for the given layer.
    /// </summary>
    public nint GetKPePointer(int layerIndex) => _kPeBuffers[layerIndex];

    public void Dispose()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            FreeIfNonZero(ref _kNopeBuffers[i]);
            FreeIfNonZero(ref _vBuffers[i]);
            FreeIfNonZero(ref _kPeBuffers[i]);
        }
    }

    private static nint AllocFloats(long count)
    {
        return (nint)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);
    }

    private static void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0)
        {
            NativeMemory.AlignedFree((void*)ptr);
            ptr = 0;
        }
    }
}
