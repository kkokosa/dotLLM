using System.Runtime.InteropServices;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Latent (compressed) KV cache for MLA layers — the production storage
/// layout that turns MLA's ~8× KV-memory reduction into a real win. This
/// is the <b>Phase B</b> cache; <see cref="MlaExpandedKvState"/> remains
/// the Phase A correctness oracle.
/// </summary>
/// <remarks>
/// <para>
/// <b>What is stored</b> (per layer, per token, 64-byte-aligned native):
/// </para>
/// <list type="bullet">
///   <item><c>Latent[layer]</c> : <c>[maxSeqLen, kv_lora_rank]</c> — the
///     compressed <c>c_kv = RMSNorm(kv_a_proj @ hidden)</c> shared across
///     all heads.</item>
///   <item><c>KPe[layer]</c> : <c>[maxSeqLen, qk_rope_head_dim]</c> — the
///     single MQA-shared rope-K, identical to Phase A.</item>
/// </list>
/// <para>
/// <b>What is NOT stored</b>: the per-head <c>K_nope</c> and per-head
/// <c>V</c> that Phase A writes to memory. These are recovered at
/// attention time by the absorbed kernel: the nope half uses
/// <c>Q_latent = W_UK_T @ Q_nope</c> and dots against the shared latent;
/// the V side is expanded on the way out via <c>out = W_UV @ out_latent</c>.
/// </para>
/// <para>
/// <b>Memory footprint (DeepSeek-V2-Lite, F32, per token per layer)</b>:
/// Phase A = <c>(16·128 + 16·128 + 64)·4 = 16,640 B</c>.
/// Phase B = <c>(512 + 64)·4 = 2,304 B</c>. Ratio 7.22×. At 8K context
/// over 27 layers the Phase A cache is ~3.6 GB vs Phase B's 500 MB.
/// </para>
/// <para>
/// <b>Lifecycle</b> is identical to <see cref="MlaExpandedKvState"/>:
/// lazily constructed on the first MLA forward, <see cref="Reset"/> at
/// <c>positions[0] == 0</c>, <see cref="Advance"/> after each layer's
/// kernel call. Single-stream only; not thread-safe.
/// </para>
/// </remarks>
internal sealed unsafe class MlaLatentKvState : IDisposable
{
    private readonly int _numLayers;
    private readonly int _maxSeqLen;
    private readonly int _kvLoraRank;
    private readonly int _qkRopeHeadDim;

    private readonly nint[] _latentBuffers;
    private readonly nint[] _kPeBuffers;
    private readonly int[] _currentLengths;

    /// <summary>
    /// Total bytes held across Latent + K_pe for all layers at the
    /// configured max sequence length.
    /// </summary>
    public long AllocatedBytes
    {
        get
        {
            long perTokenLatentBytes = (long)_kvLoraRank * sizeof(float);
            long perTokenKPeBytes = (long)_qkRopeHeadDim * sizeof(float);
            return _numLayers * _maxSeqLen * (perTokenLatentBytes + perTokenKPeBytes);
        }
    }

    public int MaxSeqLen => _maxSeqLen;
    public int NumLayers => _numLayers;

    public MlaLatentKvState(int numLayers, int maxSeqLen, int kvLoraRank, int qkRopeHeadDim)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (kvLoraRank <= 0) throw new ArgumentOutOfRangeException(nameof(kvLoraRank));

        _numLayers = numLayers;
        _maxSeqLen = maxSeqLen;
        _kvLoraRank = kvLoraRank;
        _qkRopeHeadDim = qkRopeHeadDim;

        _latentBuffers = new nint[numLayers];
        _kPeBuffers = new nint[numLayers];
        _currentLengths = new int[numLayers];

        long latentFloatsPerLayer = (long)maxSeqLen * kvLoraRank;
        long kPeFloatsPerLayer = (long)maxSeqLen * qkRopeHeadDim;

        for (int i = 0; i < numLayers; i++)
        {
            _latentBuffers[i] = AllocFloats(latentFloatsPerLayer);
            _kPeBuffers[i] = AllocFloats(kPeFloatsPerLayer);
        }
    }

    public void Reset() => Array.Clear(_currentLengths);

    public int GetCurrentLength(int layerIndex) => _currentLengths[layerIndex];

    public void Advance(int layerIndex, int tokensAdded)
    {
        int newLen = _currentLengths[layerIndex] + tokensAdded;
        if (newLen > _maxSeqLen)
            throw new InvalidOperationException(
                $"MLA latent cache overflow on layer {layerIndex}: {newLen} > maxSeqLen={_maxSeqLen}.");
        _currentLengths[layerIndex] = newLen;
    }

    /// <summary>
    /// Native pointer to the <c>[maxSeqLen, kv_lora_rank]</c> latent buffer
    /// for the given layer (post-RMSNorm, pre-kv_b expansion).
    /// </summary>
    public nint GetLatentPointer(int layerIndex) => _latentBuffers[layerIndex];

    /// <summary>
    /// Native pointer to the <c>[maxSeqLen, qk_rope_head_dim]</c> shared
    /// K_pe buffer for the given layer (post-RoPE rotation).
    /// </summary>
    public nint GetKPePointer(int layerIndex) => _kPeBuffers[layerIndex];

    public void Dispose()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            FreeIfNonZero(ref _latentBuffers[i]);
            FreeIfNonZero(ref _kPeBuffers[i]);
        }
    }

    private static nint AllocFloats(long count) =>
        (nint)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);

    private static void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0)
        {
            NativeMemory.AlignedFree((void*)ptr);
            ptr = 0;
        }
    }
}
