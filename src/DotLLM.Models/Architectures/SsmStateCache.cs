using System.Runtime.InteropServices;
using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Per-sequence recurrent state cache for the Mamba2 SSM sub-layers of a Nemotron-H
/// model. One cache instance holds the state for all SSM layers of a single sequence.
/// </summary>
/// <remarks>
/// <para>
/// For each SSM layer the cache stores:
/// </para>
/// <list type="bullet">
///   <item>
///     <description>
///       <c>conv_state</c> — the last <c>d_conv-1</c> rows of the conv1d input,
///       shape <c>[d_conv-1, conv_dim]</c> row-major, where <c>conv_dim = d_inner + 2*n_group*d_state</c>.
///     </description>
///   </item>
///   <item>
///     <description>
///       <c>ssm_state</c> — the recurrent SSM hidden state, shape
///       <c>[n_head, head_dim, d_state]</c> row-major.
///     </description>
///   </item>
/// </list>
/// <para>
/// Buffers are allocated in unmanaged memory via <see cref="NativeMemory.AlignedAlloc"/>
/// with 64-byte alignment (AVX-512 friendly) and zero-initialised on creation. The cache
/// is non-paged and single-sequence by design — a later stage can fold it into a paged
/// multi-sequence store mirroring the attention <c>IKvCache</c>.
/// </para>
/// <para>
/// Ownership: each cache owns its buffers and frees them in <see cref="Dispose"/>.
/// Forward passes obtain <see cref="Span{T}"/>s over the per-layer slices and mutate
/// them in place.
/// </para>
/// </remarks>
public sealed unsafe class SsmStateCache : IDisposable
{
    private readonly int _numSsmLayers;
    private readonly int _convStateElements;
    private readonly int _ssmStateElements;

    // Contiguous per-layer blocks. Layer i occupies:
    //   conv: _convState[i*convStateElements .. (i+1)*convStateElements)
    //   ssm:  _ssmState [i*ssmStateElements  .. (i+1)*ssmStateElements)
    private nint _convState;
    private nint _ssmState;

    private bool _disposed;

    /// <summary>Number of SSM layers covered by this cache.</summary>
    public int NumSsmLayers => _numSsmLayers;

    /// <summary>Elements per layer in the conv state (<c>(d_conv-1) * conv_dim</c>).</summary>
    public int ConvStateElements => _convStateElements;

    /// <summary>Elements per layer in the SSM state (<c>n_head * head_dim * d_state</c>).</summary>
    public int SsmStateElements => _ssmStateElements;

    /// <summary>
    /// Creates a new recurrent-state cache sized for the given SSM config and layer count.
    /// All buffers are zero-initialised.
    /// </summary>
    /// <param name="ssm">SSM hyperparameters shared by all SSM layers.</param>
    /// <param name="numSsmLayers">Number of SSM layers in the model (layers classified as <c>HybridLayerKind.Ssm</c>).</param>
    public SsmStateCache(MambaSsmConfig ssm, int numSsmLayers)
    {
        if (numSsmLayers < 0) throw new ArgumentOutOfRangeException(nameof(numSsmLayers));

        _numSsmLayers = numSsmLayers;
        _convStateElements = ssm.ConvStateElements; // (d_conv-1) * conv_dim
        _ssmStateElements = ssm.SsmStateElements;   // d_inner * d_state

        if (numSsmLayers == 0)
        {
            _convState = 0;
            _ssmState = 0;
            return;
        }

        long convBytes = (long)_numSsmLayers * _convStateElements * sizeof(float);
        long ssmBytes = (long)_numSsmLayers * _ssmStateElements * sizeof(float);

        _convState = (nint)NativeMemory.AlignedAlloc((nuint)convBytes, 64);
        _ssmState = (nint)NativeMemory.AlignedAlloc((nuint)ssmBytes, 64);

        // Mamba2 starts with zero state by convention (llama.cpp memsets the recurrent tensors on cache init).
        NativeMemory.Clear((void*)_convState, (nuint)convBytes);
        NativeMemory.Clear((void*)_ssmState, (nuint)ssmBytes);
    }

    /// <summary>
    /// Returns the conv-state slice for SSM layer <paramref name="ssmLayerIndex"/>.
    /// Indexed by SSM-layer ordinal, not by absolute block index.
    /// </summary>
    public Span<float> GetConvState(int ssmLayerIndex)
    {
        ThrowIfDisposed();
        if ((uint)ssmLayerIndex >= (uint)_numSsmLayers)
            throw new ArgumentOutOfRangeException(nameof(ssmLayerIndex));
        return new Span<float>(
            (float*)_convState + (long)ssmLayerIndex * _convStateElements,
            _convStateElements);
    }

    /// <summary>
    /// Returns the SSM-state slice for SSM layer <paramref name="ssmLayerIndex"/>.
    /// Indexed by SSM-layer ordinal, not by absolute block index.
    /// </summary>
    public Span<float> GetSsmState(int ssmLayerIndex)
    {
        ThrowIfDisposed();
        if ((uint)ssmLayerIndex >= (uint)_numSsmLayers)
            throw new ArgumentOutOfRangeException(nameof(ssmLayerIndex));
        return new Span<float>(
            (float*)_ssmState + (long)ssmLayerIndex * _ssmStateElements,
            _ssmStateElements);
    }

    /// <summary>Zeroes every layer's state. Useful at the start of a fresh sequence.</summary>
    public void Reset()
    {
        ThrowIfDisposed();
        if (_numSsmLayers == 0) return;
        NativeMemory.Clear((void*)_convState, (nuint)((long)_numSsmLayers * _convStateElements * sizeof(float)));
        NativeMemory.Clear((void*)_ssmState, (nuint)((long)_numSsmLayers * _ssmStateElements * sizeof(float)));
    }

    /// <summary>Total bytes allocated across both state buffers.</summary>
    public long AllocatedBytes =>
        (long)_numSsmLayers * (_convStateElements + _ssmStateElements) * sizeof(float);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        if (_convState != 0) { NativeMemory.AlignedFree((void*)_convState); _convState = 0; }
        if (_ssmState != 0) { NativeMemory.AlignedFree((void*)_ssmState); _ssmState = 0; }
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(SsmStateCache));
    }

    /// <summary>Finalizer — last-ditch free if the cache was not disposed.</summary>
    ~SsmStateCache()
    {
        if (_disposed) return;
        if (_convState != 0) NativeMemory.AlignedFree((void*)_convState);
        if (_ssmState != 0) NativeMemory.AlignedFree((void*)_ssmState);
    }
}
