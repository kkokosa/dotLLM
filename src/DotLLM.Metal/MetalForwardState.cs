using System.Numerics;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal;

/// <summary>
/// Per-request scratch state for the Metal forward pass. Every buffer is an
/// <c>MTLResourceStorageModeShared</c> MTLBuffer allocated via
/// <see cref="MetalNative.AllocShared"/>. The pointers exposed here are
/// <c>MTLBuffer.contents</c>, so:
/// <list type="bullet">
///   <item><description>The CPU can read/write them directly (regular host RAM).</description></item>
///   <item><description>Kernels can recover the backing MTLBuffer by pointer (registered
///   in the context's shared-buffer map) and use it zero-copy in encoders.</description></item>
/// </list>
/// This avoids per-kernel CPU↔Metal copies and is the prerequisite for
/// batched-command-buffer execution.
/// </summary>
public sealed class MetalForwardState : IDisposable
{
    private readonly nint _ctx;
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly int _vocabSize;

    private int _currentSeqLen;
    private bool _disposed;

    /// <summary>Total bytes currently allocated across all buffers.</summary>
    public long AllocatedBytes { get; private set; }

    /// <summary>Hidden-state stream — <c>[seqLen, hiddenSize]</c> FP16.</summary>
    public nint HiddenState   { get; private set; }
    /// <summary>Residual stream — <c>[seqLen, hiddenSize]</c> FP16.</summary>
    public nint Residual      { get; private set; }
    /// <summary>Normalization / projection output scratch — <c>[seqLen, hiddenSize]</c> FP16.</summary>
    public nint NormOutput    { get; private set; }
    /// <summary>Query projection — <c>[seqLen, numHeads * headDim]</c> FP16.</summary>
    public nint Q             { get; private set; }
    /// <summary>Key projection — <c>[seqLen, numKvHeads * headDim]</c> FP16.</summary>
    public nint K             { get; private set; }
    /// <summary>Value projection — <c>[seqLen, numKvHeads * headDim]</c> FP16.</summary>
    public nint V             { get; private set; }
    /// <summary>Attention output — <c>[seqLen, numHeads * headDim]</c> FP16.</summary>
    public nint AttnOutput    { get; private set; }
    /// <summary>FFN gate projection — <c>[seqLen, intermediateSize]</c> FP16.</summary>
    public nint FfnGate       { get; private set; }
    /// <summary>FFN up projection — <c>[seqLen, intermediateSize]</c> FP16.</summary>
    public nint FfnUp         { get; private set; }
    /// <summary>SwiGLU activation output — <c>[seqLen, intermediateSize]</c> FP16.</summary>
    public nint SiluOutput    { get; private set; }
    /// <summary>Final logits (FP16) — <c>[vocabSize]</c>.</summary>
    public nint LogitsF16     { get; private set; }
    /// <summary>Final logits (FP32, for the sampler) — <c>[vocabSize]</c>.</summary>
    public nint LogitsF32     { get; private set; }
    /// <summary>Scratch for GEMM (MPS) FP16 outputs.</summary>
    public nint GemmOutputF16 { get; private set; }
    /// <summary>Scratch for dequantized FP16 weights (on-the-fly dequant path).</summary>
    public nint DequantScratch { get; private set; }
    /// <summary>Input token ids — <c>[seqLen]</c> int32.</summary>
    public nint TokenIds      { get; private set; }
    /// <summary>Input positions — <c>[seqLen]</c> int32.</summary>
    public nint Positions     { get; private set; }

    /// <summary>
    /// Allocates fixed-size buffers (logits, dequant scratch) and an initial set
    /// of seq-length-dependent buffers sized for decode (<c>seqLen = 1</c>).
    /// All allocations come from the Metal device's shared-storage pool.
    /// </summary>
    public MetalForwardState(MetalContext ctx, int hiddenSize, int numHeads, int numKvHeads, int headDim,
                             int intermediateSize, int vocabSize)
    {
        ArgumentNullException.ThrowIfNull(ctx);
        _ctx = ctx.Handle;
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _intermediateSize = intermediateSize;
        _vocabSize = vocabSize;

        LogitsF16 = AllocShared((long)vocabSize * sizeof(ushort));
        LogitsF32 = AllocShared((long)vocabSize * sizeof(float));

        long maxProjectionElements = Math.Max(
            (long)Math.Max(numHeads * headDim, numKvHeads * headDim) * hiddenSize,
            (long)intermediateSize * hiddenSize);
        DequantScratch = AllocShared(maxProjectionElements * sizeof(ushort));

        EnsureCapacity(1);
    }

    /// <summary>Grows the seq-length-dependent buffers to fit <paramref name="seqLen"/> tokens.</summary>
    public void EnsureCapacity(int seqLen)
    {
        if (seqLen <= _currentSeqLen)
            return;

        int newCapacity = (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeSequenceBuffers();

        const int half = sizeof(ushort);

        HiddenState = AllocShared((long)newCapacity * _hiddenSize * half);
        Residual    = AllocShared((long)newCapacity * _hiddenSize * half);
        NormOutput  = AllocShared((long)newCapacity * _hiddenSize * half);
        Q           = AllocShared((long)newCapacity * _numHeads   * _headDim * half);
        K           = AllocShared((long)newCapacity * _numKvHeads * _headDim * half);
        V           = AllocShared((long)newCapacity * _numKvHeads * _headDim * half);
        AttnOutput  = AllocShared((long)newCapacity * _numHeads   * _headDim * half);
        FfnGate     = AllocShared((long)newCapacity * _intermediateSize * half);
        FfnUp       = AllocShared((long)newCapacity * _intermediateSize * half);
        SiluOutput  = AllocShared((long)newCapacity * _intermediateSize * half);

        long maxPerLayer = (long)newCapacity * Math.Max(
            Math.Max(_numHeads * _headDim, _intermediateSize), _hiddenSize);
        long maxLmHead = _vocabSize;
        GemmOutputF16 = AllocShared(Math.Max(maxPerLayer, maxLmHead) * half);

        TokenIds  = AllocShared((long)newCapacity * sizeof(int));
        Positions = AllocShared((long)newCapacity * sizeof(int));

        _currentSeqLen = newCapacity;
    }

    private nint AllocShared(long bytes)
    {
        nint ptr = MetalNative.AllocShared(_ctx, (nuint)bytes);
        if (ptr == 0)
            throw new InvalidOperationException(
                $"Metal shared allocation failed for {bytes} bytes (OOM or invalid context).");
        AllocatedBytes += bytes;
        return ptr;
    }

    private void FreeIfNonZero(ref nint ptr)
    {
        if (ptr == 0) return;
        MetalNative.FreeShared(_ctx, ptr);
        ptr = 0;
    }

    private void FreeSequenceBuffers()
    {
        nint p;
        p = HiddenState;   FreeIfNonZero(ref p); HiddenState   = p;
        p = Residual;      FreeIfNonZero(ref p); Residual      = p;
        p = NormOutput;    FreeIfNonZero(ref p); NormOutput    = p;
        p = Q;             FreeIfNonZero(ref p); Q             = p;
        p = K;             FreeIfNonZero(ref p); K             = p;
        p = V;             FreeIfNonZero(ref p); V             = p;
        p = AttnOutput;    FreeIfNonZero(ref p); AttnOutput    = p;
        p = FfnGate;       FreeIfNonZero(ref p); FfnGate       = p;
        p = FfnUp;         FreeIfNonZero(ref p); FfnUp         = p;
        p = SiluOutput;    FreeIfNonZero(ref p); SiluOutput    = p;
        p = GemmOutputF16; FreeIfNonZero(ref p); GemmOutputF16 = p;
        p = TokenIds;      FreeIfNonZero(ref p); TokenIds      = p;
        p = Positions;     FreeIfNonZero(ref p); Positions     = p;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        FreeSequenceBuffers();
        nint p;
        p = LogitsF16;      FreeIfNonZero(ref p); LogitsF16      = p;
        p = LogitsF32;      FreeIfNonZero(ref p); LogitsF32      = p;
        p = DequantScratch; FreeIfNonZero(ref p); DequantScratch = p;
        _currentSeqLen = 0;
    }
}
