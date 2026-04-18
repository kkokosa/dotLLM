using System.Numerics;
using System.Runtime.InteropServices;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Pre-allocated scratch buffers for the Nemotron-H hybrid forward pass. Mirrors
/// <see cref="TransformerForwardState"/> but with only the subset of buffers the
/// hybrid dispatch needs: a single sub-layer output per block (no parallel
/// attention + FFN), plus an FFN-intermediate scratch sized to the largest
/// FFN layer. Resized in power-of-two steps by <see cref="EnsureCapacity"/>.
/// </summary>
internal sealed unsafe class NemotronHForwardState : IDisposable
{
    private readonly int _hiddenSize;
    private readonly int _maxIntermediateSize;
    private readonly int _vocabSize;
    private readonly int _inputScratchRowBytes;

    private int _currentSeqLen;

    public nint HiddenState;
    public nint Residual;
    public nint NormOutput;
    public nint FfnIntermediate;
    public nint Logits;
    public nint InputQ8Scratch;

    public long AllocatedBytes
    {
        get
        {
            long s = _currentSeqLen;
            if (s == 0) return 0;
            long bytes = 0;
            bytes += s * _hiddenSize * 3;
            bytes += s * _maxIntermediateSize;
            bytes += s * _vocabSize;
            bytes *= sizeof(float);
            bytes += s * _inputScratchRowBytes;
            return bytes;
        }
    }

    public NemotronHForwardState(int hiddenSize, int maxIntermediateSize, int vocabSize)
    {
        _hiddenSize = hiddenSize;
        _maxIntermediateSize = maxIntermediateSize;
        _vocabSize = vocabSize;

        int scratchBase = Math.Max(hiddenSize, maxIntermediateSize);
        int q8_0RowBytes = (scratchBase / 32) * 34;
        int q8_1RowBytes = (scratchBase / 32) * 36;
        int q8_kRowBytes = (scratchBase / 256) * 292;
        _inputScratchRowBytes = Math.Max(Math.Max(q8_0RowBytes, q8_1RowBytes), q8_kRowBytes);

        _currentSeqLen = 0;
        EnsureCapacity(1);
    }

    public void EnsureCapacity(int seqLen)
    {
        if (seqLen <= _currentSeqLen)
            return;

        int newCapacity = (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeBuffers();

        HiddenState = AllocFloats((long)newCapacity * _hiddenSize);
        Residual = AllocFloats((long)newCapacity * _hiddenSize);
        NormOutput = AllocFloats((long)newCapacity * _hiddenSize);
        FfnIntermediate = AllocFloats((long)newCapacity * _maxIntermediateSize);
        Logits = AllocFloats((long)newCapacity * _vocabSize);
        InputQ8Scratch = AllocBytes((long)newCapacity * _inputScratchRowBytes);

        _currentSeqLen = newCapacity;
    }

    private static nint AllocFloats(long count)
        => (nint)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);

    private static nint AllocBytes(long count)
        => (nint)NativeMemory.AlignedAlloc((nuint)count, 64);

    private void FreeBuffers()
    {
        FreeIfNonZero(ref HiddenState);
        FreeIfNonZero(ref Residual);
        FreeIfNonZero(ref NormOutput);
        FreeIfNonZero(ref FfnIntermediate);
        FreeIfNonZero(ref Logits);
        FreeIfNonZero(ref InputQ8Scratch);
    }

    private static void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0)
        {
            NativeMemory.AlignedFree((void*)ptr);
            ptr = 0;
        }
    }

    public void Dispose()
    {
        FreeBuffers();
        _currentSeqLen = 0;
    }
}
