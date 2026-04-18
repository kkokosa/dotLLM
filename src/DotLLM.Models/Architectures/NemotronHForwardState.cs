using System.Numerics;
using System.Runtime.InteropServices;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Pre-allocated, 64-byte-aligned scratch buffers for the Nemotron-H hybrid forward pass.
/// Sized for the widest of the three sub-layer kinds and grown in power-of-two steps by
/// <see cref="EnsureCapacity"/> to keep the hot path allocation-free.
/// </summary>
internal sealed unsafe class NemotronHForwardState : IDisposable
{
    private readonly int _hiddenSize;
    private readonly int _maxIntermediateSize;
    private readonly int _vocabSize;
    private readonly int _qElems;
    private readonly int _kvElems;
    private readonly int _inputScratchRowBytes;

    private readonly int _inputProjectionDim;
    private readonly int _convDim;
    private readonly int _dConv;
    private readonly int _dInner;
    private readonly int _nHead;
    private readonly int _bcDim; // n_group * d_state — per-token width of each of B and C.

    private int _currentSeqLen;

    public nint HiddenState;
    public nint Residual;
    public nint NormOutput;
    public nint FfnIntermediate;
    public nint Logits;
    public nint InputQ8Scratch;

    public nint QScratch;
    public nint KScratch;
    public nint VScratch;
    public nint AttnOutput;

    public nint Zxbcdt;
    public nint ConvInput;
    public nint XBC;
    public nint DtBuffer;
    public nint SsmX;   // [T, d_inner] — x split out of xBC, fed to Mamba2SelectiveScan and reused in D*x skip.
    public nint SsmB;   // [T, n_group, d_state]
    public nint SsmC;   // [T, n_group, d_state]
    public nint SsmY;

    public long AllocatedBytes
    {
        get
        {
            long s = _currentSeqLen;
            if (s == 0) return 0;
            long floats = 0;
            floats += s * _hiddenSize * 3;                // HiddenState, Residual, NormOutput
            floats += s * _maxIntermediateSize;            // FfnIntermediate
            floats += s * _vocabSize;                      // Logits
            floats += s * _qElems;                         // QScratch
            floats += s * _kvElems * 2;                    // KScratch, VScratch
            floats += s * _qElems;                         // AttnOutput
            floats += s * _inputProjectionDim;             // Zxbcdt
            floats += (_dConv - 1 + s) * _convDim;         // ConvInput
            floats += s * _convDim;                        // XBC
            floats += s * _nHead;                          // DtBuffer
            floats += s * _dInner;                         // SsmX
            floats += s * _bcDim * 2;                      // SsmB, SsmC
            floats += s * _dInner;                         // SsmY
            long bytes = floats * sizeof(float);
            bytes += s * _inputScratchRowBytes;            // InputQ8Scratch (byte-sized)
            return bytes;
        }
    }

    public NemotronHForwardState(
        int hiddenSize,
        int maxIntermediateSize,
        int vocabSize,
        int qElems,
        int kvElems,
        int inputProjectionDim,
        int convDim,
        int dConv,
        int dInner,
        int nHead,
        int nGroup,
        int dState)
    {
        _hiddenSize = hiddenSize;
        _maxIntermediateSize = maxIntermediateSize;
        _vocabSize = vocabSize;
        _qElems = qElems;
        _kvElems = kvElems;
        _inputProjectionDim = inputProjectionDim;
        _convDim = convDim;
        _dConv = dConv;
        _dInner = dInner;
        _nHead = nHead;
        _bcDim = nGroup * dState;

        int scratchBase = Math.Max(Math.Max(hiddenSize, maxIntermediateSize), dInner);
        int q8_0RowBytes = (scratchBase / 32) * 34;
        int q8_1RowBytes = (scratchBase / 32) * 36;
        int q8_kRowBytes = (scratchBase / 256) * 292;
        _inputScratchRowBytes = Math.Max(Math.Max(q8_0RowBytes, q8_1RowBytes), q8_kRowBytes);

        _currentSeqLen = 0;
        EnsureCapacity(1);
    }

    public void EnsureCapacity(int seqLen)
    {
        if (seqLen <= _currentSeqLen) return;

        int cap = (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeBuffers();

        HiddenState = AllocFloats((long)cap * _hiddenSize);
        Residual = AllocFloats((long)cap * _hiddenSize);
        NormOutput = AllocFloats((long)cap * _hiddenSize);
        FfnIntermediate = AllocFloats((long)cap * _maxIntermediateSize);
        Logits = AllocFloats((long)cap * _vocabSize);
        InputQ8Scratch = AllocBytes((long)cap * _inputScratchRowBytes);

        QScratch = AllocFloats((long)cap * _qElems);
        KScratch = AllocFloats((long)cap * _kvElems);
        VScratch = AllocFloats((long)cap * _kvElems);
        AttnOutput = AllocFloats((long)cap * _qElems);

        Zxbcdt = AllocFloats((long)cap * _inputProjectionDim);
        ConvInput = AllocFloats((long)(_dConv - 1 + cap) * _convDim);
        XBC = AllocFloats((long)cap * _convDim);
        DtBuffer = AllocFloats((long)cap * _nHead);
        SsmX = AllocFloats((long)cap * _dInner);
        SsmB = AllocFloats((long)cap * _bcDim);
        SsmC = AllocFloats((long)cap * _bcDim);
        SsmY = AllocFloats((long)cap * _dInner);

        _currentSeqLen = cap;
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
        FreeIfNonZero(ref QScratch);
        FreeIfNonZero(ref KScratch);
        FreeIfNonZero(ref VScratch);
        FreeIfNonZero(ref AttnOutput);
        FreeIfNonZero(ref Zxbcdt);
        FreeIfNonZero(ref ConvInput);
        FreeIfNonZero(ref XBC);
        FreeIfNonZero(ref DtBuffer);
        FreeIfNonZero(ref SsmX);
        FreeIfNonZero(ref SsmB);
        FreeIfNonZero(ref SsmC);
        FreeIfNonZero(ref SsmY);
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
