using System.Numerics;
using System.Runtime.InteropServices;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Pre-allocated scratch buffers for the Llama forward pass. Reused across calls to
/// achieve zero per-call allocation on the hot path. Call <see cref="EnsureCapacity"/>
/// before each forward pass to resize if the sequence length has grown.
/// </summary>
internal sealed unsafe class LlamaForwardState : IDisposable
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly int _vocabSize;

    private int _currentSeqLen;

    /// <summary>Total bytes currently allocated for inference scratch buffers.</summary>
    public long AllocatedBytes
    {
        get
        {
            long s = _currentSeqLen;
            if (s == 0) return 0;

            long bytes = 0;
            bytes += s * _hiddenSize * 3;                    // HiddenState + Residual + NormOutput
            bytes += s * _numHeads * _headDim * 2;           // Q + AttnOutput
            bytes += s * _numKvHeads * _headDim * 2;         // K + V
            bytes += s * _intermediateSize * 3;              // FfnGate + FfnUp + SiluOutput
            bytes += _vocabSize;                             // Logits (only last token)
            bytes *= sizeof(float);
            // RoPE tables (managed, but still part of compute memory)
            bytes += (CosTable.Length + SinTable.Length) * sizeof(float);
            return bytes;
        }
    }

    // All pointers are 64-byte-aligned via NativeMemory.AlignedAlloc.
    public nint HiddenState;
    public nint Residual;
    public nint NormOutput;
    public nint Q;
    public nint K;
    public nint V;
    public nint AttnOutput;
    public nint FfnGate;
    public nint FfnUp;
    public nint SiluOutput;
    public nint Logits;

    /// <summary>Pre-computed RoPE cosine table [maxSeqLen * halfDim].</summary>
    public float[] CosTable { get; }

    /// <summary>Pre-computed RoPE sine table [maxSeqLen * halfDim].</summary>
    public float[] SinTable { get; }

    public LlamaForwardState(
        int hiddenSize, int numHeads, int numKvHeads, int headDim,
        int intermediateSize, int vocabSize, int maxSeqLen, int ropeDim,
        float ropeTheta)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _intermediateSize = intermediateSize;
        _vocabSize = vocabSize;

        // Pre-compute RoPE frequency tables
        int halfDim = ropeDim / 2;
        CosTable = new float[maxSeqLen * halfDim];
        SinTable = new float[maxSeqLen * halfDim];
        DotLLM.Cpu.Kernels.RoPE.PrecomputeFrequencyTable(maxSeqLen, ropeDim, ropeTheta, CosTable, SinTable);

        // Initial allocation for 1 token (decode mode)
        _currentSeqLen = 0;
        EnsureCapacity(1);
    }

    /// <summary>
    /// Ensures all scratch buffers are large enough for <paramref name="seqLen"/> tokens.
    /// Uses power-of-2 growth to amortize reallocation cost.
    /// </summary>
    public void EnsureCapacity(int seqLen)
    {
        if (seqLen <= _currentSeqLen)
            return;

        int newCapacity = (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeBuffers();

        HiddenState = AllocFloats(newCapacity * _hiddenSize);
        Residual = AllocFloats(newCapacity * _hiddenSize);
        NormOutput = AllocFloats(newCapacity * _hiddenSize);
        Q = AllocFloats(newCapacity * _numHeads * _headDim);
        K = AllocFloats(newCapacity * _numKvHeads * _headDim);
        V = AllocFloats(newCapacity * _numKvHeads * _headDim);
        AttnOutput = AllocFloats(newCapacity * _numHeads * _headDim);
        FfnGate = AllocFloats(newCapacity * _intermediateSize);
        FfnUp = AllocFloats(newCapacity * _intermediateSize);
        SiluOutput = AllocFloats(newCapacity * _intermediateSize);
        Logits = AllocFloats(_vocabSize); // Only last token's logits needed

        _currentSeqLen = newCapacity;
    }

    private static nint AllocFloats(long count)
    {
        return (nint)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);
    }

    private void FreeBuffers()
    {
        FreeIfNonZero(ref HiddenState);
        FreeIfNonZero(ref Residual);
        FreeIfNonZero(ref NormOutput);
        FreeIfNonZero(ref Q);
        FreeIfNonZero(ref K);
        FreeIfNonZero(ref V);
        FreeIfNonZero(ref AttnOutput);
        FreeIfNonZero(ref FfnGate);
        FreeIfNonZero(ref FfnUp);
        FreeIfNonZero(ref SiluOutput);
        FreeIfNonZero(ref Logits);
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
