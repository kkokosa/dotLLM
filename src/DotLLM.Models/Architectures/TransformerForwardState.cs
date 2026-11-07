using System.Numerics;
using System.Runtime.InteropServices;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Pre-allocated scratch buffers for the transformer forward pass. Reused across calls to
/// achieve zero per-call allocation on the hot path. Call <see cref="EnsureCapacity"/>
/// before each forward pass to resize if the sequence length has grown.
/// </summary>
internal sealed unsafe class TransformerForwardState : IDisposable
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly int _vocabSize;

    private int _currentSeqLen;
    private int _currentLogitRows;

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
            bytes += (long)_currentLogitRows * _vocabSize;   // Logits (1 row for prefill/decode, k+1 for speculative verify)
            bytes *= sizeof(float);
            // InputQ8Scratch: seqLen × max(Q8_0, Q8_1, Q8_K) row bytes
            int maxInputDim = Math.Max(_hiddenSize, _intermediateSize);
            int q8_0Bytes = (maxInputDim / 32) * 34;
            int q8_1Bytes = (maxInputDim / 32) * 36;
            int q8_kBytes = (maxInputDim / 256) * 292;
            bytes += s * Math.Max(Math.Max(q8_0Bytes, q8_1Bytes), q8_kBytes);
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

    /// <summary>
    /// Scratch buffer for pre-quantized input rows [seqLen × rowBytes].
    /// Sized for the largest of Q8_0 (34B/32el), Q8_1 (36B/32el), and Q8_K (292B/256el).
    /// Used to quantize the input once and reuse across Q/K/V and Gate/Up projections.
    /// </summary>
    public nint InputQ8Scratch;

    /// <summary>Pre-computed RoPE cosine table [maxSeqLen * halfDim].</summary>
    public float[] CosTable { get; }

    /// <summary>Pre-computed RoPE sine table [maxSeqLen * halfDim].</summary>
    public float[] SinTable { get; }

    public TransformerForwardState(
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
        _currentLogitRows = 0;
        EnsureCapacity(1, 1);
    }

    /// <summary>
    /// Ensures all scratch buffers are large enough for <paramref name="seqLen"/> tokens
    /// and <paramref name="logitRows"/> output logit positions.
    /// Uses power-of-2 growth to amortize reallocation cost.
    /// </summary>
    public void EnsureCapacity(int seqLen, int logitRows = -1)
    {
        if (logitRows < 0)
            logitRows = seqLen;

        if (seqLen > _currentSeqLen)
        {
            int newCapacity = (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);
            FreeActivationBuffers();

            HiddenState = AllocFloats((long)newCapacity * _hiddenSize);
            Residual = AllocFloats((long)newCapacity * _hiddenSize);
            NormOutput = AllocFloats((long)newCapacity * _hiddenSize);
            Q = AllocFloats((long)newCapacity * _numHeads * _headDim);
            K = AllocFloats((long)newCapacity * _numKvHeads * _headDim);
            V = AllocFloats((long)newCapacity * _numKvHeads * _headDim);
            AttnOutput = AllocFloats((long)newCapacity * _numHeads * _headDim);
            FfnGate = AllocFloats((long)newCapacity * _intermediateSize);
            FfnUp = AllocFloats((long)newCapacity * _intermediateSize);
            SiluOutput = AllocFloats((long)newCapacity * _intermediateSize);

            // InputQ8Scratch: seqLen × max(q8_0RowBytes, q8_1RowBytes, q8_kRowBytes) for pre-quantized GEMM input reuse.
            // Q8_0: 34 bytes per 32-element block. Q8_1: 36 bytes per 32-element block. Q8_K: 292 bytes per 256-element block.
            int maxInputDim = Math.Max(_hiddenSize, _intermediateSize);
            int q8_0RowBytes = (maxInputDim / 32) * 34;
            int q8_1RowBytes = (maxInputDim / 32) * 36;
            int q8_kRowBytes = (maxInputDim / 256) * 292;
            int scratchRowBytes = Math.Max(Math.Max(q8_0RowBytes, q8_1RowBytes), q8_kRowBytes);
            InputQ8Scratch = AllocBytes((long)newCapacity * scratchRowBytes);

            _currentSeqLen = newCapacity;
        }

        if (logitRows > _currentLogitRows)
        {
            int newLogitCapacity = (int)BitOperations.RoundUpToPowerOf2((uint)logitRows);
            FreeIfNonZero(ref Logits);
            Logits = AllocFloats((long)newLogitCapacity * _vocabSize);
            _currentLogitRows = newLogitCapacity;
        }
    }

    private static nint AllocFloats(long count)
    {
        return (nint)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);
    }

    private static nint AllocBytes(long count)
    {
        return (nint)NativeMemory.AlignedAlloc((nuint)count, 64);
    }

    private void FreeActivationBuffers()
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
        FreeIfNonZero(ref InputQ8Scratch);
    }

    private void FreeBuffers()
    {
        FreeActivationBuffers();
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
        _currentLogitRows = 0;
    }
}
