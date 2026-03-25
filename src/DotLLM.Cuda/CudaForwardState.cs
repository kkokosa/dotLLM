using System.Numerics;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Pre-allocated GPU scratch buffers for the CUDA forward pass. Activation buffers
/// are FP16, logits output is FP32. Mirrors <c>TransformerForwardState</c>
/// but on GPU memory allocated via <c>cuMemAlloc_v2</c>.
/// </summary>
internal sealed class CudaForwardState : IDisposable
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly int _vocabSize;

    private int _currentSeqLen;

    /// <summary>Total bytes currently allocated on GPU.</summary>
    public long AllocatedBytes { get; private set; }

    // All activation pointers are cuMemAlloc'd device memory, FP16 (sizeof(ushort) = 2 bytes)
    public nint HiddenState;    // [seqLen, hiddenSize]
    public nint Residual;       // [seqLen, hiddenSize]
    public nint NormOutput;     // [seqLen, hiddenSize]
    public nint Q;              // [seqLen, numHeads * headDim]
    public nint K;              // [seqLen, numKvHeads * headDim]
    public nint V;              // [seqLen, numKvHeads * headDim]
    public nint AttnOutput;     // [seqLen, numHeads * headDim]
    public nint FfnGate;        // [seqLen, intermediateSize]
    public nint FfnUp;          // [seqLen, intermediateSize]
    public nint SiluOutput;     // [seqLen, intermediateSize]

    // Logits — FP16 on device, then converted to FP32
    public nint LogitsF16;      // [vocabSize] FP16
    public nint LogitsF32;      // [vocabSize] FP32

    // General-purpose FP16 scratch buffer
    public nint GemmOutputF16;

    // On-the-fly dequantization scratch: holds one projection's FP16 weights
    // for cuBLAS GEMM. Sized for the largest projection (max of Gate/Up/Down/Q/O).
    // Reused across all cuBLAS calls — safe because all ops are on the same stream.
    public nint DequantScratch;

    // Small device buffers for H2D copy of token IDs and positions
    public nint TokenIdsDevice; // [maxSeqLen] int32
    public nint PositionsDevice;// [maxSeqLen] int32

    public CudaForwardState(int hiddenSize, int numHeads, int numKvHeads, int headDim,
                              int intermediateSize, int vocabSize)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _intermediateSize = intermediateSize;
        _vocabSize = vocabSize;
        _currentSeqLen = 0;

        // Logits are fixed-size (only last token)
        LogitsF16 = AllocDevice((long)vocabSize * sizeof(ushort));
        LogitsF32 = AllocDevice((long)vocabSize * sizeof(float));

        // Dequant scratch: sized for the largest per-layer projection in FP16.
        // Used for on-the-fly dequantization of quantized weights before cuBLAS GEMM.
        long maxProjectionElements = Math.Max(
            (long)Math.Max(numHeads * headDim, numKvHeads * headDim) * hiddenSize,
            (long)intermediateSize * hiddenSize);
        DequantScratch = AllocDevice(maxProjectionElements * sizeof(ushort));

        // Initial allocation for decode (seqLen=1)
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
        FreeSequenceBuffers();

        int half = sizeof(ushort); // FP16 = 2 bytes

        // All activation buffers are FP16 — per GPU.md spec for memory-bandwidth-optimal inference.
        // Only LogitsF32 (output to host) stays FP32.
        HiddenState = AllocDevice((long)newCapacity * _hiddenSize * half);
        Residual = AllocDevice((long)newCapacity * _hiddenSize * half);
        NormOutput = AllocDevice((long)newCapacity * _hiddenSize * half);
        Q = AllocDevice((long)newCapacity * _numHeads * _headDim * half);
        K = AllocDevice((long)newCapacity * _numKvHeads * _headDim * half);
        V = AllocDevice((long)newCapacity * _numKvHeads * _headDim * half);
        AttnOutput = AllocDevice((long)newCapacity * _numHeads * _headDim * half);
        FfnGate = AllocDevice((long)newCapacity * _intermediateSize * half);
        FfnUp = AllocDevice((long)newCapacity * _intermediateSize * half);
        SiluOutput = AllocDevice((long)newCapacity * _intermediateSize * half);
        // General scratch: must fit largest projection output or LM head logits
        long maxPerLayer = (long)newCapacity * Math.Max(Math.Max(_numHeads * _headDim, _intermediateSize), _hiddenSize);
        long maxLmHead = _vocabSize;
        GemmOutputF16 = AllocDevice(Math.Max(maxPerLayer, maxLmHead) * half);
        TokenIdsDevice = AllocDevice((long)newCapacity * sizeof(int));
        PositionsDevice = AllocDevice((long)newCapacity * sizeof(int));

        _currentSeqLen = newCapacity;
    }

    private static nint AllocDevice(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)bytes).ThrowOnError();
        return ptr;
    }

    private void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0)
        {
            CudaDriverApi.cuMemFree_v2(ptr);
            ptr = 0;
        }
    }

    private void FreeSequenceBuffers()
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
        FreeIfNonZero(ref GemmOutputF16);
        FreeIfNonZero(ref TokenIdsDevice);
        FreeIfNonZero(ref PositionsDevice);
    }

    public void Dispose()
    {
        FreeSequenceBuffers();
        FreeIfNonZero(ref LogitsF16);
        FreeIfNonZero(ref LogitsF32);
        FreeIfNonZero(ref DequantScratch);
        _currentSeqLen = 0;
    }
}
