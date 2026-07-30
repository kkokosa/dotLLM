using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// Embedding lookup kernels accelerated via Metal GPU.
/// Direct translation of embedding_f32out.cu.
/// For each token t: output[t] = dequant(embedTable[tokenIds[t]]) as FP32.
/// </summary>
public static class EmbeddingLookup
{
    private const int s_q80BlockSize  = 32;
    private const int s_q80BlockBytes = 34; // 2-byte half scale + 32 int8 values

    /// <summary>
    /// Embedding lookup from a float32 table.
    /// </summary>
    /// <param name="ctx">The Metal context.</param>
    /// <param name="embedTable">Full embedding table, shape <c>[vocabSize × hiddenSize]</c>.</param>
    /// <param name="tokenIds">Token indices, length <c>seqLen</c>.</param>
    /// <param name="output">Output buffer, shape <c>[seqLen × hiddenSize]</c>.</param>
    /// <param name="vocabSize">Number of vocabulary entries.</param>
    /// <param name="hiddenSize">Embedding dimension.</param>
    /// <param name="seqLen">Number of tokens to look up.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void F32ToF32(
        MetalContext        ctx,
        ReadOnlySpan<float> embedTable,
        ReadOnlySpan<int>   tokenIds,
        Span<float>         output,
        int vocabSize, int hiddenSize, int seqLen)
    {
        Validate(embedTable.Length, vocabSize, hiddenSize, seqLen, sizeof(float), tokenIds, output);

        unsafe
        {
            fixed (float* pTable  = embedTable)
            fixed (int*   pIds    = tokenIds)
            fixed (float* pOut    = output)
            {
                int code = MetalNative.EmbeddingF32F32Out(
                    ctx.Handle, pTable, pIds, pOut, vocabSize, hiddenSize, seqLen);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal embedding_f32_f32out failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Embedding lookup from a float16 table — cast to float32 on the GPU.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void F16ToF32(
        MetalContext          ctx,
        ReadOnlySpan<Half>    embedTable,
        ReadOnlySpan<int>     tokenIds,
        Span<float>           output,
        int vocabSize, int hiddenSize, int seqLen)
    {
        Validate(embedTable.Length, vocabSize, hiddenSize, seqLen, sizeof(ushort), tokenIds, output);

        ReadOnlySpan<ushort> raw = MemoryMarshal.Cast<Half, ushort>(embedTable);

        unsafe
        {
            fixed (ushort* pTable = raw)
            fixed (int*    pIds   = tokenIds)
            fixed (float*  pOut   = output)
            {
                int code = MetalNative.EmbeddingF16F32Out(
                    ctx.Handle, pTable, pIds, pOut, vocabSize, hiddenSize, seqLen);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal embedding_f16_f32out failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Forward-pass overload: takes raw <see cref="nint"/> pointers and does not check buffer lengths.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void F16ToF32(
        MetalContext ctx,
        nint embedTable,
        nint tokenIds,
        nint output,
        int vocabSize,
        int hiddenSize,
        int seqLen)
    {
        int code = MetalNative.EmbeddingF16F32Out(
            ctx.Handle, (ushort*)embedTable, (int*)tokenIds, (float*)output, vocabSize, hiddenSize, seqLen);
        if (code != 0)
        {
            throw new InvalidOperationException($"Metal embedding_f16_f32out failed with code {code}.");
        }
    }

    /// <summary>
    /// Embedding lookup from a Q8_0 quantized table — dequantize to float32 on the GPU.
    /// Each block is 34 bytes: 2-byte half scale followed by 32 int8 weights.
    /// </summary>
    /// <param name="ctx">Metal context</param>
    /// <param name="embedTable">Raw Q8_0 bytes, length = <c>vocabSize × (hiddenSize/32) × 34</c>.</param>
    /// <param name="tokenIds">Token ids</param>
    /// <param name="output">Output float32 embeddings</param>
    /// <param name="vocabSize">Vocab size</param>
    /// <param name="hiddenSize">Hidden size</param>
    /// <param name="seqLen">Sequence length</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Q8_0ToF32(
        MetalContext       ctx,
        ReadOnlySpan<byte> embedTable,
        ReadOnlySpan<int>  tokenIds,
        Span<float>        output,
        int vocabSize, int hiddenSize, int seqLen)
    {
        if (hiddenSize % s_q80BlockSize != 0)
            throw new ArgumentException(
                $"hiddenSize ({hiddenSize}) must be a multiple of {s_q80BlockSize} for Q8_0.", nameof(hiddenSize));

        int blocksPerRow      = hiddenSize / s_q80BlockSize;
        int expectedTableLen  = vocabSize * blocksPerRow * s_q80BlockBytes;
        if (embedTable.Length != expectedTableLen)
            throw new ArgumentException(
                $"embedTable.Length ({embedTable.Length}) must equal vocabSize × blocksPerRow × {s_q80BlockBytes} ({expectedTableLen}).", nameof(embedTable));
        if (tokenIds.Length != seqLen)
            throw new ArgumentException(
                $"tokenIds.Length ({tokenIds.Length}) must equal seqLen ({seqLen}).", nameof(tokenIds));
        if (output.Length != seqLen * hiddenSize)
            throw new ArgumentException(
                $"output.Length ({output.Length}) must equal seqLen × hiddenSize ({seqLen * hiddenSize}).", nameof(output));

        unsafe
        {
            fixed (byte*  pTable = embedTable)
            fixed (int*   pIds   = tokenIds)
            fixed (float* pOut   = output)
            {
                int code = MetalNative.EmbeddingQ8_0F32Out(
                    ctx.Handle, pTable, pIds, pOut, vocabSize, hiddenSize, seqLen);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal embedding_q8_0_f32out failed with code {code}.");
            }
        }
    }

    private static void Validate(
        int tableLen, int vocabSize, int hiddenSize, int seqLen,
        int elemBytes, ReadOnlySpan<int> tokenIds, Span<float> output)
    {
        if (tableLen != vocabSize * hiddenSize)
            throw new ArgumentException(
                $"embedTable.Length ({tableLen}) must equal vocabSize × hiddenSize ({vocabSize * hiddenSize}).");
        if (tokenIds.Length != seqLen)
            throw new ArgumentException(
                $"tokenIds.Length ({tokenIds.Length}) must equal seqLen ({seqLen}).");
        if (output.Length != seqLen * hiddenSize)
            throw new ArgumentException(
                $"output.Length ({output.Length}) must equal seqLen × hiddenSize ({seqLen * hiddenSize}).");
    }

    /// <summary>
    /// Forward-pass entry point. Dispatches to the right native kernel based on the
    /// embedding table's storage format. Uses raw <see cref="nint"/> pointers because
    /// in the forward pass the inputs/outputs already live in
    /// <see cref="MetalForwardState"/> as native allocations, not managed spans.
    /// </summary>
    /// <param name="ctx">Metal context.</param>
    /// <param name="embedTable">Pointer to the start of the embedding table.</param>
    /// <param name="embedDtype">Storage format of the table (F32, F16, or Q8_0 today).</param>
    /// <param name="tokenIds">Pointer to <c>seqLen</c> int32 token IDs.</param>
    /// <param name="output">Pointer to the FP32 output buffer of size <c>seqLen × hiddenSize</c>.</param>
    /// <param name="seqLen">Number of tokens to look up.</param>
    /// <param name="hiddenSize">Embedding dimension.</param>
    /// <param name="vocabSize">Number of rows in the embedding table.</param>
    /// <exception cref="NotSupportedException">If the table format has no kernel yet (e.g. Q4_K, Q5_K).</exception>
    public static unsafe void Execute(
        MetalContext     ctx,
        nint             embedTable,
        QuantizationType embedDtype,
        nint             tokenIds,
        nint             output,
        int              seqLen,
        int              hiddenSize,
        int              vocabSize)
    {
        int code = embedDtype switch
        {
            QuantizationType.F32  => MetalNative.EmbeddingF32F32Out(
                ctx.Handle, (float*)embedTable,  (int*)tokenIds, (float*)output,
                vocabSize, hiddenSize, seqLen),

            QuantizationType.F16  => MetalNative.EmbeddingF16F32Out(
                ctx.Handle, (ushort*)embedTable, (int*)tokenIds, (float*)output,
                vocabSize, hiddenSize, seqLen),

            QuantizationType.Q8_0 => MetalNative.EmbeddingQ8_0F32Out(
                ctx.Handle, (byte*)embedTable,   (int*)tokenIds, (float*)output,
                vocabSize, hiddenSize, seqLen),

            _ => throw new NotSupportedException(
                $"Embedding type {embedDtype} not supported on Metal yet. " +
                $"Add a dedicated kernel or load the model with DequantToFp16Strategy."),
        };

        if (code != 0)
            throw new InvalidOperationException(
                $"Metal embedding_{embedDtype}_f32out failed with code {code}.");
    }

    /// <summary>
    /// Forward-pass entry point with FP16 output. Used by the FP16 forward path so
    /// the embedding result drops straight into the FP16 hidden-state buffer with
    /// no FP32 intermediate.
    /// </summary>
    public static unsafe void ExecuteF16Out(
        MetalContext     ctx,
        nint             embedTable,
        QuantizationType embedDtype,
        nint             tokenIds,
        nint             output,
        int              seqLen,
        int              hiddenSize,
        int              vocabSize)
    {
        int? code = embedDtype switch
        {
            QuantizationType.F32  => MetalNative.EmbeddingF32F16Out(
                ctx.Handle, (float*)embedTable,  (int*)tokenIds, (ushort*)output,
                vocabSize, hiddenSize, seqLen),

            QuantizationType.F16  => MetalNative.EmbeddingF16F16Out(
                ctx.Handle, (ushort*)embedTable, (int*)tokenIds, (ushort*)output,
                vocabSize, hiddenSize, seqLen),

            QuantizationType.Q8_0 => MetalNative.EmbeddingQ8_0F16Out(
                ctx.Handle, (byte*)embedTable,   (int*)tokenIds, (ushort*)output,
                vocabSize, hiddenSize, seqLen),

            QuantizationType.Q6_K => MetalNative.EmbeddingQ6_KF16Out(
                ctx.Handle, (byte*)embedTable,   (int*)tokenIds, (ushort*)output,
                vocabSize, hiddenSize, seqLen),

            _ => null, // CPU fallback for formats without a dedicated GPU kernel
        };

        if (code is null)
        {
            CpuFallbackF16Out(embedTable, embedDtype, tokenIds, output, seqLen, hiddenSize);
            return;
        }

        if (code != 0)
            throw new InvalidOperationException(
                $"Metal embedding_{embedDtype}_f16out failed with code {code}.");
    }

    // Dequantizes seqLen rows on the CPU into the shared-memory FP16 output buffer.
    // Used for quantization formats that have no dedicated embedding GPU kernel yet
    // (e.g. Q4_K, Q5_K, Q6_K). On Apple Silicon the output pointer is shared memory
    // so the GPU will read the written FP16 values directly.
    private static unsafe void CpuFallbackF16Out(
        nint embedTable, QuantizationType embedDtype,
        nint tokenIds, nint output,
        int seqLen, int hiddenSize)
    {
        long rowBytes = Dequantize.RowByteSize(hiddenSize, embedDtype);
        float[] scratch = ArrayPool<float>.Shared.Rent(hiddenSize);
        try
        {
            var scratch32 = scratch.AsSpan(0, hiddenSize);
            int* ids = (int*)tokenIds;
            for (int t = 0; t < seqLen; t++)
            {
                nint rowSrc = embedTable + (nint)(ids[t] * rowBytes);
                Dequantize.ToFloat32(rowSrc, hiddenSize, embedDtype, scratch32);

                var dst16 = new Span<Half>((Half*)output + (nint)(t * hiddenSize), hiddenSize);
                TensorPrimitives.ConvertToHalf(scratch32, dst16);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scratch);
        }
    }
}
