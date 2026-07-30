using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Metal.Interop;
using DotLLM.Metal.Kernels;

namespace DotLLM.Metal.Weights.Strategies;

/// <summary>
/// Loads every quantized tensor by dequantizing once into a freshly-allocated
/// FP16 buffer, then drops the reference to the original mmap bytes (the OS
/// page cache will reclaim them on memory pressure).
///
/// One code path through the forward pass: every projection becomes a plain
/// FP16 GEMM via MPS. Best when you want to validate end-to-end correctness
/// before adding the quantized fast paths.
/// </summary>
public sealed class DequantToFp16Strategy : IWeightLoadStrategy
{
    // Geometry constants — must match Dequant.cs and the .metal kernels.
    private const int BlockElems       = 32;    // Q8_0, Q4_0, Q5_0
    private const int SuperblockElems  = 256;   // Q4_K, Q5_K, Q6_K

    /// <inheritdoc/>
    public string Name => "DequantToFp16";

    /// <inheritdoc/>
    /// <remarks>
    /// FP16 destination = 2 bytes per element, regardless of the source format.
    /// The original quantized bytes live in the OS-mapped GGUF file and are
    /// not counted (no extra allocation, just shared page cache).
    /// </remarks>
    public long EstimateBytesFor(int totalElements, QuantizationType sourceFormat)
        => (long)totalElements * sizeof(ushort);

    /// <inheritdoc/>
    public LoadedWeight Load(MetalContext ctx, in TensorSource src)
    {
        int totalElements = src.OutputDim * src.InputDim;

        // 1. Allocate the FP16 destination buffer (64-byte aligned per project convention).
        nint dst = AllocFp16Aligned(ctx, totalElements);

        // 2. Wrap the raw mmap range and the destination as Spans.
        //    The Dequant API takes Span<byte> / Span<Half> — we cannot pass
        //    raw nint pointers directly.
        unsafe
        {
            var srcSpan = new ReadOnlySpan<byte>((void*)src.MmapPointer, (int)src.ByteLength);
            var dstSpan = new Span<Half>((void*)dst, totalElements);

            // 3. Dispatch on the source format. Each kernel knows its own
            //    geometry (block vs superblock, payload size).
            switch (src.Format)
            {
                // ─── 32-element blocks ──────────────────────────────────
                case QuantizationType.Q8_0:
                    Dequant.Q8_0ToF16(ctx, srcSpan, dstSpan, totalElements / BlockElems);
                    break;
                case QuantizationType.Q4_0:
                    Dequant.Q4_0ToF16(ctx, srcSpan, dstSpan, totalElements / BlockElems);
                    break;
                case QuantizationType.Q5_0:
                    Dequant.Q5_0ToF16(ctx, srcSpan, dstSpan, totalElements / BlockElems);
                    break;

                // ─── 256-element superblocks (K-quants) ─────────────────
                case QuantizationType.Q4_K:
                    Dequant.Q4_KToF16(ctx, srcSpan, dstSpan, totalElements / SuperblockElems);
                    break;
                case QuantizationType.Q5_K:
                    Dequant.Q5_KToF16(ctx, srcSpan, dstSpan, totalElements / SuperblockElems);
                    break;
                case QuantizationType.Q6_K:
                    Dequant.Q6_KToF16(ctx, srcSpan, dstSpan, totalElements / SuperblockElems);
                    break;

                // ─── Already FP16: byte-copy, no kernel needed ──────────
                case QuantizationType.F16:
                    srcSpan.CopyTo(MemoryMarshal.Cast<Half, byte>(dstSpan));
                    break;

                // ─── FP32: not yet handled (would need Convert.F32ToF16) ─
                case QuantizationType.F32:
                    throw new NotImplementedException(
                        $"{Name}: FP32 → FP16 conversion at load time not yet implemented.");

                default:
                    throw new NotSupportedException(
                        $"{Name}: unsupported source format {src.Format}.");
            }
        }

        // 4. Return the LoadedWeight pointing at the new FP16 buffer.
        //    QuantizedPointer is 0 because we deliberately drop the quantized
        //    representation — only the FP16 copy lives on.
        return new LoadedWeight(
            QuantizedPointer: 0,
            QuantizedFormat:  default,
            Fp16Pointer:      dst,
            OwnsFp16Buffer:   true,
            OutputDim:        src.OutputDim,
            InputDim:         src.InputDim);
    }

    /// <summary>
    /// Allocates an FP16 buffer as a page-aligned MTLBuffer registered in the
    /// context's shared-buffer map. Kernels can recover the MTLBuffer by
    /// pointer for zero-copy encoding (no CPU↔GPU copy at dispatch time).
    /// Caller must free via <see cref="MetalNative.FreeShared"/>.
    /// </summary>
    private static IntPtr AllocFp16Aligned(MetalContext ctx, int count)
    {
        nuint bytes = (nuint)count * sizeof(ushort);
        return MetalNative.AllocShared(ctx.Handle, bytes);
    }
}
