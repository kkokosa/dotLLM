using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Repacks quantized weight matrices from row-major to R4 interleaved layout.
/// Groups of 4 consecutive rows have their blocks interleaved column-by-column,
/// so that 4-row SIMD kernels read sequentially instead of striding across rows.
/// This improves cache utilization, TLB locality, and hardware prefetch effectiveness.
/// </summary>
public static unsafe class WeightRepacking
{
    /// <summary>Number of rows interleaved per group. Matches existing 4-row kernel batching.</summary>
    public const int InterleaveFactor = 4;

    /// <summary>
    /// Holds a repacked weight matrix with R4 interleaved layout.
    /// Full groups of 4 rows have their blocks interleaved. Tail rows (M % 4) are appended in original order.
    /// </summary>
    internal readonly struct RepackedWeight : IDisposable
    {
        /// <summary>64-byte aligned pointer to repacked data.</summary>
        public readonly nint Ptr;

        /// <summary>Number of complete 4-row groups (M / 4).</summary>
        public readonly int FullGroupCount;

        /// <summary>Number of leftover rows (M % 4).</summary>
        public readonly int TailRows;

        /// <summary>Number of quant blocks per row (K / groupSize).</summary>
        public readonly int BlocksPerRow;

        /// <summary>Byte size of one quant block.</summary>
        public readonly int BlockBytes;

        /// <summary>Total allocated bytes.</summary>
        public readonly long AllocatedBytes;

        public RepackedWeight(nint ptr, int fullGroupCount, int tailRows,
                              int blocksPerRow, int blockBytes, long allocatedBytes)
        {
            Ptr = ptr;
            FullGroupCount = fullGroupCount;
            TailRows = tailRows;
            BlocksPerRow = blocksPerRow;
            BlockBytes = blockBytes;
            AllocatedBytes = allocatedBytes;
        }

        /// <summary>Byte size of one original row: blocksPerRow * blockBytes.</summary>
        public int RowBytes
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => BlocksPerRow * BlockBytes;
        }

        /// <summary>Pointer to the start of tail rows (after all interleaved groups).</summary>
        public byte* TailPtr
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (byte*)Ptr + (long)FullGroupCount * InterleaveFactor * RowBytes;
        }

        public void Dispose()
        {
            if (Ptr != 0)
                NativeMemory.AlignedFree((void*)Ptr);
        }
    }

    /// <summary>
    /// Returns block byte size and group size for a given quantization type.
    /// Returns (0, 0) for unsupported types (F32, F16).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static (int blockBytes, int groupSize) GetBlockInfo(QuantizationType qt) => qt switch
    {
        QuantizationType.Q8_0 => (34, 32),    // Q8_0BlockBytes, Q8_0GroupSize
        QuantizationType.Q5_0 => (22, 32),    // Q5_0BlockBytes, Q5_0GroupSize
        QuantizationType.Q4_K => (144, 256),  // Q4_K_BlockBytes, KQuantGroupSize
        QuantizationType.Q5_K => (176, 256),  // Q5_K_BlockBytes, KQuantGroupSize
        QuantizationType.Q6_K => (210, 256),  // Q6_K_BlockBytes, KQuantGroupSize
        _ => (0, 0),
    };

    /// <summary>
    /// Returns true if the given quant type supports R4 repacking.
    /// F32/F16 are not block-structured and are skipped.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsRepackable(QuantizationType qt) =>
        qt is QuantizationType.Q8_0 or QuantizationType.Q5_0
            or QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K;

    /// <summary>
    /// Repacks a [M, K] quantized weight matrix from row-major to R4 interleaved layout.
    /// For each group of 4 rows, blocks are stored column-by-column:
    /// [row0_blk0][row1_blk0][row2_blk0][row3_blk0][row0_blk1][row1_blk1]...
    /// Tail rows (M % 4) are appended in original row-major order after the interleaved data.
    /// </summary>
    /// <param name="sourcePtr">Pointer to original row-major quantized weights.</param>
    /// <param name="qt">Quantization type of the weights.</param>
    /// <param name="m">Number of rows (output dimension).</param>
    /// <param name="k">Number of columns / elements per row (input dimension).</param>
    /// <returns>A <see cref="RepackedWeight"/> with the interleaved layout. Caller owns disposal.</returns>
    internal static RepackedWeight RepackR4(nint sourcePtr, QuantizationType qt, int m, int k)
    {
        var (blockBytes, groupSize) = GetBlockInfo(qt);
        if (blockBytes == 0)
            throw new ArgumentException($"Quantization type {qt} does not support R4 repacking.", nameof(qt));

        if (k % groupSize != 0)
            throw new ArgumentException(
                $"k ({k}) must be a multiple of group size ({groupSize}) for {qt}.", nameof(k));

        int blocksPerRow = k / groupSize;
        int rowBytes = blocksPerRow * blockBytes;
        int fullGroups = m / InterleaveFactor;
        int tailRows = m % InterleaveFactor;

        // Total size = same as original (we're just rearranging blocks)
        long totalBytes = (long)m * rowBytes;
        nint destPtr = (nint)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);

        byte* src = (byte*)sourcePtr;
        byte* dst = (byte*)destPtr;

        // Interleave full groups: for each group of 4 rows, write blocks column-by-column
        for (int g = 0; g < fullGroups; g++)
        {
            int baseRow = g * InterleaveFactor;
            byte* groupDst = dst + (long)g * InterleaveFactor * rowBytes;

            for (int b = 0; b < blocksPerRow; b++)
            {
                byte* colDst = groupDst + (long)b * InterleaveFactor * blockBytes;

                for (int r = 0; r < InterleaveFactor; r++)
                {
                    byte* srcBlock = src + (long)(baseRow + r) * rowBytes + (long)b * blockBytes;
                    byte* dstBlock = colDst + (long)r * blockBytes;
                    Buffer.MemoryCopy(srcBlock, dstBlock, blockBytes, blockBytes);
                }
            }
        }

        // Copy tail rows as-is (row-major)
        if (tailRows > 0)
        {
            int tailStartRow = fullGroups * InterleaveFactor;
            byte* tailSrc = src + (long)tailStartRow * rowBytes;
            byte* tailDst = dst + (long)fullGroups * InterleaveFactor * rowBytes;
            long tailBytes = (long)tailRows * rowBytes;
            Buffer.MemoryCopy(tailSrc, tailDst, tailBytes, tailBytes);
        }

        return new RepackedWeight(destPtr, fullGroups, tailRows, blocksPerRow, blockBytes, totalBytes);
    }
}
