namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Describes a single tensor entry in a safetensors file: name, dtype, shape,
/// and the byte range in the raw data section.
/// </summary>
/// <remarks>
/// <para>
/// Per the
/// <see href="https://github.com/huggingface/safetensors">safetensors spec</see>,
/// the JSON header contains one entry per tensor with a
/// <c>"data_offsets": [begin, end]</c> pair. Both offsets are relative to
/// the start of the <i>data region</i> — which begins at byte offset
/// <c>8 + header_len</c> in the file (the 8-byte little-endian u64 length
/// prefix plus the UTF-8 JSON header itself).
/// </para>
/// <para>
/// This descriptor preserves the spec-relative offsets verbatim — the file
/// reader resolves them to an absolute pointer by adding
/// <see cref="SafetensorsFile.DataBasePointer"/>.
/// </para>
/// </remarks>
/// <param name="Name">Tensor name (e.g. <c>backbone.embeddings.weight</c>).</param>
/// <param name="DType">Storage dtype as parsed from the header.</param>
/// <param name="Shape">Row-major dimensions, in declaration order.</param>
/// <param name="DataBeginOffset">
/// Byte offset of the first byte of this tensor, relative to the start of
/// the data region.
/// </param>
/// <param name="DataEndOffset">
/// Byte offset one past the last byte of this tensor, relative to the
/// start of the data region. <c>DataEndOffset - DataBeginOffset</c> must
/// equal <c>element_count * dtype_size</c>.
/// </param>
public readonly record struct SafetensorsTensorDescriptor(
    string Name,
    SafetensorsDType DType,
    int[] Shape,
    long DataBeginOffset,
    long DataEndOffset)
{
    /// <summary>Size, in bytes, of this tensor's raw data payload.</summary>
    public long ByteCount => DataEndOffset - DataBeginOffset;

    /// <summary>
    /// Total element count = product of all shape dimensions. Returns
    /// <c>1</c> for a scalar (rank-0) tensor.
    /// </summary>
    public long ElementCount
    {
        get
        {
            long n = 1;
            for (int i = 0; i < Shape.Length; i++) n *= Shape[i];
            return n;
        }
    }
}
