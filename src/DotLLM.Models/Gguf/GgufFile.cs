using System.IO.MemoryMappedFiles;
using System.Numerics;
using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Represents an opened GGUF file with parsed metadata, tensor descriptors, and memory-mapped tensor data.
/// Owns the memory-mapped file resources and must be disposed when no longer needed.
/// </summary>
public sealed unsafe class GgufFile : IDisposable
{
    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _accessor;
    private byte* _basePointer;
    private bool _disposed;

    /// <summary>Parsed GGUF header.</summary>
    public GgufHeader Header { get; }

    /// <summary>Typed metadata accessor.</summary>
    public GgufMetadata Metadata { get; }

    /// <summary>Ordered list of tensor descriptors as they appear in the file.</summary>
    public IReadOnlyList<GgufTensorDescriptor> Tensors { get; }

    /// <summary>Tensor descriptors indexed by name for fast lookup.</summary>
    public IReadOnlyDictionary<string, GgufTensorDescriptor> TensorsByName { get; }

    /// <summary>
    /// Pointer to the start of the tensor data section. Individual tensor data is at
    /// <c>DataBasePointer + tensor.DataOffset</c>.
    /// Returns <see cref="nint.Zero"/> if the file contains no tensors.
    /// </summary>
    public nint DataBasePointer { get; }

    /// <summary>Byte offset of the tensor data section from the start of the file.</summary>
    public long DataSectionOffset { get; }

    private GgufFile(
        GgufHeader header,
        GgufMetadata metadata,
        IReadOnlyList<GgufTensorDescriptor> tensors,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensorsByName,
        long dataSectionOffset,
        nint dataBasePointer,
        MemoryMappedFile? mmf,
        MemoryMappedViewAccessor? accessor,
        byte* basePointer)
    {
        Header = header;
        Metadata = metadata;
        Tensors = tensors;
        TensorsByName = tensorsByName;
        DataSectionOffset = dataSectionOffset;
        DataBasePointer = dataBasePointer;
        _mmf = mmf;
        _accessor = accessor;
        _basePointer = basePointer;
    }

    /// <summary>
    /// Opens and parses a GGUF file. The tensor data section is memory-mapped for zero-copy access.
    /// </summary>
    /// <param name="filePath">Path to the GGUF file.</param>
    /// <returns>A <see cref="GgufFile"/> instance. Caller owns disposal.</returns>
    /// <exception cref="FileNotFoundException">File does not exist.</exception>
    /// <exception cref="InvalidDataException">File is not a valid GGUF file.</exception>
    public static GgufFile Open(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"GGUF file not found: {filePath}", filePath);

        GgufHeader header;
        Dictionary<string, GgufMetadataValue> rawMetadata;
        List<GgufTensorDescriptor> tensors;
        long streamPositionAfterInfos;

        long fileLength;
        using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
        using (var reader = new BinaryReader(fs))
        {
            header = GgufReader.ReadHeader(reader);
            rawMetadata = GgufReader.ReadMetadata(reader, header);
            tensors = GgufReader.ReadTensorInfos(reader, header);
            streamPositionAfterInfos = fs.Position;
            fileLength = fs.Length;
        }

        var metadata = new GgufMetadata(rawMetadata);

        // Alignment: default 32, overridable via general.alignment.
        uint alignment = metadata.GetUInt32OrDefault("general.alignment", 32);
        if (alignment == 0 || !BitOperations.IsPow2(alignment))
            throw new InvalidDataException(
                $"GGUF alignment must be a power of 2, got {alignment}.");

        long dataSectionOffset = AlignUp(streamPositionAfterInfos, alignment);

        // Validate tensor data fits within the file.
        long dataSectionLength = fileLength - dataSectionOffset;
        foreach (var tensor in tensors)
        {
            long tensorBytes = tensor.QuantizationType.ComputeByteCount(tensor.Shape.ElementCount);
            long endOffset = (long)tensor.DataOffset + tensorBytes;
            if (endOffset > dataSectionLength)
                throw new InvalidDataException(
                    $"Tensor '{tensor.Name}' data extends beyond file boundary " +
                    $"(offset {tensor.DataOffset}, size {tensorBytes}, " +
                    $"data section size {dataSectionLength}).");
        }

        var tensorsByName = new Dictionary<string, GgufTensorDescriptor>(tensors.Count);
        foreach (var tensor in tensors)
            tensorsByName[tensor.Name] = tensor;

        // Memory-map the file for tensor data access.
        MemoryMappedFile? mmf = null;
        MemoryMappedViewAccessor? accessor = null;
        byte* basePointer = null;
        nint dataBasePointer = nint.Zero;

        if (header.TensorCount > 0)
        {
            try
            {
                mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
                accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
                accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePointer);
                dataBasePointer = (nint)(basePointer + accessor.PointerOffset + dataSectionOffset);
            }
            catch
            {
                if (basePointer != null)
                    accessor?.SafeMemoryMappedViewHandle.ReleasePointer();
                accessor?.Dispose();
                mmf?.Dispose();
                throw;
            }
        }

        return new GgufFile(
            header,
            metadata,
            tensors.AsReadOnly(),
            tensorsByName,
            dataSectionOffset,
            dataBasePointer,
            mmf,
            accessor,
            basePointer);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        if (_basePointer != null)
        {
            _accessor?.SafeMemoryMappedViewHandle.ReleasePointer();
            _basePointer = null;
        }

        _accessor?.Dispose();
        _accessor = null;

        _mmf?.Dispose();
        _mmf = null;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static long AlignUp(long value, uint alignment)
    {
        long mask = alignment - 1;
        return (value + mask) & ~mask;
    }
}
