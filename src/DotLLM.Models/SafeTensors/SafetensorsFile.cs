using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using System.Text.Json;

namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Represents an opened safetensors file: parsed header plus a
/// memory-mapped view of the raw tensor data region. Owns the mmap
/// resources and must be disposed.
/// </summary>
/// <remarks>
/// <para>
/// Safetensors file layout (HuggingFace canonical format):
/// </para>
/// <list type="number">
///   <item>Bytes <c>[0, 8)</c>: little-endian u64 <c>header_len</c>.</item>
///   <item>Bytes <c>[8, 8 + header_len)</c>: UTF-8 JSON header.</item>
///   <item>Bytes <c>[8 + header_len, file_end)</c>: raw tensor data,
///   row-major, concatenated back-to-back, per-tensor ranges declared in
///   the header's <c>"data_offsets"</c> arrays (relative to the start of
///   this region).</item>
/// </list>
/// <para>
/// The JSON header is a top-level object whose keys are tensor names, each
/// mapping to <c>{"dtype": "F32", "shape": [...], "data_offsets": [a, b]}</c>.
/// An optional <c>"__metadata__"</c> key carries free-form metadata and is
/// filtered out of <see cref="Tensors"/>.
/// </para>
/// <para>
/// Consistent with <see cref="DotLLM.Models.Gguf.GgufFile"/>: the whole
/// file (not just the data region) is memory-mapped read-only, and tensor
/// pointers are derived by adding <c>DataBasePointer + descriptor.DataBeginOffset</c>.
/// No managed copies of tensor data are made at open time.
/// </para>
/// </remarks>
public sealed unsafe class SafetensorsFile : IDisposable
{
    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _accessor;
    private byte* _basePointer;
    private bool _disposed;

    /// <summary>Byte length of the JSON header, read from the 8-byte prefix.</summary>
    public long HeaderLength { get; }

    /// <summary>
    /// Byte offset from the start of the file to the first byte of the
    /// tensor data region (<c>= 8 + HeaderLength</c>).
    /// </summary>
    public long DataSectionOffset => 8 + HeaderLength;

    /// <summary>Total size of the mapped file in bytes.</summary>
    public long FileLength { get; }

    /// <summary>
    /// Optional free-form metadata from the <c>"__metadata__"</c> header key.
    /// Empty dictionary if absent.
    /// </summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    /// <summary>Tensor descriptors in the order they appear in the header JSON.</summary>
    public IReadOnlyList<SafetensorsTensorDescriptor> Tensors { get; }

    /// <summary>Tensor descriptors indexed by name for O(1) lookup.</summary>
    public IReadOnlyDictionary<string, SafetensorsTensorDescriptor> TensorsByName { get; }

    /// <summary>
    /// Pointer to the first byte of the data region. Individual tensor data
    /// is at <c>DataBasePointer + descriptor.DataBeginOffset</c>. Returns
    /// <see cref="nint.Zero"/> if the file declares no tensors.
    /// </summary>
    public nint DataBasePointer { get; }

    private SafetensorsFile(
        long headerLength,
        long fileLength,
        IReadOnlyDictionary<string, string> metadata,
        IReadOnlyList<SafetensorsTensorDescriptor> tensors,
        IReadOnlyDictionary<string, SafetensorsTensorDescriptor> tensorsByName,
        nint dataBasePointer,
        MemoryMappedFile? mmf,
        MemoryMappedViewAccessor? accessor,
        byte* basePointer)
    {
        HeaderLength = headerLength;
        FileLength = fileLength;
        Metadata = metadata;
        Tensors = tensors;
        TensorsByName = tensorsByName;
        DataBasePointer = dataBasePointer;
        _mmf = mmf;
        _accessor = accessor;
        _basePointer = basePointer;
    }

    /// <summary>
    /// Opens a safetensors file, parses its JSON header, and memory-maps
    /// the tensor data region for zero-copy access.
    /// </summary>
    /// <param name="filePath">Absolute path to a <c>*.safetensors</c> file.</param>
    /// <returns>
    /// An opened <see cref="SafetensorsFile"/>. Caller owns disposal.
    /// </returns>
    /// <exception cref="FileNotFoundException">File does not exist.</exception>
    /// <exception cref="InvalidDataException">Header malformed or data ranges inconsistent.</exception>
    public static SafetensorsFile Open(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Safetensors file not found: {filePath}", filePath);

        long headerLen;
        byte[] headerJsonBytes;
        long fileLength;

        using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            fileLength = fs.Length;
            if (fileLength < 8)
                throw new InvalidDataException(
                    $"Safetensors file '{filePath}' is too small ({fileLength} bytes) to contain an 8-byte header length prefix.");

            Span<byte> lenBuf = stackalloc byte[8];
            int read = fs.Read(lenBuf);
            if (read != 8)
                throw new InvalidDataException(
                    $"Safetensors file '{filePath}': could not read 8-byte header length prefix (read {read}).");

            ulong raw = BinaryPrimitives.ReadUInt64LittleEndian(lenBuf);
            if (raw > (ulong)long.MaxValue)
                throw new InvalidDataException(
                    $"Safetensors header length prefix {raw} exceeds Int64.MaxValue.");
            headerLen = (long)raw;

            if (headerLen < 2)
                throw new InvalidDataException(
                    $"Safetensors header length {headerLen} is implausibly small (must contain at least '{{}}').");
            if (8 + headerLen > fileLength)
                throw new InvalidDataException(
                    $"Safetensors header length {headerLen} exceeds file length {fileLength} (header would read past EOF).");

            headerJsonBytes = new byte[headerLen];
            int headerRead = 0;
            while (headerRead < headerLen)
            {
                int n = fs.Read(headerJsonBytes, headerRead, (int)(headerLen - headerRead));
                if (n <= 0)
                    throw new InvalidDataException(
                        $"Safetensors file '{filePath}': unexpected EOF while reading header (got {headerRead} of {headerLen} bytes).");
                headerRead += n;
            }
        }

        long dataSectionLength = fileLength - 8 - headerLen;
        var (metadata, tensors) = ParseHeader(headerJsonBytes, dataSectionLength);

        var byName = new Dictionary<string, SafetensorsTensorDescriptor>(tensors.Count, StringComparer.Ordinal);
        foreach (var t in tensors)
        {
            if (!byName.TryAdd(t.Name, t))
                throw new InvalidDataException(
                    $"Safetensors header contains duplicate tensor name '{t.Name}'.");
        }

        // Memory-map read-only and anchor the pointer for the data region.
        MemoryMappedFile? mmf = null;
        MemoryMappedViewAccessor? accessor = null;
        byte* basePointer = null;
        nint dataBasePointer = nint.Zero;

        if (tensors.Count > 0)
        {
            try
            {
                mmf = MemoryMappedFile.CreateFromFile(
                    filePath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
                accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
                accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePointer);
                dataBasePointer = (nint)(basePointer + accessor.PointerOffset + 8 + headerLen);
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

        return new SafetensorsFile(
            headerLen,
            fileLength,
            metadata,
            tensors,
            byName,
            dataBasePointer,
            mmf,
            accessor,
            basePointer);
    }

    /// <summary>
    /// Parses the safetensors header JSON into metadata + descriptor list.
    /// Made internal so the loader-level tests can round-trip a synthesized
    /// header without a real file.
    /// </summary>
    internal static (IReadOnlyDictionary<string, string> Metadata,
                     IReadOnlyList<SafetensorsTensorDescriptor> Tensors)
        ParseHeader(byte[] headerJsonBytes, long dataSectionLength)
    {
        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(headerJsonBytes);
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException(
                $"Safetensors header is not valid JSON: {ex.Message}", ex);
        }

        using (doc)
        {
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException(
                    "Safetensors header JSON root must be an object.");

            var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
            var tensors = new List<SafetensorsTensorDescriptor>();

            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                if (prop.NameEquals("__metadata__"))
                {
                    if (prop.Value.ValueKind == JsonValueKind.Object)
                    {
                        foreach (var m in prop.Value.EnumerateObject())
                        {
                            if (m.Value.ValueKind == JsonValueKind.String)
                                metadata[m.Name] = m.Value.GetString() ?? string.Empty;
                        }
                    }
                    continue;
                }

                string name = prop.Name;
                if (prop.Value.ValueKind != JsonValueKind.Object)
                    throw new InvalidDataException(
                        $"Safetensors header entry '{name}' must be a JSON object.");

                if (!prop.Value.TryGetProperty("dtype", out var dtypeEl) ||
                    dtypeEl.ValueKind != JsonValueKind.String)
                    throw new InvalidDataException(
                        $"Safetensors tensor '{name}' is missing a string 'dtype'.");
                var dtype = SafetensorsDTypeExtensions.Parse(dtypeEl.GetString()!);

                if (!prop.Value.TryGetProperty("shape", out var shapeEl) ||
                    shapeEl.ValueKind != JsonValueKind.Array)
                    throw new InvalidDataException(
                        $"Safetensors tensor '{name}' is missing an array 'shape'.");
                int rank = shapeEl.GetArrayLength();
                int[] shape = new int[rank];
                int axis = 0;
                foreach (var dim in shapeEl.EnumerateArray())
                {
                    if (dim.ValueKind != JsonValueKind.Number || !dim.TryGetInt32(out int d) || d < 0)
                        throw new InvalidDataException(
                            $"Safetensors tensor '{name}': invalid dimension at axis {axis}.");
                    shape[axis++] = d;
                }

                if (!prop.Value.TryGetProperty("data_offsets", out var offEl) ||
                    offEl.ValueKind != JsonValueKind.Array || offEl.GetArrayLength() != 2)
                    throw new InvalidDataException(
                        $"Safetensors tensor '{name}' must declare 'data_offsets' as a 2-element array.");

                long begin, end;
                {
                    var itr = offEl.EnumerateArray();
                    itr.MoveNext(); begin = itr.Current.GetInt64();
                    itr.MoveNext(); end = itr.Current.GetInt64();
                }

                if (begin < 0 || end < begin)
                    throw new InvalidDataException(
                        $"Safetensors tensor '{name}': illegal data_offsets [{begin}, {end}].");
                if (end > dataSectionLength)
                    throw new InvalidDataException(
                        $"Safetensors tensor '{name}': data_offsets [{begin}, {end}] exceed data section length {dataSectionLength}.");

                long byteCount = end - begin;
                int elemSize = dtype.ElementSizeInBytes();
                if (elemSize > 0)
                {
                    long n = 1;
                    for (int i = 0; i < shape.Length; i++) n *= shape[i];
                    long expected = n * elemSize;
                    if (byteCount != expected)
                        throw new InvalidDataException(
                            $"Safetensors tensor '{name}': declared shape/dtype implies {expected} bytes but data_offsets span {byteCount}.");
                }

                tensors.Add(new SafetensorsTensorDescriptor(name, dtype, shape, begin, end));
            }

            return (metadata, tensors);
        }
    }

    /// <summary>
    /// Returns a pointer to the first byte of the given tensor's raw data
    /// in the memory-mapped region. Throws if the tensor is unknown.
    /// </summary>
    public nint GetTensorPointer(string name)
    {
        if (!TensorsByName.TryGetValue(name, out var desc))
            throw new KeyNotFoundException($"Safetensors file has no tensor named '{name}'.");
        return DataBasePointer + (nint)desc.DataBeginOffset;
    }

    /// <summary>
    /// Returns a <see cref="ReadOnlySpan{Byte}"/> over the raw bytes of the
    /// given tensor's data (directly backed by the memory-mapped view —
    /// valid until this <see cref="SafetensorsFile"/> is disposed). Throws
    /// if the tensor is unknown or its byte count exceeds <c>Int32.MaxValue</c>.
    /// </summary>
    public ReadOnlySpan<byte> GetTensorSpan(string name)
    {
        if (!TensorsByName.TryGetValue(name, out var desc))
            throw new KeyNotFoundException($"Safetensors file has no tensor named '{name}'.");
        if (desc.ByteCount > int.MaxValue)
            throw new InvalidOperationException(
                $"Tensor '{name}' byte count {desc.ByteCount} exceeds Int32.MaxValue; use GetTensorPointer instead.");
        byte* p = (byte*)DataBasePointer + desc.DataBeginOffset;
        return new ReadOnlySpan<byte>(p, (int)desc.ByteCount);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
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
}
