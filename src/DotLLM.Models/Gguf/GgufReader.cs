using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Static binary parser for the GGUF file format. Pure functions: bytes in, structs out.
/// Supports GGUF v2 and v3, which are identical on the wire: tensor/metadata counts,
/// string lengths and array lengths are all <c>uint64</c>. (The <c>uint32</c> form belongs to
/// the obsolete v1, which the header validation rejects.)
/// </summary>
public static class GgufReader
{
    /// <summary>GGUF magic number: "GGUF" in little-endian.</summary>
    public const uint GgufMagic = 0x46554747;

    /// <summary>
    /// Reads and validates the GGUF header from the current reader position.
    /// </summary>
    /// <param name="reader">A <see cref="BinaryReader"/> positioned at the start of a GGUF file.</param>
    /// <returns>The parsed header.</returns>
    /// <exception cref="InvalidDataException">Invalid magic number or unsupported version.</exception>
    public static GgufHeader ReadHeader(BinaryReader reader)
    {
        uint magic = reader.ReadUInt32();
        if (magic != GgufMagic)
            throw new InvalidDataException(
                $"Invalid GGUF magic number: 0x{magic:X8}. Expected 0x{GgufMagic:X8}.");

        uint version = reader.ReadUInt32();
        if (version is not (2 or 3))
            throw new InvalidDataException(
                $"Unsupported GGUF version: {version}. Only versions 2 and 3 are supported.");

        // GGUF v2 and v3 both store these counts as uint64 (the uint32 form was v1 only).
        ulong tensorCount = reader.ReadUInt64();
        ulong metadataKvCount = reader.ReadUInt64();

        return new GgufHeader(version, tensorCount, metadataKvCount);
    }

    /// <summary>
    /// Reads all metadata key-value pairs from the current reader position.
    /// </summary>
    /// <param name="reader">A <see cref="BinaryReader"/> positioned after the header.</param>
    /// <param name="header">The previously parsed header (provides version and KV count).</param>
    /// <returns>Dictionary of metadata entries keyed by name.</returns>
    public static Dictionary<string, GgufMetadataValue> ReadMetadata(BinaryReader reader, GgufHeader header)
    {
        var metadata = new Dictionary<string, GgufMetadataValue>((int)Math.Min(header.MetadataKvCount, int.MaxValue));

        for (ulong i = 0; i < header.MetadataKvCount; i++)
        {
            string key = ReadGgufString(reader, header.Version);
            var valueType = (GgufValueType)reader.ReadUInt32();
            object value = ReadMetadataValue(reader, header.Version, valueType);
            metadata[key] = new GgufMetadataValue(valueType, value);
        }

        return metadata;
    }

    /// <summary>
    /// Reads all tensor info entries from the current reader position.
    /// </summary>
    /// <param name="reader">A <see cref="BinaryReader"/> positioned after the metadata section.</param>
    /// <param name="header">The previously parsed header (provides version and tensor count).</param>
    /// <returns>List of tensor descriptors.</returns>
    /// <exception cref="NotSupportedException">Unrecognized quantization type.</exception>
    public static List<GgufTensorDescriptor> ReadTensorInfos(BinaryReader reader, GgufHeader header)
    {
        var tensors = new List<GgufTensorDescriptor>((int)Math.Min(header.TensorCount, int.MaxValue));

        for (ulong i = 0; i < header.TensorCount; i++)
        {
            string name = ReadGgufString(reader, header.Version);
            uint nDims = reader.ReadUInt32();

            var dims = new int[nDims];
            for (int d = 0; d < (int)nDims; d++)
            {
                ulong dim = reader.ReadUInt64();
                if (dim > int.MaxValue)
                    throw new InvalidDataException(
                        $"Tensor '{name}' dimension {d} is {dim}, which exceeds Int32.MaxValue.");
                dims[d] = (int)dim;
            }

            uint rawType = reader.ReadUInt32();
            if (!Enum.IsDefined(typeof(QuantizationType), (int)rawType))
                throw new NotSupportedException(
                    $"Tensor '{name}' has unrecognized quantization type: {rawType}.");

            var quantType = (QuantizationType)rawType;
            ulong offset = reader.ReadUInt64();

            tensors.Add(new GgufTensorDescriptor(name, new TensorShape(dims), quantType, offset));
        }

        return tensors;
    }

    /// <summary>
    /// Reads a GGUF length-prefixed UTF-8 string. The length is uint64 in both supported
    /// versions (v2 and v3); only the obsolete v1 used a uint32 length.
    /// </summary>
    internal static string ReadGgufString(BinaryReader reader, uint version)
    {
        ulong length = reader.ReadUInt64();

        if (length == 0)
            return string.Empty;

        if (length > int.MaxValue)
            throw new InvalidDataException($"GGUF string length {length} exceeds Int32.MaxValue.");

        byte[] bytes = reader.ReadBytes((int)length);
        return System.Text.Encoding.UTF8.GetString(bytes);
    }

    private static object ReadMetadataValue(BinaryReader reader, uint version, GgufValueType valueType)
    {
        return valueType switch
        {
            GgufValueType.UInt8 => reader.ReadByte(),
            GgufValueType.Int8 => reader.ReadSByte(),
            GgufValueType.UInt16 => reader.ReadUInt16(),
            GgufValueType.Int16 => reader.ReadInt16(),
            GgufValueType.UInt32 => reader.ReadUInt32(),
            GgufValueType.Int32 => reader.ReadInt32(),
            GgufValueType.Float32 => reader.ReadSingle(),
            GgufValueType.Bool => reader.ReadByte() != 0,
            GgufValueType.String => ReadGgufString(reader, version),
            GgufValueType.UInt64 => reader.ReadUInt64(),
            GgufValueType.Int64 => reader.ReadInt64(),
            GgufValueType.Float64 => reader.ReadDouble(),
            GgufValueType.Array => ReadArray(reader, version),
            _ => throw new InvalidDataException($"Unknown GGUF value type: {valueType}.")
        };
    }

    private static object ReadArray(BinaryReader reader, uint version)
    {
        var elementType = (GgufValueType)reader.ReadUInt32();
        // Array length is uint64 in both supported versions (v2 and v3); v1 used uint32.
        ulong count = reader.ReadUInt64();

        if (count > int.MaxValue)
            throw new InvalidDataException($"GGUF array length {count} exceeds Int32.MaxValue.");

        int len = (int)count;

        // Return strongly-typed arrays for common element types.
        return elementType switch
        {
            GgufValueType.UInt8 => ReadPrimitiveArray(reader, len, static r => r.ReadByte()),
            GgufValueType.Int8 => ReadPrimitiveArray(reader, len, static r => r.ReadSByte()),
            GgufValueType.UInt16 => ReadPrimitiveArray(reader, len, static r => r.ReadUInt16()),
            GgufValueType.Int16 => ReadPrimitiveArray(reader, len, static r => r.ReadInt16()),
            GgufValueType.UInt32 => ReadPrimitiveArray(reader, len, static r => r.ReadUInt32()),
            GgufValueType.Int32 => ReadPrimitiveArray(reader, len, static r => r.ReadInt32()),
            GgufValueType.Float32 => ReadPrimitiveArray(reader, len, static r => r.ReadSingle()),
            GgufValueType.Bool => ReadPrimitiveArray(reader, len, static r => r.ReadByte() != 0),
            GgufValueType.String => ReadStringArray(reader, len, version),
            GgufValueType.UInt64 => ReadPrimitiveArray(reader, len, static r => r.ReadUInt64()),
            GgufValueType.Int64 => ReadPrimitiveArray(reader, len, static r => r.ReadInt64()),
            GgufValueType.Float64 => ReadPrimitiveArray(reader, len, static r => r.ReadDouble()),
            _ => throw new InvalidDataException($"Unknown GGUF array element type: {elementType}.")
        };
    }

    private static T[] ReadPrimitiveArray<T>(BinaryReader reader, int count, Func<BinaryReader, T> readElement)
    {
        var array = new T[count];
        for (int i = 0; i < count; i++)
            array[i] = readElement(reader);
        return array;
    }

    private static string[] ReadStringArray(BinaryReader reader, int count, uint version)
    {
        var array = new string[count];
        for (int i = 0; i < count; i++)
            array[i] = ReadGgufString(reader, version);
        return array;
    }
}
