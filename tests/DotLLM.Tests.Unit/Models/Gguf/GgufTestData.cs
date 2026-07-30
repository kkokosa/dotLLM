using System.Text;
using DotLLM.Models.Gguf;

namespace DotLLM.Tests.Unit.Models.Gguf;

/// <summary>
/// Helper that builds synthetic GGUF byte arrays in-memory for testing.
/// </summary>
internal sealed class GgufTestData
{
    private readonly MemoryStream _stream = new();
    private readonly BinaryWriter _writer;
    private readonly uint _version;

    /// <summary>Metadata entries to write.</summary>
    private readonly List<Action<BinaryWriter>> _metadataWriters = [];

    /// <summary>Tensor info entries to write.</summary>
    private readonly List<Action<BinaryWriter>> _tensorWriters = [];

    /// <summary>Raw tensor data blobs (written after alignment padding).</summary>
    private readonly List<byte[]> _tensorDataBlobs = [];

    public GgufTestData(uint version = 3)
    {
        _version = version;
        _writer = new BinaryWriter(_stream);
    }

    /// <summary>Adds a metadata key-value pair.</summary>
    public GgufTestData AddMetadata(string key, GgufValueType type, Action<BinaryWriter> writeValue)
    {
        _metadataWriters.Add(writer =>
        {
            WriteGgufString(writer, key);
            writer.Write((uint)type);
            writeValue(writer);
        });
        return this;
    }

    /// <summary>Adds a string metadata entry.</summary>
    public GgufTestData AddString(string key, string value) =>
        AddMetadata(key, GgufValueType.String, w => WriteGgufString(w, value));

    /// <summary>Adds a uint32 metadata entry.</summary>
    public GgufTestData AddUInt32(string key, uint value) =>
        AddMetadata(key, GgufValueType.UInt32, w => w.Write(value));

    /// <summary>Adds a float32 metadata entry.</summary>
    public GgufTestData AddFloat32(string key, float value) =>
        AddMetadata(key, GgufValueType.Float32, w => w.Write(value));

    /// <summary>Adds a bool metadata entry.</summary>
    public GgufTestData AddBool(string key, bool value) =>
        AddMetadata(key, GgufValueType.Bool, w => w.Write((byte)(value ? 1 : 0)));

    /// <summary>Adds a uint8 metadata entry.</summary>
    public GgufTestData AddUInt8(string key, byte value) =>
        AddMetadata(key, GgufValueType.UInt8, w => w.Write(value));

    /// <summary>Adds an int8 metadata entry.</summary>
    public GgufTestData AddInt8(string key, sbyte value) =>
        AddMetadata(key, GgufValueType.Int8, w => w.Write(value));

    /// <summary>Adds a uint16 metadata entry.</summary>
    public GgufTestData AddUInt16(string key, ushort value) =>
        AddMetadata(key, GgufValueType.UInt16, w => w.Write(value));

    /// <summary>Adds an int16 metadata entry.</summary>
    public GgufTestData AddInt16(string key, short value) =>
        AddMetadata(key, GgufValueType.Int16, w => w.Write(value));

    /// <summary>Adds an int32 metadata entry.</summary>
    public GgufTestData AddInt32(string key, int value) =>
        AddMetadata(key, GgufValueType.Int32, w => w.Write(value));

    /// <summary>Adds a uint64 metadata entry.</summary>
    public GgufTestData AddUInt64(string key, ulong value) =>
        AddMetadata(key, GgufValueType.UInt64, w => w.Write(value));

    /// <summary>Adds an int64 metadata entry.</summary>
    public GgufTestData AddInt64(string key, long value) =>
        AddMetadata(key, GgufValueType.Int64, w => w.Write(value));

    /// <summary>Adds a float64 metadata entry.</summary>
    public GgufTestData AddFloat64(string key, double value) =>
        AddMetadata(key, GgufValueType.Float64, w => w.Write(value));

    /// <summary>Adds a string array metadata entry.</summary>
    public GgufTestData AddStringArray(string key, string[] values) =>
        AddMetadata(key, GgufValueType.Array, w =>
        {
            w.Write((uint)GgufValueType.String);
            WriteLength(w, (ulong)values.Length);
            foreach (string s in values)
                WriteGgufString(w, s);
        });

    /// <summary>Adds a float32 array metadata entry.</summary>
    public GgufTestData AddFloat32Array(string key, float[] values) =>
        AddMetadata(key, GgufValueType.Array, w =>
        {
            w.Write((uint)GgufValueType.Float32);
            WriteLength(w, (ulong)values.Length);
            foreach (float f in values)
                w.Write(f);
        });

    /// <summary>Adds an int32 array metadata entry.</summary>
    public GgufTestData AddInt32Array(string key, int[] values) =>
        AddMetadata(key, GgufValueType.Array, w =>
        {
            w.Write((uint)GgufValueType.Int32);
            WriteLength(w, (ulong)values.Length);
            foreach (int v in values)
                w.Write(v);
        });

    /// <summary>Adds a tensor info entry and its data blob.</summary>
    public GgufTestData AddTensor(string name, int[] dims, uint quantType, byte[] data)
    {
        ulong dataOffset = 0;
        foreach (byte[] blob in _tensorDataBlobs)
            dataOffset += (ulong)blob.Length;

        _tensorWriters.Add(writer =>
        {
            WriteGgufString(writer, name);
            writer.Write((uint)dims.Length);
            foreach (int dim in dims)
                writer.Write((ulong)dim);
            writer.Write(quantType);
            writer.Write(dataOffset);
        });

        _tensorDataBlobs.Add(data);
        return this;
    }

    /// <summary>Builds the complete GGUF byte array.</summary>
    public byte[] Build()
    {
        _stream.SetLength(0);
        _stream.Position = 0;

        // Header. GGUF v2 and v3 both store counts as uint64 (only the obsolete v1 used uint32).
        _writer.Write(GgufReader.GgufMagic);
        _writer.Write(_version);
        _writer.Write((ulong)_tensorWriters.Count);
        _writer.Write((ulong)_metadataWriters.Count);

        // Metadata
        foreach (var writeMetadata in _metadataWriters)
            writeMetadata(_writer);

        // Tensor infos
        foreach (var writeTensor in _tensorWriters)
            writeTensor(_writer);

        _writer.Flush();

        // Align to 32 bytes (default alignment)
        long dataStart = AlignUp(_stream.Position, 32);
        while (_stream.Position < dataStart)
            _writer.Write((byte)0);

        // Tensor data
        foreach (byte[] blob in _tensorDataBlobs)
            _writer.Write(blob);

        _writer.Flush();
        return _stream.ToArray();
    }

    /// <summary>Writes the GGUF bytes to a temp file and returns the path.</summary>
    public string WriteToTempFile()
    {
        byte[] data = Build();
        string path = Path.Combine(Path.GetTempPath(), $"dotllm_test_{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, data);
        return path;
    }

    private void WriteGgufString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        WriteLength(writer, (ulong)bytes.Length);
        writer.Write(bytes);
    }

    private void WriteLength(BinaryWriter writer, ulong length)
    {
        // String and array lengths are uint64 in both supported versions (v2 and v3); v1 used uint32.
        writer.Write(length);
    }

    private static long AlignUp(long value, uint alignment)
    {
        long mask = alignment - 1;
        return (value + mask) & ~mask;
    }
}
