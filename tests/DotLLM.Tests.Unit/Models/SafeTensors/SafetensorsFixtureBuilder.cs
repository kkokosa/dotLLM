using System.Buffers.Binary;
using System.Text.Json;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// Writes a valid, byte-accurate synthetic safetensors file to disk for
/// use by the Stage D2 loader tests. Mirrors the HuggingFace layout
/// (LE u64 header length, UTF-8 JSON header, raw row-major data region).
/// </summary>
/// <remarks>
/// <para>
/// The test harness uses this builder exclusively — no real 1.55 GB
/// checkpoint is downloaded. Tensor content is deterministic (a ramp
/// pattern indexed by the writer's per-name call order) so tests can
/// assert on specific element values.
/// </para>
/// </remarks>
internal sealed class SafetensorsFixtureBuilder
{
    private readonly List<(string Name, string DType, int[] Shape, byte[] Bytes)> _tensors = new();
    private Dictionary<string, string>? _metadata;

    /// <summary>
    /// Adds an F32 tensor whose values are <c>startValue, startValue+1, …</c>.
    /// </summary>
    public SafetensorsFixtureBuilder AddFloat32(string name, int[] shape, float startValue = 0.0f)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        var bytes = new byte[n * sizeof(float)];
        for (long i = 0; i < n; i++)
        {
            float v = startValue + i;
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan((int)(i * 4), 4), v);
        }
        _tensors.Add((name, "F32", shape, bytes));
        return this;
    }

    /// <summary>
    /// Adds an F32 tensor with user-supplied element values.
    /// </summary>
    public SafetensorsFixtureBuilder AddFloat32(string name, int[] shape, ReadOnlySpan<float> values)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        if (values.Length != n)
            throw new ArgumentException(
                $"Shape implies {n} elements but values has length {values.Length}.", nameof(values));
        var bytes = new byte[n * sizeof(float)];
        for (long i = 0; i < n; i++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan((int)(i * 4), 4), values[(int)i]);
        _tensors.Add((name, "F32", shape, bytes));
        return this;
    }

    /// <summary>
    /// Adds a tensor with an arbitrary dtype token (for testing unsupported-
    /// dtype paths). The caller supplies both the dtype string and the raw
    /// data bytes; no validation that the string is a canonical safetensors
    /// dtype is performed.
    /// </summary>
    public SafetensorsFixtureBuilder AddRaw(string name, string dtype, int[] shape, byte[] bytes)
    {
        _tensors.Add((name, dtype, shape, bytes));
        return this;
    }

    public SafetensorsFixtureBuilder WithMetadata(string key, string value)
    {
        _metadata ??= new(StringComparer.Ordinal);
        _metadata[key] = value;
        return this;
    }

    /// <summary>
    /// Writes the safetensors binary to <paramref name="path"/>. Returns
    /// the header length (for round-trip assertions).
    /// </summary>
    public long WriteTo(string path)
    {
        // Build header JSON. Preserve insertion order so tests can assert
        // ordered Tensors list.
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = false }))
        {
            w.WriteStartObject();
            long offset = 0;
            foreach (var (name, dtype, shape, bytes) in _tensors)
            {
                w.WriteStartObject(name);
                w.WriteString("dtype", dtype);
                w.WritePropertyName("shape");
                w.WriteStartArray();
                foreach (var d in shape) w.WriteNumberValue(d);
                w.WriteEndArray();
                w.WritePropertyName("data_offsets");
                w.WriteStartArray();
                w.WriteNumberValue(offset);
                w.WriteNumberValue(offset + bytes.Length);
                w.WriteEndArray();
                w.WriteEndObject();
                offset += bytes.Length;
            }
            if (_metadata is not null)
            {
                w.WriteStartObject("__metadata__");
                foreach (var (k, v) in _metadata)
                    w.WriteString(k, v);
                w.WriteEndObject();
            }
            w.WriteEndObject();
        }

        byte[] headerJson = ms.ToArray();
        long headerLen = headerJson.Length;

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        Span<byte> prefix = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)headerLen);
        fs.Write(prefix);
        fs.Write(headerJson);
        foreach (var (_, _, _, bytes) in _tensors)
            fs.Write(bytes);

        return headerLen;
    }

}
