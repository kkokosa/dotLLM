using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Gguf;

public class GgufReaderTests
{
    #region Header

    [Fact]
    public void ReadHeader_V3_ParsesCorrectly()
    {
        var data = new GgufTestData(version: 3)
            .AddString("test.key", "value");
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);

        var header = GgufReader.ReadHeader(reader);

        Assert.Equal(3u, header.Version);
        Assert.Equal(0ul, header.TensorCount);
        Assert.Equal(1ul, header.MetadataKvCount);
    }

    [Fact]
    public void ReadHeader_V2_ParsesCorrectly()
    {
        var data = new GgufTestData(version: 2)
            .AddString("test.key", "value");
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);

        var header = GgufReader.ReadHeader(reader);

        Assert.Equal(2u, header.Version);
        Assert.Equal(0ul, header.TensorCount);
        Assert.Equal(1ul, header.MetadataKvCount);
    }

    [Fact]
    public void ReadHeader_InvalidMagic_Throws()
    {
        byte[] bytes = [0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00];
        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);

        var ex = Assert.Throws<InvalidDataException>(() => GgufReader.ReadHeader(reader));
        Assert.Contains("Invalid GGUF magic", ex.Message);
    }

    [Fact]
    public void ReadHeader_UnsupportedVersion_Throws()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        writer.Write(GgufReader.GgufMagic);
        writer.Write(99u); // bad version
        writer.Flush();
        stream.Position = 0;

        using var reader = new BinaryReader(stream);

        var ex = Assert.Throws<InvalidDataException>(() => GgufReader.ReadHeader(reader));
        Assert.Contains("Unsupported GGUF version: 99", ex.Message);
    }

    [Fact]
    public void ReadHeader_V3_WithTensors_ParsesCounts()
    {
        var data = new GgufTestData(version: 3)
            .AddString("general.architecture", "llama")
            .AddTensor("token_embd.weight", [32000, 4096], 0, new byte[32000 * 4096 * 4]);
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);

        var header = GgufReader.ReadHeader(reader);

        Assert.Equal(1ul, header.TensorCount);
        Assert.Equal(1ul, header.MetadataKvCount);
    }

    #endregion

    #region Metadata

    [Fact]
    public void ReadMetadata_AllScalarTypes_V3()
    {
        var data = new GgufTestData(version: 3)
            .AddUInt8("val.u8", 42)
            .AddInt8("val.i8", -7)
            .AddUInt16("val.u16", 1000)
            .AddInt16("val.i16", -500)
            .AddUInt32("val.u32", 100_000)
            .AddInt32("val.i32", -200_000)
            .AddFloat32("val.f32", 3.14f)
            .AddBool("val.bool", true)
            .AddString("val.str", "hello")
            .AddUInt64("val.u64", 9_000_000_000)
            .AddInt64("val.i64", -9_000_000_000)
            .AddFloat64("val.f64", 2.718281828);
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        var metadata = GgufReader.ReadMetadata(reader, header);

        Assert.Equal(12, metadata.Count);
        Assert.Equal((byte)42, metadata["val.u8"].Value);
        Assert.Equal((sbyte)-7, metadata["val.i8"].Value);
        Assert.Equal((ushort)1000, metadata["val.u16"].Value);
        Assert.Equal((short)-500, metadata["val.i16"].Value);
        Assert.Equal(100_000u, metadata["val.u32"].Value);
        Assert.Equal(-200_000, metadata["val.i32"].Value);
        Assert.Equal(3.14f, metadata["val.f32"].Value);
        Assert.Equal(true, metadata["val.bool"].Value);
        Assert.Equal("hello", metadata["val.str"].Value);
        Assert.Equal(9_000_000_000ul, metadata["val.u64"].Value);
        Assert.Equal(-9_000_000_000L, metadata["val.i64"].Value);
        Assert.Equal(2.718281828, metadata["val.f64"].Value);
    }

    [Fact]
    public void ReadMetadata_StringArray()
    {
        var data = new GgufTestData(version: 3)
            .AddStringArray("tokens", ["hello", "world", "test"]);
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        var metadata = GgufReader.ReadMetadata(reader, header);

        Assert.Equal(GgufValueType.Array, metadata["tokens"].Type);
        var array = (string[])metadata["tokens"].Value;
        Assert.Equal(["hello", "world", "test"], array);
    }

    [Fact]
    public void ReadMetadata_Float32Array()
    {
        float[] scores = [1.0f, 2.5f, -3.7f];
        var data = new GgufTestData(version: 3)
            .AddFloat32Array("scores", scores);
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        var metadata = GgufReader.ReadMetadata(reader, header);

        var array = (float[])metadata["scores"].Value;
        Assert.Equal(scores, array);
    }

    [Fact]
    public void ReadMetadata_EmptyMetadata()
    {
        var data = new GgufTestData(version: 3);
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        var metadata = GgufReader.ReadMetadata(reader, header);

        Assert.Empty(metadata);
    }

    [Fact]
    public void ReadMetadata_V2_ScalarTypes()
    {
        var data = new GgufTestData(version: 2)
            .AddString("key", "value")
            .AddUInt32("count", 42);
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        var metadata = GgufReader.ReadMetadata(reader, header);

        Assert.Equal(2, metadata.Count);
        Assert.Equal("value", metadata["key"].Value);
        Assert.Equal(42u, metadata["count"].Value);
    }

    /// <summary>
    /// Regression for the GGUF v2 width bug: v2 stores counts, string lengths and array lengths
    /// as <c>uint64</c> (identical to v3 — the <c>uint32</c> form was v1 only). The reader and the
    /// in-memory test writer both previously treated v2 as uint32, so they agreed with each other
    /// while disagreeing with the spec — and every real v2 file (e.g. TheBloke's Llama-2-13B-GGUF)
    /// mis-parsed: a uint32 string length consumed 4 bytes where the file wrote 8, cascading through
    /// the string-array metadata into garbage tensor names and offsets (the latter surfaced as a
    /// bogus "data extends beyond file boundary" with an empty tensor name). A scalar-only v2 case
    /// can't catch this — it needs a multi-element array followed by tensor infos so misalignment
    /// compounds. This asserts a full v2 parse (header → array metadata → tensor infos) is exact.
    /// </summary>
    [Fact]
    public void Reads_V2_FullFile_ArrayMetadataThenTensors_Exactly()
    {
        var data = new GgufTestData(version: 2)
            .AddString("general.architecture", "llama")
            .AddStringArray("tokenizer.ggml.tokens", ["<s>", "</s>", "hello", "world"])
            .AddUInt32("llama.block_count", 2)
            .AddTensor("token_embd.weight", [8, 16], 0, new byte[8 * 16 * 4])   // F32
            .AddTensor("blk.0.attn_q.weight", [16, 16], 1, new byte[16 * 16 * 2]) // F16
            .AddTensor("output_norm.weight", [16], 0, new byte[16 * 4]);          // F32
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);

        var header = GgufReader.ReadHeader(reader);
        Assert.Equal(2u, header.Version);
        Assert.Equal(3ul, header.TensorCount);
        Assert.Equal(3ul, header.MetadataKvCount);

        var metadata = GgufReader.ReadMetadata(reader, header);
        Assert.Equal("llama", metadata["general.architecture"].Value);
        Assert.Equal(["<s>", "</s>", "hello", "world"],
            (string[])metadata["tokenizer.ggml.tokens"].Value);
        Assert.Equal(2u, metadata["llama.block_count"].Value);

        var tensors = GgufReader.ReadTensorInfos(reader, header);
        Assert.Equal(3, tensors.Count);
        Assert.Equal("token_embd.weight", tensors[0].Name);
        Assert.Equal("blk.0.attn_q.weight", tensors[1].Name);
        Assert.Equal("output_norm.weight", tensors[2].Name);
        // Offsets are sequential blobs (no alignment between them in the writer).
        Assert.Equal(0ul, tensors[0].DataOffset);
        Assert.Equal((ulong)(8 * 16 * 4), tensors[1].DataOffset);
        Assert.Equal((ulong)(8 * 16 * 4 + 16 * 16 * 2), tensors[2].DataOffset);
    }

    #endregion

    #region Tensor Infos

    [Fact]
    public void ReadTensorInfos_SingleTensor()
    {
        var data = new GgufTestData(version: 3)
            .AddTensor("weight", [4096, 4096], 0, new byte[4096 * 4096 * 4]); // F32
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        _ = GgufReader.ReadMetadata(reader, header);
        var tensors = GgufReader.ReadTensorInfos(reader, header);

        Assert.Single(tensors);
        Assert.Equal("weight", tensors[0].Name);
        Assert.Equal(2, tensors[0].Shape.Rank);
        Assert.Equal(4096, tensors[0].Shape[0]);
        Assert.Equal(4096, tensors[0].Shape[1]);
        Assert.Equal(Core.Configuration.QuantizationType.F32, tensors[0].QuantizationType);
        Assert.Equal(0ul, tensors[0].DataOffset);
    }

    [Fact]
    public void ReadTensorInfos_MultipleTensors()
    {
        var data = new GgufTestData(version: 3)
            .AddTensor("a", [10], 0, new byte[40])
            .AddTensor("b", [20], 1, new byte[40]); // F16
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        _ = GgufReader.ReadMetadata(reader, header);
        var tensors = GgufReader.ReadTensorInfos(reader, header);

        Assert.Equal(2, tensors.Count);
        Assert.Equal("a", tensors[0].Name);
        Assert.Equal("b", tensors[1].Name);
        Assert.Equal(0ul, tensors[0].DataOffset);
        Assert.Equal(40ul, tensors[1].DataOffset);
    }

    [Fact]
    public void ReadTensorInfos_UnrecognizedQuantType_Throws()
    {
        var data = new GgufTestData(version: 3)
            .AddTensor("bad_tensor", [10], 999, new byte[10]);
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        _ = GgufReader.ReadMetadata(reader, header);

        var ex = Assert.Throws<NotSupportedException>(() => GgufReader.ReadTensorInfos(reader, header));
        Assert.Contains("bad_tensor", ex.Message);
        Assert.Contains("999", ex.Message);
    }

    [Fact]
    public void ReadTensorInfos_NoTensors()
    {
        var data = new GgufTestData(version: 3)
            .AddString("key", "value");
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        _ = GgufReader.ReadMetadata(reader, header);
        var tensors = GgufReader.ReadTensorInfos(reader, header);

        Assert.Empty(tensors);
    }

    #endregion
}
