using System.Buffers.Binary;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// Unit tests for the bare <see cref="SafetensorsFile"/> parser. Covers
/// the 8-byte little-endian length prefix, JSON header parsing, dtype
/// token mapping, and the memory-mapped data view. Exercises success
/// cases plus the hostile inputs we expect to catch at open time.
/// </summary>
public sealed class SafetensorsFileTests : IDisposable
{
    private readonly string _scratch;

    public SafetensorsFileTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-st-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string Scratch(string name) => Path.Combine(_scratch, name);

    [Fact]
    public void Open_ParsesHeader_RoundTripsTensors()
    {
        string path = Scratch("basic.safetensors");
        new SafetensorsFixtureBuilder()
            .AddFloat32("alpha", [2, 3], startValue: 1.0f)
            .AddFloat32("beta",  [4],    startValue: 100.0f)
            .WriteTo(path);

        using var sf = SafetensorsFile.Open(path);

        Assert.Equal(2, sf.Tensors.Count);
        Assert.Equal("alpha", sf.Tensors[0].Name);
        Assert.Equal("beta", sf.Tensors[1].Name);

        var alpha = sf.TensorsByName["alpha"];
        Assert.Equal(SafetensorsDType.F32, alpha.DType);
        Assert.Equal([2, 3], alpha.Shape);
        Assert.Equal(6, alpha.ElementCount);
        Assert.Equal(24, alpha.ByteCount);
        Assert.Equal(0, alpha.DataBeginOffset);

        var beta = sf.TensorsByName["beta"];
        Assert.Equal(24, beta.DataBeginOffset);
        Assert.Equal(40, beta.DataEndOffset);
    }

    [Fact]
    public void Open_DataBasePointer_YieldsExpectedBytes()
    {
        string path = Scratch("pointer.safetensors");
        new SafetensorsFixtureBuilder()
            .AddFloat32("ramp", [4], startValue: 7.0f)
            .WriteTo(path);

        using var sf = SafetensorsFile.Open(path);

        var span = sf.GetTensorSpan("ramp");
        Assert.Equal(16, span.Length);
        // Reinterpret as four floats; must equal 7,8,9,10.
        var asFloats = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(span);
        Assert.Equal([7.0f, 8.0f, 9.0f, 10.0f], asFloats.ToArray());
    }

    [Fact]
    public void Open_HeaderLength_MatchesPrefix()
    {
        string path = Scratch("hdrlen.safetensors");
        long written = new SafetensorsFixtureBuilder()
            .AddFloat32("x", [3])
            .WriteTo(path);

        using var sf = SafetensorsFile.Open(path);
        Assert.Equal(written, sf.HeaderLength);
        Assert.Equal(8 + written, sf.DataSectionOffset);
    }

    [Fact]
    public void Open_Metadata_Ignored_ButCaptured()
    {
        string path = Scratch("meta.safetensors");
        new SafetensorsFixtureBuilder()
            .AddFloat32("t", [1])
            .WithMetadata("format", "pt")
            .WithMetadata("note", "hi")
            .WriteTo(path);

        using var sf = SafetensorsFile.Open(path);
        Assert.Single(sf.Tensors); // __metadata__ is filtered out
        Assert.Equal("pt", sf.Metadata["format"]);
        Assert.Equal("hi", sf.Metadata["note"]);
    }

    [Fact]
    public void Open_MissingFile_Throws()
    {
        Assert.Throws<FileNotFoundException>(() =>
            SafetensorsFile.Open(Scratch("nope.safetensors")));
    }

    [Fact]
    public void Open_FileTooSmall_Throws()
    {
        string path = Scratch("tiny.safetensors");
        File.WriteAllBytes(path, [0x00, 0x01, 0x02]);
        Assert.Throws<InvalidDataException>(() => SafetensorsFile.Open(path));
    }

    [Fact]
    public void Open_HeaderLength_ExceedsFile_Throws()
    {
        string path = Scratch("oversize.safetensors");
        // Declared header length 1 GiB in an 8-byte-only file.
        Span<byte> buf = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(buf, 1UL << 30);
        using (var fs = File.Create(path)) { fs.Write(buf); }
        Assert.Throws<InvalidDataException>(() => SafetensorsFile.Open(path));
    }

    [Fact]
    public void Open_ShapeBytes_MismatchDtype_Throws()
    {
        // Declare [2,3] F32 but only provide 12 bytes (should be 24).
        string path = Scratch("shape-mismatch.safetensors");
        string header = "{\"bad\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,12]}}";
        byte[] headerBytes = System.Text.Encoding.UTF8.GetBytes(header);
        using (var fs = File.Create(path))
        {
            Span<byte> len = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(len, (ulong)headerBytes.Length);
            fs.Write(len);
            fs.Write(headerBytes);
            fs.Write(new byte[12]);
        }
        Assert.Throws<InvalidDataException>(() => SafetensorsFile.Open(path));
    }

    [Fact]
    public void DTypeExtensions_ParsesAllCanonicalTokens()
    {
        Assert.Equal(SafetensorsDType.F32, SafetensorsDTypeExtensions.Parse("F32"));
        Assert.Equal(SafetensorsDType.BF16, SafetensorsDTypeExtensions.Parse("BF16"));
        Assert.Equal(SafetensorsDType.F16, SafetensorsDTypeExtensions.Parse("F16"));
        Assert.Equal(SafetensorsDType.I64, SafetensorsDTypeExtensions.Parse("I64"));
        Assert.Equal(SafetensorsDType.Bool, SafetensorsDTypeExtensions.Parse("BOOL"));
        Assert.Equal(SafetensorsDType.Unknown, SafetensorsDTypeExtensions.Parse("QUACK"));
    }

    [Fact]
    public void DTypeExtensions_ElementSizes()
    {
        Assert.Equal(4, SafetensorsDType.F32.ElementSizeInBytes());
        Assert.Equal(2, SafetensorsDType.BF16.ElementSizeInBytes());
        Assert.Equal(2, SafetensorsDType.F16.ElementSizeInBytes());
        Assert.Equal(8, SafetensorsDType.F64.ElementSizeInBytes());
        Assert.Equal(1, SafetensorsDType.U8.ElementSizeInBytes());
        Assert.Equal(0, SafetensorsDType.Unknown.ElementSizeInBytes());
    }

    [Fact]
    public void Dispose_IsIdempotent()
    {
        string path = Scratch("dispose.safetensors");
        new SafetensorsFixtureBuilder().AddFloat32("a", [1]).WriteTo(path);

        var sf = SafetensorsFile.Open(path);
        sf.Dispose();
        sf.Dispose(); // must not throw
    }
}
