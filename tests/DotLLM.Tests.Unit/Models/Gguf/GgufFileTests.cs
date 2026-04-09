using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Gguf;

public class GgufFileTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (string path in _tempFiles)
        {
            try { File.Delete(path); } catch { /* best-effort cleanup */ }
        }
    }

    private string WriteTempGguf(GgufTestData data)
    {
        string path = data.WriteToTempFile();
        _tempFiles.Add(path);
        return path;
    }

    [Fact]
    public void Open_ValidFile_ParsesHeaderAndMetadata()
    {
        var data = new GgufTestData(version: 3)
            .AddString("general.architecture", "llama")
            .AddUInt32("llama.block_count", 32);
        string path = WriteTempGguf(data);

        using var file = GgufFile.Open(path);

        Assert.Equal(3u, file.Header.Version);
        Assert.Equal(0ul, file.Header.TensorCount);
        Assert.Equal(2ul, file.Header.MetadataKvCount);
        Assert.Equal("llama", file.Metadata.GetString("general.architecture"));
        Assert.Equal(32u, file.Metadata.GetUInt32("llama.block_count"));
    }

    [Fact]
    public void Open_WithTensors_ProvidesDataPointer()
    {
        // F32 tensor with shape [8, 8] = 64 elements = 256 bytes
        byte[] tensorData = new byte[256];
        for (int i = 0; i < tensorData.Length; i++)
            tensorData[i] = (byte)((i % 255) + 1);

        var data = new GgufTestData(version: 3)
            .AddString("general.architecture", "llama")
            .AddTensor("test.weight", [8, 8], 0, tensorData); // F32, 8x8
        string path = WriteTempGguf(data);

        using var file = GgufFile.Open(path);

        Assert.Equal(1ul, file.Header.TensorCount);
        Assert.Single(file.Tensors);
        Assert.NotEqual(nint.Zero, file.DataBasePointer);

        // Verify tensor data is accessible through the pointer.
        unsafe
        {
            byte* ptr = (byte*)file.DataBasePointer;
            Assert.Equal(1, ptr[0]); // First byte of our tensor data
            Assert.Equal(1, ptr[255]); // Last byte (256 % 255 + 1 = 2, but wraps)
        }
    }

    [Fact]
    public void Open_TensorsByName_LookupWorks()
    {
        var data = new GgufTestData(version: 3)
            .AddTensor("layer.0.weight", [10], 0, new byte[40])
            .AddTensor("layer.1.weight", [20], 0, new byte[80]);
        string path = WriteTempGguf(data);

        using var file = GgufFile.Open(path);

        Assert.True(file.TensorsByName.ContainsKey("layer.0.weight"));
        Assert.True(file.TensorsByName.ContainsKey("layer.1.weight"));
        Assert.False(file.TensorsByName.ContainsKey("nonexistent"));
        Assert.Equal(10, file.TensorsByName["layer.0.weight"].Shape[0]);
    }

    [Fact]
    public void Open_NoTensors_DataPointerIsZero()
    {
        var data = new GgufTestData(version: 3)
            .AddString("key", "value");
        string path = WriteTempGguf(data);

        using var file = GgufFile.Open(path);

        Assert.Equal(nint.Zero, file.DataBasePointer);
        Assert.Empty(file.Tensors);
    }

    [Fact]
    public void Open_FileNotFound_Throws()
    {
        Assert.Throws<FileNotFoundException>(() => GgufFile.Open("/nonexistent/path.gguf"));
    }

    [Fact]
    public void Dispose_CanBeCalledTwice()
    {
        var data = new GgufTestData(version: 3)
            .AddTensor("w", [4], 0, new byte[16]);
        string path = WriteTempGguf(data);

        var file = GgufFile.Open(path);
        file.Dispose();
        file.Dispose(); // Should not throw.
    }

    [Fact]
    public void Open_V2File_ParsesCorrectly()
    {
        var data = new GgufTestData(version: 2)
            .AddString("general.architecture", "llama")
            .AddTensor("w", [4], 0, new byte[16]);
        string path = WriteTempGguf(data);

        using var file = GgufFile.Open(path);

        Assert.Equal(2u, file.Header.Version);
        Assert.Equal("llama", file.Metadata.GetString("general.architecture"));
        Assert.Single(file.Tensors);
    }

    [Fact]
    public void Open_DataSectionOffset_IsAligned()
    {
        var data = new GgufTestData(version: 3)
            .AddString("key", "value")
            .AddTensor("w", [4], 0, new byte[16]);
        string path = WriteTempGguf(data);

        using var file = GgufFile.Open(path);

        Assert.Equal(0, file.DataSectionOffset % 32);
    }

    [Fact]
    public void Open_TensorOffsetBeyondFile_Throws()
    {
        // Build a GGUF with two F32 tensors. First: [4] = 16 bytes at offset 0.
        // Second: [4] = 16 bytes at offset 16. Total data = 32 bytes.
        // Truncate just enough so second tensor's data doesn't fit but first still does.
        var data = new GgufTestData(version: 3)
            .AddTensor("first", [4], 0, new byte[16])
            .AddTensor("second", [4], 0, new byte[16]);
        string path = WriteTempGguf(data);

        // Remove 4 bytes: first tensor (offset 0, 16 bytes) still fits,
        // but second tensor (offset 16, needs 16 bytes) extends past EOF.
        byte[] bytes = File.ReadAllBytes(path);
        File.WriteAllBytes(path, bytes[..(bytes.Length - 4)]);

        var ex = Assert.Throws<InvalidDataException>(() => GgufFile.Open(path));
        Assert.Contains("second", ex.Message);
    }

    [Fact]
    public void Open_TensorDataExtendsPastEof_Throws()
    {
        // Tensor offset is valid (0) but file is truncated so data doesn't fully fit.
        var data = new GgufTestData(version: 3)
            .AddTensor("partial", [8], 0, new byte[32]); // F32, 8 elements = 32 bytes
        string path = WriteTempGguf(data);

        // Remove just a few bytes from the end — offset 0 is valid but 32 bytes don't fit.
        byte[] bytes = File.ReadAllBytes(path);
        File.WriteAllBytes(path, bytes[..(bytes.Length - 4)]);

        var ex = Assert.Throws<InvalidDataException>(() => GgufFile.Open(path));
        Assert.Contains("partial", ex.Message);
        Assert.Contains("extends beyond", ex.Message);
    }

    [Fact]
    public void Open_NonPowerOf2Alignment_Throws()
    {
        var data = new GgufTestData(version: 3)
            .AddUInt32("general.alignment", 3)
            .AddTensor("w", [4], 0, new byte[16]);
        string path = WriteTempGguf(data);

        var ex = Assert.Throws<InvalidDataException>(() => GgufFile.Open(path));
        Assert.Contains("power of 2", ex.Message);
    }

    [Fact]
    public void Open_ZeroAlignment_Throws()
    {
        var data = new GgufTestData(version: 3)
            .AddUInt32("general.alignment", 0)
            .AddTensor("w", [4], 0, new byte[16]);
        string path = WriteTempGguf(data);

        var ex = Assert.Throws<InvalidDataException>(() => GgufFile.Open(path));
        Assert.Contains("power of 2", ex.Message);
    }
}
