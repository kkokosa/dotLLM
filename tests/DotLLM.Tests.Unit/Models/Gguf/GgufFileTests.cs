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
        byte[] tensorData = new byte[64];
        for (int i = 0; i < tensorData.Length; i++)
            tensorData[i] = (byte)(i + 1);

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
            Assert.Equal(64, ptr[63]); // Last byte
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
        // Build a GGUF with two tensors. The second tensor's offset
        // is computed from the first blob size. Then truncate the file
        // so the second tensor's data section offset is past the end.
        var data = new GgufTestData(version: 3)
            .AddTensor("first", [4], 0, new byte[16])
            .AddTensor("second", [4], 0, new byte[16]);
        string path = WriteTempGguf(data);

        // Truncate the file to remove the second tensor's data entirely.
        // The second tensor has DataOffset=16 but the data section will be < 16 bytes.
        byte[] bytes = File.ReadAllBytes(path);
        File.WriteAllBytes(path, bytes[..(bytes.Length - 20)]);

        var ex = Assert.Throws<InvalidDataException>(() => GgufFile.Open(path));
        Assert.Contains("second", ex.Message);
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
