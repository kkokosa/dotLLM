using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

namespace DotLLM.Diagnostics;

/// <summary>
/// Loads a pre-trained <see cref="SparseAutoencoder"/> from disk.
/// </summary>
/// <remarks>
/// <para>
/// Supports the SAELens-convention SafeTensors layout: a <c>.safetensors</c> file containing
/// the four weight tensors (<c>W_enc</c>, <c>b_enc</c>, <c>W_dec</c>, <c>b_dec</c>) alongside an
/// optional sibling <c>cfg.json</c> describing architecture parameters (<c>d_in</c>,
/// <c>d_sae</c>, <c>apply_b_dec_to_input</c>).
/// </para>
/// <para>
/// This is a deliberately minimal reader scoped to SAE checkpoints — it parses just the
/// 8-byte header-length prefix, the JSON header, and slices F32 tensors out of the data
/// region. We avoid taking a project reference on <c>DotLLM.Models</c> to keep
/// <c>DotLLM.Diagnostics</c>'s dependency surface narrow (Core-only). Once
/// <c>DotLLM.Models.SafeTensors.SafetensorsFile</c> is promoted into a more general location
/// the loader can be retargeted at it — the public <see cref="Load"/> / <see cref="LoadFromBytes"/>
/// surface is the seam.
/// </para>
/// <para>
/// dtype support: F32 only. F16 / BF16 are an explicit follow-up — when needed, add a
/// per-tensor dtype branch in <see cref="ReadTensorAsAlignedF32"/> that upcasts on copy.
/// </para>
/// </remarks>
public static class SaeLoader
{
    private const string WEncName = "W_enc";
    private const string BEncName = "b_enc";
    private const string WDecName = "W_dec";
    private const string BDecName = "b_dec";

    /// <summary>
    /// Loads an SAE from <paramref name="safetensorsPath"/>, picking up a sibling
    /// <c>cfg.json</c> if it exists.
    /// </summary>
    /// <param name="safetensorsPath">Path to the SAE's <c>.safetensors</c> file.</param>
    /// <param name="topK">Top-K to assign to the returned SAE. Defaults to 32.</param>
    /// <returns>A new <see cref="SparseAutoencoder"/> ready for use.</returns>
    /// <exception cref="FileNotFoundException">When <paramref name="safetensorsPath"/> does not exist.</exception>
    /// <exception cref="InvalidDataException">When the file does not contain the expected SAE tensors.</exception>
    public static SparseAutoencoder Load(string safetensorsPath, int topK = 32)
    {
        ArgumentException.ThrowIfNullOrEmpty(safetensorsPath);
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException("SAE safetensors file not found.", safetensorsPath);

        byte[] bytes = File.ReadAllBytes(safetensorsPath);

        SaeCfg? cfg = null;
        string? cfgPath = Path.Combine(Path.GetDirectoryName(safetensorsPath) ?? ".", "cfg.json");
        if (File.Exists(cfgPath))
            cfg = SaeCfg.ParseJson(File.ReadAllText(cfgPath));

        return LoadFromBytes(bytes, cfg, topK);
    }

    /// <summary>
    /// Loads an SAE from an in-memory safetensors byte buffer, optionally taking an
    /// explicit <paramref name="cfg"/>. Intended for tests and for callers that have already
    /// resolved the cfg out-of-band.
    /// </summary>
    /// <param name="safetensorsBytes">Raw safetensors file contents (header + data region).</param>
    /// <param name="cfg">Optional cfg.json — overrides defaults for <c>apply_b_dec_to_input</c>. When null, defaults are used.</param>
    /// <param name="topK">Top-K to assign to the returned SAE.</param>
    /// <returns>A new <see cref="SparseAutoencoder"/>.</returns>
    /// <exception cref="InvalidDataException">When the buffer does not contain the expected SAE tensors.</exception>
    public static unsafe SparseAutoencoder LoadFromBytes(ReadOnlySpan<byte> safetensorsBytes, SaeCfg? cfg = null, int topK = 32)
    {
        if (safetensorsBytes.Length < 8)
            throw new InvalidDataException("Safetensors buffer too short for the 8-byte header-length prefix.");

        ulong headerLen = BinaryPrimitives.ReadUInt64LittleEndian(safetensorsBytes[..8]);
        if (headerLen == 0 || headerLen > (ulong)(safetensorsBytes.Length - 8))
            throw new InvalidDataException($"Invalid safetensors header length {headerLen}.");

        string headerJson = Encoding.UTF8.GetString(safetensorsBytes.Slice(8, (int)headerLen));
        long dataOffset = 8 + (long)headerLen;

        using var doc = JsonDocument.Parse(headerJson);
        var root = doc.RootElement;

        if (!TryGetTensor(root, WEncName, out var wEncDesc) ||
            !TryGetTensor(root, BEncName, out var bEncDesc) ||
            !TryGetTensor(root, WDecName, out var wDecDesc) ||
            !TryGetTensor(root, BDecName, out var bDecDesc))
        {
            throw new InvalidDataException(
                $"Safetensors file is missing one or more required SAE tensors: " +
                $"{WEncName}, {BEncName}, {WDecName}, {BDecName}. Found: " +
                string.Join(", ", EnumerateTensorNames(root)));
        }

        if (bEncDesc.Shape.Length != 1) throw new InvalidDataException($"{BEncName} must be 1-D, got shape [{string.Join(",", bEncDesc.Shape)}].");
        if (bDecDesc.Shape.Length != 1) throw new InvalidDataException($"{BDecName} must be 1-D, got shape [{string.Join(",", bDecDesc.Shape)}].");
        if (wEncDesc.Shape.Length != 2) throw new InvalidDataException($"{WEncName} must be 2-D, got shape [{string.Join(",", wEncDesc.Shape)}].");
        if (wDecDesc.Shape.Length != 2) throw new InvalidDataException($"{WDecName} must be 2-D, got shape [{string.Join(",", wDecDesc.Shape)}].");

        // Convention: W_enc is [d_in, d_sae], W_dec is [d_sae, d_in]. b_enc length = d_sae;
        // b_dec length = d_in. Cross-check shapes for internal consistency.
        int dIn = (int)bDecDesc.Shape[0];
        int dSae = (int)bEncDesc.Shape[0];

        if (wEncDesc.Shape[0] != dIn || wEncDesc.Shape[1] != dSae)
            throw new InvalidDataException(
                $"{WEncName} shape [{wEncDesc.Shape[0]},{wEncDesc.Shape[1]}] does not match (d_in={dIn}, d_sae={dSae}).");
        if (wDecDesc.Shape[0] != dSae || wDecDesc.Shape[1] != dIn)
            throw new InvalidDataException(
                $"{WDecName} shape [{wDecDesc.Shape[0]},{wDecDesc.Shape[1]}] does not match (d_sae={dSae}, d_in={dIn}).");

        // Cross-check with cfg.json if provided — cfg wins if it disagrees with the
        // header (it carries provenance metadata that the tensor shapes can't express,
        // e.g. apply_b_dec_to_input). Shape mismatch is fatal.
        if (cfg is SaeCfg cfgValue)
        {
            if (cfgValue.DIn is int cfgDIn && cfgDIn != dIn)
                throw new InvalidDataException($"cfg.json d_in={cfgDIn} disagrees with tensor-derived d_in={dIn}.");
            if (cfgValue.DSae is int cfgDSae && cfgDSae != dSae)
                throw new InvalidDataException($"cfg.json d_sae={cfgDSae} disagrees with tensor-derived d_sae={dSae}.");
        }

        // Read each tensor into 64-byte-aligned unmanaged memory.
        nint wEncPtr = ReadTensorAsAlignedF32(safetensorsBytes, dataOffset, wEncDesc);
        nint bEncPtr = ReadTensorAsAlignedF32(safetensorsBytes, dataOffset, bEncDesc);
        nint wDecPtr = ReadTensorAsAlignedF32(safetensorsBytes, dataOffset, wDecDesc);
        nint bDecPtr = ReadTensorAsAlignedF32(safetensorsBytes, dataOffset, bDecDesc);

        bool applyBDecToInput = cfg?.ApplyBDecToInput ?? false;
        return new SparseAutoencoder(dIn, dSae, wEncPtr, bEncPtr, wDecPtr, bDecPtr, topK, applyBDecToInput);
    }

    private static bool TryGetTensor(JsonElement root, string name, out TensorDescriptor descriptor)
    {
        descriptor = default;
        if (!root.TryGetProperty(name, out var el)) return false;
        if (!el.TryGetProperty("dtype", out var dtypeEl)) return false;
        if (!el.TryGetProperty("shape", out var shapeEl)) return false;
        if (!el.TryGetProperty("data_offsets", out var offsetsEl)) return false;

        string dtype = dtypeEl.GetString() ?? "";
        var shape = new long[shapeEl.GetArrayLength()];
        int i = 0;
        foreach (var dim in shapeEl.EnumerateArray()) shape[i++] = dim.GetInt64();

        long begin = offsetsEl[0].GetInt64();
        long end = offsetsEl[1].GetInt64();

        descriptor = new TensorDescriptor(dtype, shape, begin, end);
        return true;
    }

    private static IEnumerable<string> EnumerateTensorNames(JsonElement root)
    {
        foreach (var prop in root.EnumerateObject())
        {
            if (prop.Name == "__metadata__") continue;
            yield return prop.Name;
        }
    }

    private static unsafe nint ReadTensorAsAlignedF32(ReadOnlySpan<byte> bytes, long dataOffset, TensorDescriptor desc)
    {
        if (!string.Equals(desc.DType, "F32", StringComparison.Ordinal))
            throw new NotSupportedException(
                $"SAE tensor dtype '{desc.DType}' is not supported by the minimal SAE loader. " +
                $"F32 only. (F16/BF16 support is a planned follow-up.)");

        long byteLen = desc.End - desc.Begin;
        if (byteLen <= 0 || byteLen % sizeof(float) != 0)
            throw new InvalidDataException($"Tensor byte length {byteLen} is not a positive multiple of 4 (F32).");

        long elemCount = 1;
        foreach (long dim in desc.Shape) elemCount *= dim;
        if (elemCount * sizeof(float) != byteLen)
            throw new InvalidDataException(
                $"Tensor element count {elemCount} * 4 = {elemCount * 4} does not match byte length {byteLen}.");

        long absoluteBegin = dataOffset + desc.Begin;
        if (absoluteBegin < 0 || absoluteBegin + byteLen > bytes.Length)
            throw new InvalidDataException($"Tensor data range [{absoluteBegin}, {absoluteBegin + byteLen}) exceeds buffer length {bytes.Length}.");

        var ptr = NativeMemory.AlignedAlloc((nuint)byteLen, 64);
        bytes.Slice((int)absoluteBegin, (int)byteLen)
            .CopyTo(new Span<byte>(ptr, (int)byteLen));
        return (nint)ptr;
    }

    private readonly record struct TensorDescriptor(string DType, long[] Shape, long Begin, long End);
}

/// <summary>
/// Architecture metadata loaded from an SAE's sibling <c>cfg.json</c> (SAELens convention).
/// All fields are optional — when absent, the loader infers what it can from tensor shapes.
/// </summary>
/// <param name="DIn">Input/output activation dimensionality (<c>d_in</c>), if specified.</param>
/// <param name="DSae">Dictionary size (<c>d_sae</c>), if specified.</param>
/// <param name="ApplyBDecToInput">Whether to pre-subtract <c>b_dec</c> from the activation before encoding. SAELens default is <c>true</c>; we default to <c>false</c> when absent since synthetic test fixtures use the simpler form.</param>
/// <param name="HookPoint">SAELens <c>hook_name</c> string (e.g. <c>blocks.5.hook_resid_post</c>). Informational only at this layer — the caller chooses which layer to attach the hook to.</param>
public readonly record struct SaeCfg(int? DIn, int? DSae, bool ApplyBDecToInput, string? HookPoint)
{
    /// <summary>
    /// Parses a SAELens-style <c>cfg.json</c> document.
    /// </summary>
    /// <param name="json">JSON document text.</param>
    /// <returns>The parsed config; unrecognised fields are ignored.</returns>
    public static SaeCfg ParseJson(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        int? dIn = TryGetInt(root, "d_in");
        int? dSae = TryGetInt(root, "d_sae");
        bool applyBDec = root.TryGetProperty("apply_b_dec_to_input", out var bel) && bel.ValueKind == JsonValueKind.True;
        string? hookPoint = root.TryGetProperty("hook_name", out var hpel) ? hpel.GetString() : null;

        return new SaeCfg(dIn, dSae, applyBDec, hookPoint);
    }

    private static int? TryGetInt(JsonElement el, string name)
    {
        if (!el.TryGetProperty(name, out var v)) return null;
        return v.ValueKind == JsonValueKind.Number ? v.GetInt32() : null;
    }
}
