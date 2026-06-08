namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Canonical dtypes declared in a safetensors header, per the
/// <see href="https://github.com/huggingface/safetensors">safetensors spec</see>
/// v0.4.x. Mapped to the string tokens that appear in the header JSON
/// (<c>"F32"</c>, <c>"BF16"</c>, …).
/// </summary>
/// <remarks>
/// <para>
/// Stage D2 of the Mamba-3 PoC only materialises an F32 read path — the
/// only dtype actually present in <c>ib-ssm/mamba3-370M-10BT</c> despite the
/// config declaring <c>bfloat16</c>. The other enum members exist so the
/// reader can surface a structured "unsupported dtype" diagnostic rather
/// than throwing at the JSON-parse layer, and so Stage D3 can add bf16
/// without reshaping the API.
/// </para>
/// </remarks>
public enum SafetensorsDType
{
    /// <summary>Default sentinel — dtype token was absent or unknown.</summary>
    Unknown = 0,

    /// <summary>IEEE-754 binary32. Matches <c>"F32"</c>.</summary>
    F32,

    /// <summary>IEEE-754 binary16. Matches <c>"F16"</c>.</summary>
    F16,

    /// <summary>Brain-float (truncated float32). Matches <c>"BF16"</c>.</summary>
    BF16,

    /// <summary>Double-precision float. Matches <c>"F64"</c>.</summary>
    F64,

    /// <summary>Signed 8-bit integer. Matches <c>"I8"</c>.</summary>
    I8,

    /// <summary>Unsigned 8-bit integer. Matches <c>"U8"</c>.</summary>
    U8,

    /// <summary>Signed 16-bit integer. Matches <c>"I16"</c>.</summary>
    I16,

    /// <summary>Signed 32-bit integer. Matches <c>"I32"</c>.</summary>
    I32,

    /// <summary>Signed 64-bit integer. Matches <c>"I64"</c>.</summary>
    I64,

    /// <summary>Boolean (1 byte per element). Matches <c>"BOOL"</c>.</summary>
    Bool,
}

/// <summary>
/// Parsing/formatting helpers for <see cref="SafetensorsDType"/>.
/// </summary>
public static class SafetensorsDTypeExtensions
{
    /// <summary>
    /// Parses a safetensors dtype token (case-insensitive) into the
    /// corresponding <see cref="SafetensorsDType"/>. Returns
    /// <see cref="SafetensorsDType.Unknown"/> for any unrecognised token.
    /// </summary>
    public static SafetensorsDType Parse(string token) => token switch
    {
        "F32" or "f32" => SafetensorsDType.F32,
        "F16" or "f16" => SafetensorsDType.F16,
        "BF16" or "bf16" => SafetensorsDType.BF16,
        "F64" or "f64" => SafetensorsDType.F64,
        "I8" or "i8" => SafetensorsDType.I8,
        "U8" or "u8" => SafetensorsDType.U8,
        "I16" or "i16" => SafetensorsDType.I16,
        "I32" or "i32" => SafetensorsDType.I32,
        "I64" or "i64" => SafetensorsDType.I64,
        "BOOL" or "bool" => SafetensorsDType.Bool,
        _ => SafetensorsDType.Unknown,
    };

    /// <summary>
    /// Size, in bytes, of a single element of the given dtype. Returns
    /// <c>0</c> for <see cref="SafetensorsDType.Unknown"/>.
    /// </summary>
    public static int ElementSizeInBytes(this SafetensorsDType dtype) => dtype switch
    {
        SafetensorsDType.F32 => 4,
        SafetensorsDType.F16 => 2,
        SafetensorsDType.BF16 => 2,
        SafetensorsDType.F64 => 8,
        SafetensorsDType.I8 => 1,
        SafetensorsDType.U8 => 1,
        SafetensorsDType.I16 => 2,
        SafetensorsDType.I32 => 4,
        SafetensorsDType.I64 => 8,
        SafetensorsDType.Bool => 1,
        _ => 0,
    };
}
