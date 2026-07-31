using System;
using DotLLM.Core.Configuration;

namespace DotLLM.Core.PositionEncoding;

/// <summary>
/// Configuration for Rotary Position Embeddings (RoPE).
/// </summary>
/// <param name="Theta">Base frequency. Default 10000.0, Llama 3 uses 500000.0.</param>
/// <param name="DimensionCount">Number of dimensions for the rotation.</param>
/// <param name="Type">Element-pairing convention. Must match GGUF Q/K weight layout.</param>
/// <param name="ScalingType">Context-length scaling strategy.</param>
/// <param name="ScalingFactor">Scaling factor for Linear/NTK methods.</param>
/// <param name="OrigMaxSeqLen">Original max sequence length before scaling.</param>
/// <param name="AttnFactor">YaRN attention factor.</param>
/// <param name="BetaFast">YaRN beta-fast parameter.</param>
/// <param name="BetaSlow">YaRN beta-slow parameter.</param>
public readonly record struct RoPEConfig(
    float Theta = 10000.0f,
    int DimensionCount = 0,
    RoPEType Type = RoPEType.Norm,
    RoPEScalingType ScalingType = RoPEScalingType.None,
    float ScalingFactor = 1.0f,
    int OrigMaxSeqLen = 0,
    float AttnFactor = 1.0f,
    float BetaFast = 32.0f,
    float BetaSlow = 1.0f)
{
    /// <summary>Default YaRN attention factor — see field default on the record itself.</summary>
    internal const float DefaultAttnFactor = 1.0f;
    /// <summary>Default YaRN beta-fast — see field default on the record itself.</summary>
    internal const float DefaultBetaFast = 32.0f;
    /// <summary>Default YaRN beta-slow — see field default on the record itself.</summary>
    internal const float DefaultBetaSlow = 1.0f;

    /// <summary>
    /// Validates cross-field consistency between <see cref="ScalingType"/> and the
    /// YaRN-specific fields (<see cref="AttnFactor"/>, <see cref="BetaFast"/>,
    /// <see cref="BetaSlow"/>). Setting YaRN parameters under a non-YaRN scaling
    /// strategy is almost certainly a configuration mistake — those fields are only
    /// consumed by the YaRN math path; under other strategies they are silently
    /// ignored, which is confusing to debug.
    /// </summary>
    /// <exception cref="InvalidRoPEConfigException">
    /// Thrown when a YaRN-only field is set to a non-default value while
    /// <see cref="ScalingType"/> is anything other than <see cref="RoPEScalingType.YaRN"/>.
    /// </exception>
    public void Validate()
    {
        if (ScalingType == RoPEScalingType.YaRN)
        {
            // YaRN actively consumes the YaRN fields — any value is acceptable.
            return;
        }

        if (!IsDefault(AttnFactor, DefaultAttnFactor))
        {
            throw new InvalidRoPEConfigException(nameof(AttnFactor), Reason(AttnFactor, ScalingType));
        }
        if (!IsDefault(BetaFast, DefaultBetaFast))
        {
            throw new InvalidRoPEConfigException(nameof(BetaFast), Reason(BetaFast, ScalingType));
        }
        if (!IsDefault(BetaSlow, DefaultBetaSlow))
        {
            throw new InvalidRoPEConfigException(nameof(BetaSlow), Reason(BetaSlow, ScalingType));
        }
    }

    // Invariant culture so the message reads the same regardless of the ambient
    // culture's decimal separator — these strings end up in logs and bug reports.
    private static string Reason(float value, RoPEScalingType scalingType) =>
        FormattableString.Invariant(
            $"is set to {value} but {nameof(ScalingType)} is {scalingType} (only YaRN consumes YaRN-specific fields).");

    // Bit-exact comparison against the compile-time default literal. This cannot (and
    // does not try to) tell whether the caller explicitly passed the default value —
    // it only answers "does this field still hold the documented default bit pattern".
    // Comparing bits rather than using `==` keeps the check exact and total: no
    // tolerance to pick, and no float-equality edge cases (NaN, ±0) to reason about.
    private static bool IsDefault(float actual, float expected) =>
        BitConverter.SingleToInt32Bits(actual) == BitConverter.SingleToInt32Bits(expected);
}

/// <summary>
/// Thrown when a <see cref="RoPEConfig"/> field combination is internally inconsistent —
/// most commonly a YaRN-only field set to a non-default value under a non-YaRN scaling strategy.
/// </summary>
public sealed class InvalidRoPEConfigException : Exception
{
    /// <summary>Name of the offending <see cref="RoPEConfig"/> field.</summary>
    public string FieldName { get; }

    /// <summary>
    /// Creates a new exception describing the invalid field and the reason.
    /// </summary>
    /// <param name="fieldName">Name of the offending <see cref="RoPEConfig"/> field.</param>
    /// <param name="reason">Human-readable reason the field is invalid.</param>
    public InvalidRoPEConfigException(string fieldName, string reason)
        : base($"{nameof(RoPEConfig)}.{fieldName} {reason}")
    {
        FieldName = fieldName;
    }
}
