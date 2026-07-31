using DotLLM.Core.Configuration;
using DotLLM.Core.PositionEncoding;

namespace DotLLM.Core.Models;

/// <summary>
/// Complete configuration for a transformer model architecture. Populated from GGUF metadata or explicit construction.
/// A single <see cref="ModelConfig"/> parameterizes the transformer block to handle Llama/Mistral/Phi/Qwen/DeepSeek.
/// </summary>
public record ModelConfig
{
    /// <summary>Model architecture family.</summary>
    public required Architecture Architecture { get; init; }

    /// <summary>Vocabulary size (number of token embeddings).</summary>
    public required int VocabSize { get; init; }

    /// <summary>Hidden size (embedding dimension).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>FFN intermediate dimension.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Number of transformer layers.</summary>
    public required int NumLayers { get; init; }

    /// <summary>Number of attention heads for queries.</summary>
    public required int NumAttentionHeads { get; init; }

    /// <summary>Number of KV heads. Equal to <see cref="NumAttentionHeads"/> for MHA, 1 for MQA, between for GQA.</summary>
    public required int NumKvHeads { get; init; }

    /// <summary>Dimension per attention head. Typically <see cref="HiddenSize"/> / <see cref="NumAttentionHeads"/>.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Maximum supported sequence length.</summary>
    public required int MaxSequenceLength { get; init; }

    /// <summary>Attention mechanism type (GQA or MLA).</summary>
    public AttentionType AttentionType { get; init; } = AttentionType.GQA;

    /// <summary>Positional encoding type.</summary>
    public PositionEncodingType PositionEncodingType { get; init; } = PositionEncodingType.RoPE;

    /// <summary>RoPE-specific configuration. Null when not using RoPE.</summary>
    public RoPEConfig? RoPEConfig { get; init; }

    /// <summary>Activation function used in FFN layers.</summary>
    public ActivationFunction ActivationFunction { get; init; } = ActivationFunction.SiLU;

    /// <summary>Normalization layer type.</summary>
    public NormType NormType { get; init; } = NormType.RMSNorm;

    /// <summary>Epsilon for normalization layers.</summary>
    public float NormEpsilon { get; init; } = 1e-5f;

    /// <summary>Whether input and output embeddings share weights.</summary>
    public bool TiedEmbeddings { get; init; }

    /// <summary>Sliding window size for local attention. Null = full attention.</summary>
    public int? SlidingWindowSize { get; init; }

    /// <summary>MLA configuration. Only set for DeepSeek-style MLA attention.</summary>
    public MlaConfig? MlaConfig { get; init; }

    /// <summary>Jinja2 chat template from model metadata. Null if not present.</summary>
    public string? ChatTemplate { get; init; }

    /// <summary>
    /// Validates internal consistency of the configuration. Throws
    /// <see cref="InvalidModelConfigException"/> with a precise message for the
    /// first violation found. Loaders should call this after constructing a
    /// <see cref="ModelConfig"/> from GGUF / HF metadata so the failure is
    /// reported at load time, not deep inside kernel code where the misconfigured
    /// value would produce a segfault or silent garbage output.
    /// </summary>
    /// <exception cref="InvalidModelConfigException">Thrown on the first validation failure.</exception>
    public void Validate()
    {
        // Positive scalar guards.
        if (VocabSize <= 0) throw new InvalidModelConfigException(nameof(VocabSize), $"must be positive (got {VocabSize}).");
        if (HiddenSize <= 0) throw new InvalidModelConfigException(nameof(HiddenSize), $"must be positive (got {HiddenSize}).");
        if (IntermediateSize <= 0) throw new InvalidModelConfigException(nameof(IntermediateSize), $"must be positive (got {IntermediateSize}).");
        if (NumLayers <= 0) throw new InvalidModelConfigException(nameof(NumLayers), $"must be positive (got {NumLayers}).");
        if (NumAttentionHeads <= 0) throw new InvalidModelConfigException(nameof(NumAttentionHeads), $"must be positive (got {NumAttentionHeads}).");
        if (NumKvHeads <= 0) throw new InvalidModelConfigException(nameof(NumKvHeads), $"must be positive (got {NumKvHeads}).");
        if (HeadDim <= 0) throw new InvalidModelConfigException(nameof(HeadDim), $"must be positive (got {HeadDim}).");
        if (MaxSequenceLength <= 0) throw new InvalidModelConfigException(nameof(MaxSequenceLength), $"must be positive (got {MaxSequenceLength}).");

        // GQA invariant — NumAttentionHeads must be an integer multiple of NumKvHeads.
        // Each KV head services NumAttentionHeads / NumKvHeads query heads; a non-integer
        // ratio breaks the head-grouping math in the attention kernel.
        if (NumAttentionHeads % NumKvHeads != 0)
        {
            throw new InvalidModelConfigException(
                nameof(NumKvHeads),
                $"must divide {nameof(NumAttentionHeads)}: " +
                $"{NumKvHeads} does not divide {NumAttentionHeads}.");
        }

        // HeadDim is required in the GGUF metadata but is otherwise redundant with
        // HiddenSize / NumAttentionHeads. If both are supplied they must agree; a
        // mismatch produces silently wrong attention dims rather than a hard failure.
        int derivedHeadDim = HiddenSize / NumAttentionHeads;
        if (HiddenSize % NumAttentionHeads == 0 && derivedHeadDim != HeadDim)
        {
            throw new InvalidModelConfigException(
                nameof(HeadDim),
                $"({HeadDim}) does not match {nameof(HiddenSize)} / {nameof(NumAttentionHeads)} " +
                $"({HiddenSize} / {NumAttentionHeads} = {derivedHeadDim}).");
        }

        // RoPE config presence must match the position encoding type. A RoPE config
        // present on a non-RoPE model is at best ignored, at worst silently consumed
        // by a code path that no longer applies — confusing to debug.
        if (PositionEncodingType == PositionEncodingType.RoPE && RoPEConfig is null)
        {
            throw new InvalidModelConfigException(
                nameof(RoPEConfig),
                $"is required when {nameof(PositionEncodingType)} is RoPE.");
        }
        if (PositionEncodingType != PositionEncodingType.RoPE && RoPEConfig is not null)
        {
            throw new InvalidModelConfigException(
                nameof(RoPEConfig),
                $"must be null when {nameof(PositionEncodingType)} is {PositionEncodingType} (only RoPE accepts it).");
        }
    }
}

/// <summary>
/// Thrown when a <see cref="ModelConfig"/> fails internal-consistency validation.
/// </summary>
public sealed class InvalidModelConfigException : Exception
{
    /// <summary>Name of the field whose value is invalid.</summary>
    public string FieldName { get; }

    /// <summary>
    /// Creates a new exception describing the invalid field and the reason.
    /// </summary>
    /// <param name="fieldName">Name of the offending <see cref="ModelConfig"/> field.</param>
    /// <param name="reason">Human-readable reason the field is invalid.</param>
    public InvalidModelConfigException(string fieldName, string reason)
        : base($"{nameof(ModelConfig)}.{fieldName} {reason}")
    {
        FieldName = fieldName;
    }

    /// <summary>
    /// Creates a new exception describing the invalid field and the reason, wrapping the
    /// lower-level failure that surfaced it. Use this when a loader detects the invalid
    /// value while handling another exception (e.g. a metadata parse or conversion error),
    /// so the original stack and context are not lost.
    /// </summary>
    /// <param name="fieldName">Name of the offending <see cref="ModelConfig"/> field.</param>
    /// <param name="reason">Human-readable reason the field is invalid.</param>
    /// <param name="innerException">The exception that caused this validation failure.</param>
    public InvalidModelConfigException(string fieldName, string reason, Exception innerException)
        : base($"{nameof(ModelConfig)}.{fieldName} {reason}", innerException)
    {
        FieldName = fieldName;
    }
}
