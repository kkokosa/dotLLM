namespace DotLLM.Core.Configuration;

/// <summary>
/// Specifies the desired response format for generation.
/// Controls whether constrained decoding is applied.
/// </summary>
public abstract record ResponseFormat
{
    /// <summary>No constraint — free-form text generation (default).</summary>
    public sealed record Text : ResponseFormat;

    /// <summary>
    /// Constrain output to syntactically valid JSON (RFC 8259).
    /// Corresponds to <c>response_format: {"type": "json_object"}</c> in the OpenAI API.
    /// </summary>
    public sealed record JsonObject : ResponseFormat;
}
