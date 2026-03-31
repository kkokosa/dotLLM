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

    /// <summary>
    /// Constrain output to conform to a specific JSON Schema.
    /// Corresponds to <c>response_format: {"type": "json_schema", "json_schema": {...}}</c> in the OpenAI API.
    /// </summary>
    public sealed record JsonSchema : ResponseFormat
    {
        /// <summary>The JSON Schema definition as a JSON string.</summary>
        public required string Schema { get; init; }

        /// <summary>Optional name for the schema (used in API responses).</summary>
        public string? Name { get; init; }

        /// <summary>Whether to enforce strict schema adherence (default true).</summary>
        public bool Strict { get; init; } = true;
    }
}
