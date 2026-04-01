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

    /// <summary>
    /// Constrain output to match a regex pattern (compiled to minimized DFA).
    /// Entire output must match — implicit anchoring. Use cases: dates, phone numbers, enums.
    /// </summary>
    public sealed record Regex : ResponseFormat
    {
        /// <summary>The regex pattern. Entire output must match (implicit anchoring).</summary>
        public required string Pattern { get; init; }
    }

    /// <summary>
    /// Constrain output to conform to a GBNF grammar (compiled to pushdown automaton).
    /// GBNF is the grammar format used by llama.cpp.
    /// </summary>
    public sealed record Grammar : ResponseFormat
    {
        /// <summary>The GBNF grammar definition.</summary>
        public required string GbnfGrammar { get; init; }
    }
}
