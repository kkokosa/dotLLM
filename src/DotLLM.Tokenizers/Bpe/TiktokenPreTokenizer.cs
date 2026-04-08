using System.Text.RegularExpressions;

namespace DotLLM.Tokenizers.Bpe;

/// <summary>
/// Maps GGUF <c>tokenizer.ggml.pre</c> type names to compiled regex patterns
/// used for pre-tokenization in tiktoken-style BPE encodings.
/// Pre-tokenization splits input text at word/punctuation boundaries before BPE
/// merges are applied, ensuring merges do not cross segment boundaries.
/// </summary>
/// <remarks>
/// Patterns are sourced from llama.cpp's <c>llama_vocab</c> (authoritative reference).
/// Each pattern is compiled once and reused across all tokenizer instances.
/// </remarks>
internal static class TiktokenPreTokenizer
{
    // ── GPT-2 / default ─────────────────────────────────────────────
    // Contractions, letter runs, digit runs, punctuation, trailing whitespace.
    private static readonly Regex Gpt2Regex = new(
        @"(?:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    // ── Llama 3 / llama-bpe ─────────────────────────────────────────
    // Case-insensitive contractions, optional-punct + letters, 1-3 digit groups,
    // punctuation with trailing newlines, standalone newlines, trailing whitespace.
    private static readonly Regex Llama3Regex = new(
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    // ── DeepSeek LLM ────────────────────────────────────────────────
    // Same as GPT-2 with additional CJK character class support.
    private static readonly Regex DeepSeekLlmRegex = new(
        @"(?:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    // ── DeepSeek Coder ──────────────────────────────────────────────
    // Identifiers, multi-digit numbers, non-whitespace runs, whitespace groups.
    private static readonly Regex DeepSeekCoderRegex = new(
        @"[a-zA-Z_][a-zA-Z0-9_]*|\p{N}+| ?[^\s\w]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    // ── Command-R ───────────────────────────────────────────────────
    // Similar to Llama 3 pattern.
    private static readonly Regex CommandRRegex = new(
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    /// <summary>
    /// Returns the pre-tokenization regex for the given GGUF <c>tokenizer.ggml.pre</c> type,
    /// or <c>null</c> if the type is unknown or absent (no pre-tokenization).
    /// </summary>
    internal static Regex? GetRegex(string? preType) => preType switch
    {
        "default" or "gpt2" => Gpt2Regex,
        "llama3" or "llama-bpe" => Llama3Regex,
        "deepseek-llm" => DeepSeekLlmRegex,
        "deepseek-coder" => DeepSeekCoderRegex,
        "command-r" => CommandRRegex,
        _ => null,
    };
}
