using System.Text.RegularExpressions;

namespace DotLLM.Tokenizers.Bpe;

/// <summary>
/// Maps GGUF <c>tokenizer.ggml.pre</c> type names to compiled regex pipelines
/// used for pre-tokenization in tiktoken-style BPE encodings.
/// Pre-tokenization splits input text at word/punctuation boundaries before BPE
/// merges are applied, ensuring merges do not cross segment boundaries.
/// </summary>
/// <remarks>
/// <para>Patterns are sourced from llama.cpp's <c>llama_vocab</c> (authoritative reference).
/// Each pattern is compiled once and reused across all tokenizer instances.</para>
/// <para><b>A pre-type maps to an ordered pipeline, not a single expression.</b> llama.cpp's
/// <c>regex_exprs</c> is a list applied in sequence: each expression further splits the segments
/// produced by the previous one. Several pre-types genuinely need more than one stage — the
/// StarCoder/SmolLM family isolates every digit with <c>\p{N}</c> before applying its main
/// pattern — so collapsing a pipeline to its last expression silently mis-tokenizes.</para>
/// </remarks>
internal static class TiktokenPreTokenizer
{
    // ── GPT-2 / default ─────────────────────────────────────────────
    // Contractions, letter runs, digit runs, punctuation, trailing whitespace.
    private static readonly Regex[] Gpt2Pipeline =
    [
        new(@"(?:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled),
    ];

    // ── Llama 3 / llama-bpe ─────────────────────────────────────────
    // Case-insensitive contractions, optional-punct + letters, 1-3 digit groups,
    // punctuation with trailing newlines, standalone newlines, trailing whitespace.
    private static readonly Regex[] Llama3Pipeline =
    [
        new(@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled),
    ];

    // ── StarCoder / SmolLM family ───────────────────────────────────
    // Two stages, in order: isolate every digit, then the GPT-2 pattern WITHOUT its
    // trailing `|\s+` alternative. Shared by StarCoder, Refact, Command-R, SmolLM,
    // CodeShell, EXAONE, Minerva and Mellum2 — llama.cpp falls all eight through one
    // case block.
    private static readonly Regex[] StarCoderPipeline =
    [
        new(@"\p{N}", RegexOptions.Compiled),
        new(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)",
            RegexOptions.Compiled),
    ];

    // ── DeepSeek LLM ────────────────────────────────────────────────
    private static readonly Regex[] DeepSeekLlmPipeline =
    [
        new(@"(?:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled),
    ];

    // ── DeepSeek Coder ──────────────────────────────────────────────
    // Identifiers, multi-digit numbers, non-whitespace runs, whitespace groups.
    private static readonly Regex[] DeepSeekCoderPipeline =
    [
        new(@"[a-zA-Z_][a-zA-Z0-9_]*|\p{N}+| ?[^\s\w]+|\s+(?!\S)|\s+", RegexOptions.Compiled),
    ];

    /// <summary>
    /// Returns the ordered pre-tokenization regex pipeline for the given GGUF
    /// <c>tokenizer.ggml.pre</c> type, or <c>null</c> if the type is unknown or absent.
    /// </summary>
    /// <remarks>
    /// A <c>null</c> result means the caller performs <b>no</b> pre-tokenization and BPE merges
    /// run across the whole input. That is rarely what an unrecognized name should mean: merges
    /// then cross boundaries the model was trained to respect, producing a token stream that
    /// mostly matches the reference and diverges at a small number of sites — the failure mode
    /// that motivated this table (see issue #237).
    /// </remarks>
    internal static Regex[]? GetRegexes(string? preType) => preType switch
    {
        "default" or "gpt2" => Gpt2Pipeline,
        "llama3" or "llama-bpe" => Llama3Pipeline,
        "starcoder" or "refact" or "command-r" or "smollm"
            or "codeshell" or "exaone" or "minerva" or "mellum2" => StarCoderPipeline,
        "deepseek-llm" => DeepSeekLlmPipeline,
        "deepseek-coder" => DeepSeekCoderPipeline,
        _ => null,
    };
}
