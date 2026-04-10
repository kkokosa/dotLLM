using System.Text;
using BenchmarkDotNet.Attributes;
using DotLLM.Tokenizers.Bpe;

namespace DotLLM.Benchmarks;

/// <summary>
/// Benchmarks for <see cref="BpeTokenizer.Encode"/> on text containing special tokens.
/// Measures the cost of special-token pre-splitting (<see cref="BpeTokenizer"/>'s
/// trie-based scan replacing the previous O(n × m) linear search).
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class SpecialTokenEncodeBenchmarks
{
    private BpeTokenizer _tokenizer = null!;
    private string _text = null!;

    /// <summary>Number of characters in the input text.</summary>
    [Params(1024, 8192)]
    public int TextLength { get; set; }

    /// <summary>Number of distinct special tokens in the vocabulary.</summary>
    [Params(5, 20, 100)]
    public int SpecialTokenCount { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _tokenizer = BuildTokenizerWithSpecials(SpecialTokenCount);
        _text = BuildText(TextLength, SpecialTokenCount);
    }

    [Benchmark]
    public int EncodeWithSpecials()
    {
        int[] ids = _tokenizer.Encode(_text);
        return ids.Length;
    }

    /// <summary>
    /// Builds a SentencePiece-style tokenizer with <paramref name="specialCount"/> control
    /// tokens. The normal vocabulary covers ASCII chars and a handful of multi-char merges.
    /// </summary>
    private static BpeTokenizer BuildTokenizerWithSpecials(int specialCount)
    {
        var tokens = new List<string> { "<unk>" };
        var scores = new List<float> { 0f };
        var tokenTypes = new List<int> { 1 }; // 1 = normal

        // ASCII printable chars — enough for BPE segment encoding.
        for (char c = ' '; c <= '~'; c++)
        {
            tokens.Add(c.ToString());
            scores.Add(-1f);
            tokenTypes.Add(1);
        }

        // Synthetic special tokens: <|special_0|>, <|special_1|>, …
        // Type 3 = control → pre-split before BPE.
        for (int i = 0; i < specialCount; i++)
        {
            tokens.Add($"<|special_{i}|>");
            scores.Add(0f);
            tokenTypes.Add(3);
        }

        return BpeTokenizer.CreateSentencePiece(
            [.. tokens],
            [.. scores],
            [.. tokenTypes],
            bosId: 0,
            eosId: 0,
            addBosSpace: false);
    }

    /// <summary>
    /// Builds <paramref name="length"/> characters of text with special tokens sprinkled
    /// at regular intervals. This exercises both the trie match path and the
    /// FindNextSpecialToken path (scanning through text between specials).
    /// </summary>
    private static string BuildText(int length, int specialCount)
    {
        var sb = new StringBuilder(length + 1024);
        var rng = new Random(42);

        // Place a special token roughly every 256 chars.
        int charsWritten = 0;
        while (charsWritten < length)
        {
            int runLength = Math.Min(256, length - charsWritten);
            for (int i = 0; i < runLength; i++)
            {
                sb.Append((char)('a' + rng.Next(0, 26)));
                charsWritten++;
            }
            if (charsWritten < length && specialCount > 0)
            {
                sb.Append($"<|special_{rng.Next(0, specialCount)}|>");
                // Don't count the special's chars against TextLength — we want a consistent
                // count of "real" text for comparison across SpecialTokenCount params.
            }
        }
        return sb.ToString();
    }
}
