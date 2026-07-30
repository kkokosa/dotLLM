using System.Text;
using BenchmarkDotNet.Attributes;
using DotLLM.Tokenizers;

namespace DotLLM.Benchmarks;

/// <summary>
/// Compares legacy dictionary-per-node trie layout with the flat edge-array trie.
/// The vocabulary generator builds SentencePiece-like token strings at real-model scale.
/// </summary>
[MemoryDiagnoser]
[InProcess]
[SimpleJob(warmupCount: 1, iterationCount: 3)]
public class TrieBenchmarks
{
    private Trie _flatTrie = null!;
    private LegacyTrie _legacyTrie = null!;
    private string[] _vocab = null!;
    private string[] _lookupInputs = null!;

    [Params(32_000)]
    public int VocabSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _vocab = BuildRealVocab(VocabSize);
        _lookupInputs = BuildLookupInputs(_vocab);

        _flatTrie = new Trie();
        _legacyTrie = new LegacyTrie();

        for (int i = 0; i < _vocab.Length; i++)
        {
            string token = _vocab[i];
            float score = -i;
            _flatTrie.Add(token, i, score);
            _legacyTrie.Add(token, i, score);
        }
    }

    [Benchmark(Baseline = true)]
    public int LookupLegacy()
    {
        int checksum = 0;
        foreach (string input in _lookupInputs)
        {
            if (_legacyTrie.TryMatchLongest(input, out int tokenId, out _, out int length))
                checksum ^= tokenId + length;
        }
        return checksum;
    }

    [Benchmark]
    public int LookupFlat()
    {
        int checksum = 0;
        foreach (string input in _lookupInputs)
        {
            if (_flatTrie.TryMatchLongest(input, out int tokenId, out _, out int length))
                checksum ^= tokenId + length;
        }
        return checksum;
    }

    [Benchmark]
    public int BuildLegacy()
    {
        var trie = new LegacyTrie();
        for (int i = 0; i < _vocab.Length; i++)
            trie.Add(_vocab[i], i, -i);
        trie.TryMatchLongest(_vocab[0], out int tokenId, out _, out int length);
        return tokenId + length;
    }

    [Benchmark]
    public int BuildFlat()
    {
        var trie = new Trie();
        for (int i = 0; i < _vocab.Length; i++)
            trie.Add(_vocab[i], i, -i);
        trie.TryMatchLongest(_vocab[0], out _, out _, out _); // force flattening
        trie.TryMatchLongest(_vocab[1], out int tokenId, out _, out int length);
        return tokenId + length;
    }

    private static string[] BuildRealVocab(int vocabSize)
    {
        var rng = new Random(42);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var vocab = new List<string>(vocabSize);

        AddToken("<unk>");
        AddToken("\u2581");
        for (int b = 0; b < 256 && vocab.Count < vocabSize; b++)
            AddToken($"<0x{b:X2}>");

        string[] syllables =
        [
            "a", "an", "ar", "as", "at", "be", "ca", "co", "de", "di", "en", "er", "es", "ex", "for",
            "gen", "ing", "ion", "is", "la", "le", "li", "ll", "lo", "ma", "ment", "na", "ne", "net",
            "on", "or", "out", "per", "pre", "pro", "ra", "re", "ri", "ro", "s", "se", "sh", "st",
            "t", "te", "th", "tion", "to", "tr", "un", "ver", "with", "x", "y", "z"
        ];

        while (vocab.Count < vocabSize)
        {
            int partCount = 1 + rng.Next(4);
            bool startsWord = rng.NextDouble() < 0.55;

            var builder = new StringBuilder(16);
            if (startsWord)
                builder.Append('\u2581');

            for (int i = 0; i < partCount; i++)
                builder.Append(syllables[rng.Next(syllables.Length)]);

            if (rng.NextDouble() < 0.12)
                builder.Append((char)('0' + rng.Next(10)));
            if (rng.NextDouble() < 0.08)
                builder.Append((char)('a' + rng.Next(26)));

            AddToken(builder.ToString());
        }

        return [.. vocab];

        void AddToken(string token)
        {
            if (seen.Add(token))
                vocab.Add(token);
        }
    }

    private static string[] BuildLookupInputs(string[] vocab)
    {
        var rng = new Random(123);
        var inputs = new string[8_192];

        for (int i = 0; i < inputs.Length; i++)
        {
            if ((i & 3) != 0)
            {
                string token = vocab[rng.Next(vocab.Length)];
                char suffix = (char)('a' + rng.Next(26));
                inputs[i] = token + suffix;
            }
            else
            {
                inputs[i] = $"zzz_{i}_{rng.Next(10_000)}";
            }
        }

        return inputs;
    }

    private sealed class LegacyTrie
    {
        private readonly LegacyTrieNode _root = new();

        public void Add(ReadOnlySpan<char> key, int tokenId, float score)
        {
            LegacyTrieNode node = _root;
            foreach (char c in key)
            {
                node.Children ??= [];
                if (!node.Children.TryGetValue(c, out LegacyTrieNode? child))
                {
                    child = new LegacyTrieNode();
                    node.Children[c] = child;
                }
                node = child;
            }

            node.TokenId = tokenId;
            node.Score = score;
        }

        public bool TryMatchLongest(ReadOnlySpan<char> text, out int tokenId, out float score, out int matchLength)
        {
            LegacyTrieNode node = _root;
            int bestLen = 0;
            int bestId = -1;
            float bestScore = 0f;

            for (int i = 0; i < text.Length; i++)
            {
                if (node.Children == null || !node.Children.TryGetValue(text[i], out LegacyTrieNode? next))
                    break;

                node = next;
                if (node.TokenId >= 0)
                {
                    bestLen = i + 1;
                    bestId = node.TokenId;
                    bestScore = node.Score;
                }
            }

            if (bestLen == 0)
            {
                tokenId = -1;
                score = 0f;
                matchLength = 0;
                return false;
            }

            tokenId = bestId;
            score = bestScore;
            matchLength = bestLen;
            return true;
        }
    }

    private sealed class LegacyTrieNode
    {
        public Dictionary<char, LegacyTrieNode>? Children;
        public int TokenId = -1;
        public float Score;
    }
}
