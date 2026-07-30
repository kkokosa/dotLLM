using System.Diagnostics;
using System.Text.Json;
using System.Text;
using DotLLM.Tokenizers;

namespace DotLLM.Benchmarks;

internal static class TrieComparisonRunner
{
    public static void Run(string? tokenizerJsonPath = null)
    {
        string scenarioName;
        string[] vocab;
        if (!string.IsNullOrWhiteSpace(tokenizerJsonPath))
        {
            vocab = LoadVocabFromTokenizerJson(tokenizerJsonPath);
            scenarioName = $"real tokenizer vocab from {Path.GetFileName(tokenizerJsonPath)}";
        }
        else
        {
            const int vocabSize = 32_000;
            vocab = BuildRealVocab(vocabSize);
            scenarioName = "synthetic 32k SentencePiece-like vocab";
        }

        string[] lookupInputs = BuildLookupInputs(vocab);

        var legacy = new LegacyTrie();
        var flatTrie = new Trie();
        var flat = new FlatTrieAdapter(flatTrie);

        for (int i = 0; i < vocab.Length; i++)
        {
            string token = vocab[i];
            float score = -i;
            legacy.Add(token, i, score);
            flat.Add(token, i, score);
        }

        // Exclude one-time initialization from throughput timing:
        // force flat trie freeze and touch both implementations once before warmup/measure.
        flatTrie.TryMatchLongest(vocab[0], out _, out _, out _);
        legacy.TryMatchLongest(vocab[0], out _, out _, out _);

        ValidateSemantics(legacy, flat, lookupInputs);

        // Warm up JIT/caches.
        _ = MeasureLookup(legacy, lookupInputs, iterations: 5);
        _ = MeasureLookup(flat, lookupInputs, iterations: 5);

        double legacyOpsPerSec = MeasureLookup(legacy, lookupInputs, iterations: 25);
        double flatOpsPerSec = MeasureLookup(flat, lookupInputs, iterations: 25);
        long legacyBytes = MeasureRetainedBytes(vocab, buildFlat: false);
        long flatBytes = MeasureRetainedBytes(vocab, buildFlat: true);

        Console.WriteLine($"Trie comparison ({scenarioName}, {vocab.Length:N0} tokens)");
        Console.WriteLine("Methodology: lookup throughput excludes one-time freeze/initialization; memory reports retained managed bytes after construction.");
        Console.WriteLine($"Lookup throughput (ops/s): legacy={legacyOpsPerSec:N0}, flat={flatOpsPerSec:N0}, speedup={flatOpsPerSec / legacyOpsPerSec:N2}x");
        Console.WriteLine($"Retained managed memory: legacy={legacyBytes / (1024d * 1024d):N2} MB, flat={flatBytes / (1024d * 1024d):N2} MB, reduction={(1d - (double)flatBytes / legacyBytes) * 100d:N1}%");
    }

    private static string[] LoadVocabFromTokenizerJson(string tokenizerJsonPath)
    {
        string json = File.ReadAllText(tokenizerJsonPath);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        JsonElement vocabElement;
        if (root.TryGetProperty("model", out JsonElement model) && model.TryGetProperty("vocab", out JsonElement modelVocab))
            vocabElement = modelVocab;
        else if (root.TryGetProperty("vocab", out JsonElement directVocab))
            vocabElement = directVocab;
        else
            throw new InvalidOperationException("Tokenizer JSON does not contain a recognized vocab structure.");

        var pairs = new List<(int Id, string Token)>();
        foreach (JsonProperty entry in vocabElement.EnumerateObject())
        {
            int id = entry.Value.ValueKind switch
            {
                JsonValueKind.Number => entry.Value.GetInt32(),
                JsonValueKind.String when int.TryParse(entry.Value.GetString(), out int parsed) => parsed,
                _ => throw new InvalidOperationException($"Unsupported token id format for token '{entry.Name}'.")
            };
            pairs.Add((id, entry.Name));
        }

        pairs.Sort(static (left, right) => left.Id.CompareTo(right.Id));

        var vocab = new string[pairs.Count];
        for (int i = 0; i < pairs.Count; i++)
            vocab[i] = pairs[i].Token;

        return vocab;
    }

    private static double MeasureLookup(ITrieLike trie, string[] lookupInputs, int iterations)
    {
        var sw = Stopwatch.StartNew();
        int checksum = 0;
        for (int it = 0; it < iterations; it++)
        {
            foreach (string input in lookupInputs)
            {
                if (trie.TryMatchLongest(input, out int tokenId, out _, out int length))
                    checksum ^= tokenId + length;
            }
        }
        sw.Stop();
        GC.KeepAlive(checksum);

        return iterations * lookupInputs.Length / sw.Elapsed.TotalSeconds;
    }

    private static long MeasureRetainedBytes(string[] vocab, bool buildFlat)
    {
        var samples = new long[5];
        for (int s = 0; s < samples.Length; s++)
        {
            ForceGc();
            long before = GC.GetTotalMemory(forceFullCollection: true);

            object trie;
            if (buildFlat)
            {
                var t = new Trie();
                for (int i = 0; i < vocab.Length; i++)
                    t.Add(vocab[i], i, -i);
                t.TryMatchLongest(vocab[0], out _, out _, out _);
                trie = t;
            }
            else
            {
                var t = new LegacyTrie();
                for (int i = 0; i < vocab.Length; i++)
                    t.Add(vocab[i], i, -i);
                trie = t;
            }

            long after = GC.GetTotalMemory(forceFullCollection: true);
            samples[s] = Math.Max(0, after - before);
            GC.KeepAlive(trie);
        }

        Array.Sort(samples);
        return samples[samples.Length / 2];
    }

    private static void ValidateSemantics(ITrieLike legacy, ITrieLike flat, string[] lookupInputs)
    {
        foreach (string input in lookupInputs)
        {
            bool legacyMatch = legacy.TryMatchLongest(input, out int legacyTokenId, out float legacyScore, out int legacyLength);
            bool flatMatch = flat.TryMatchLongest(input, out int flatTokenId, out float flatScore, out int flatLength);

            if (legacyMatch != flatMatch ||
                legacyTokenId != flatTokenId ||
                legacyLength != flatLength ||
                legacyScore != flatScore)
            {
                throw new InvalidOperationException($"Trie semantics mismatch for input '{input}'.");
            }
        }
    }

    private static void ForceGc()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
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

    private interface ITrieLike
    {
        void Add(ReadOnlySpan<char> key, int tokenId, float score);
        bool TryMatchLongest(ReadOnlySpan<char> text, out int tokenId, out float score, out int matchLength);
    }

    private sealed class LegacyTrie : ITrieLike
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

    private sealed class FlatTrieAdapter(Trie trie) : ITrieLike
    {
        public void Add(ReadOnlySpan<char> key, int tokenId, float score) => trie.Add(key, tokenId, score);

        public bool TryMatchLongest(ReadOnlySpan<char> text, out int tokenId, out float score, out int matchLength) =>
            trie.TryMatchLongest(text, out tokenId, out score, out matchLength);
    }
}
