using BenchmarkDotNet.Attributes;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Bpe;

namespace DotLLM.Benchmarks.Tokenizers;

/// <summary>
/// Allocation-focused benchmark comparing the existing allocating
/// <see cref="ITokenizer.Decode(System.ReadOnlySpan{int}, bool)"/> path against the
/// zero-allocation <see cref="ITokenizer.TryDecode"/> overload introduced for the
/// <c>IncrementalDetokenizer</c> hot path. The interesting column in the BenchmarkDotNet
/// report is <c>Allocated</c> — the bytes-per-op delta is what motivates the new surface.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class DetokenizerAllocBenchmark
{
    private BpeTokenizer _spm = null!;
    private BpeTokenizer _tiktoken = null!;
    private int[] _spmIds = null!;
    private int[] _tiktokenIds = null!;
    private char[] _scratch = null!;

    /// <summary>Number of decode calls in the benchmark loop.</summary>
    [Params(1000)]
    public int CallsPerInvoke { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _spm = BuildSpmVocab();
        _tiktoken = BuildTiktokenVocab();
        _spmIds = [10, 1, 11]; // "▁hello ▁world"-ish
        _tiktokenIds = [11, 5, 15]; // "hello world"
        _scratch = new char[256];

        // Fail fast if _scratch can no longer hold either decode. The measured loops
        // deliberately ignore the TryDecode result to keep the benchmark body minimal;
        // without this guard a vocab change would silently make them time the
        // buffer-too-small failure path (written == 0) and report meaningless numbers.
        EnsureFits(_spm, _spmIds, nameof(_spmIds));
        EnsureFits(_tiktoken, _tiktokenIds, nameof(_tiktokenIds));

        void EnsureFits(ITokenizer tokenizer, int[] ids, string name)
        {
            if (!tokenizer.TryDecode(ids, stripBosSpace: false, _scratch, out int written) || written == 0)
                throw new InvalidOperationException(
                    $"Scratch buffer ({_scratch.Length} chars) is too small to decode {name}; " +
                    "the benchmark would measure the failure path.");
        }
    }

    [Benchmark(Baseline = true, Description = "SPM.Decode (allocating)")]
    public int SpmDecode()
    {
        int total = 0;
        for (int i = 0; i < CallsPerInvoke; i++)
            total += _spm.Decode(_spmIds, stripBosSpace: false).Length;
        return total;
    }

    [Benchmark(Description = "SPM.TryDecode (zero-alloc)")]
    public int SpmTryDecode()
    {
        int total = 0;
        for (int i = 0; i < CallsPerInvoke; i++)
        {
            _spm.TryDecode(_spmIds, stripBosSpace: false, _scratch, out int written);
            total += written;
        }
        return total;
    }

    [Benchmark(Description = "Tiktoken.Decode (allocating)")]
    public int TiktokenDecode()
    {
        int total = 0;
        for (int i = 0; i < CallsPerInvoke; i++)
            total += _tiktoken.Decode(_tiktokenIds, stripBosSpace: false).Length;
        return total;
    }

    [Benchmark(Description = "Tiktoken.TryDecode (zero-alloc)")]
    public int TiktokenTryDecode()
    {
        int total = 0;
        for (int i = 0; i < CallsPerInvoke; i++)
        {
            _tiktoken.TryDecode(_tiktokenIds, stripBosSpace: false, _scratch, out int written);
            total += written;
        }
        return total;
    }

    private static BpeTokenizer BuildSpmVocab()
    {
        string[] tokens =
        [
            "<unk>", "▁", "h", "e", "l", "o",
            "▁h", "▁he", "▁hel", "▁hell", "▁hello",
            "▁world", "▁w", "w", "r", "d",
        ];
        float[] scores = new float[tokens.Length];
        return BpeTokenizer.CreateSentencePiece(tokens, scores, tokenTypes: null,
            bosId: 0, eosId: 0, addBosSpace: false);
    }

    private static BpeTokenizer BuildTiktokenVocab()
    {
        string[] tokens =
        [
            "<unk>", "h", "e", "l", "o", " ", "w", "r", "d",
            "he", "lo", "hello", " w", "wo", "rl", "world",
        ];
        string[] merges =
        [
            "h e", "l o", "he llo", "w o", "r l", "wo rld",
        ];
        return BpeTokenizer.CreateTiktoken(tokens, merges, tokenTypes: null,
            bosId: 0, eosId: 0, preTokenizerType: null);
    }
}
