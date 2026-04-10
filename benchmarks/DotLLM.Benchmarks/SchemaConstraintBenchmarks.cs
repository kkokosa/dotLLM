using BenchmarkDotNet.Attributes;
using DotLLM.Engine.Constraints;
using DotLLM.Tokenizers;

namespace DotLLM.Benchmarks;

/// <summary>
/// Benchmarks for <see cref="JsonSchemaConstraint"/> mask build cost. Uses a synthetic
/// tokenizer with a configurable vocabulary size so we can measure how mask construction
/// scales with vocab size (the dominant cost in real JSON-schema-constrained generation).
/// <para>
/// Each iteration creates a fresh constraint so <see cref="JsonSchemaConstraint.GetAllowedTokens"/>
/// is always a cache miss — otherwise we'd measure a trivial dictionary lookup.
/// </para>
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class SchemaConstraintBenchmarks
{
    private const string ObjectSchema = """
        {
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age":  { "type": "integer" },
                "tags": { "type": "array", "items": { "type": "string" } }
            },
            "required": ["name", "age"]
        }
        """;

    private ITokenizer _tokenizer = null!;

    /// <summary>
    /// Vocabulary size — the number of single-char synthetic tokens. The cost of
    /// <see cref="JsonSchemaConstraint.GetAllowedTokens"/> on a cache miss is dominated
    /// by the inner per-token simulation loop, so this parameter directly controls the
    /// measured work.
    /// </summary>
    [Params(4_000, 32_000)]
    public int VocabSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _tokenizer = new SyntheticTokenizer(VocabSize);
    }

    /// <summary>
    /// Creates a constraint and builds the initial mask in one measurement. This exercises
    /// both the one-shot FirstCharBuckets construction and the first mask build on an
    /// at-start schema state.
    /// </summary>
    [Benchmark]
    public int ConstraintCreateAndFirstMask()
    {
        var constraint = new JsonSchemaConstraint(_tokenizer, ObjectSchema);
        var mask = constraint.GetAllowedTokens();
        // Touch the mask so the JIT can't elide it.
        return mask.IsAllowed(0) ? 1 : 0;
    }

    /// <summary>
    /// Measures a single cache-missing mask build on an already-constructed constraint.
    /// Isolates the BuildAndCacheMask cost from the one-shot bucket construction.
    /// </summary>
    [Benchmark]
    public int MaskBuild_CacheMiss()
    {
        // Fresh constraint every invocation: the first GetAllowedTokens call is always a miss.
        // The bucket table is rebuilt too, so this is an upper bound on per-request cost.
        var constraint = new JsonSchemaConstraint(_tokenizer, ObjectSchema);
        var mask = constraint.GetAllowedTokens();
        return mask.IsAllowed(0) ? 1 : 0;
    }

    /// <summary>
    /// Synthetic tokenizer with <paramref name="vocabSize"/> single-character tokens covering
    /// the printable ASCII range plus a few JSON-structural chars. Used only for benchmarks.
    /// </summary>
    private sealed class SyntheticTokenizer : ITokenizer
    {
        private readonly string[] _tokens;

        public SyntheticTokenizer(int vocabSize)
        {
            _tokens = new string[vocabSize];
            // Seed the first 128 slots with printable ASCII and JSON structural chars,
            // then pad the rest with repeating chars so every bucket has realistic fanout.
            const string jsonStructure = "{}[]\":,0123456789tfnrulsea \n";
            for (int i = 0; i < _tokens.Length; i++)
            {
                int idx = i % (jsonStructure.Length + 95);
                if (idx < jsonStructure.Length)
                    _tokens[i] = jsonStructure[idx].ToString();
                else
                    _tokens[i] = ((char)(33 + (idx - jsonStructure.Length))).ToString();
            }
        }

        public int VocabSize => _tokens.Length;
        public int BosTokenId => 0;
        public int EosTokenId => _tokens.Length - 1;
        public string DecodeToken(int tokenId) =>
            tokenId >= 0 && tokenId < _tokens.Length ? _tokens[tokenId] : "";
        public int[] Encode(string text) => throw new NotSupportedException();
        public string Decode(ReadOnlySpan<int> tokenIds) => throw new NotSupportedException();
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => throw new NotSupportedException();
        public int CountTokens(string text) => throw new NotSupportedException();
    }
}
