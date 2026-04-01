using System.Diagnostics;
using DotLLM.Core.Constraints;
using DotLLM.Engine.Constraints.Grammar;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Constraints;

/// <summary>
/// Constrained decoding constraint that guarantees output conforms to a GBNF grammar.
/// Compiles the grammar to a PDA at construction time.
/// Token masks are lazily computed and cached by PDA state key.
/// </summary>
public sealed class GrammarConstraint : IDecodingConstraint
{
    private PdaSimulator _simulator;
    private readonly CompiledGrammar _grammar;
    private readonly ITokenizer _tokenizer;
    private readonly int _vocabSize;
    private readonly int _eosTokenId;

    // Shared across clones. Not thread-safe — single-sequence use only.
    private readonly Dictionary<GrammarStateKey, TokenMask> _maskCache;
    private readonly int _maxCacheEntries;

    /// <summary>
    /// Creates a grammar constraint from GBNF text.
    /// Compiles: GBNF → AST → compiled grammar → PDA.
    /// </summary>
    /// <param name="tokenizer">Tokenizer for decoding token IDs to text.</param>
    /// <param name="grammar">GBNF grammar definition.</param>
    /// <param name="maxCacheEntries">Maximum cache entries before full eviction (default 4096).</param>
    public GrammarConstraint(ITokenizer tokenizer, string grammar, int maxCacheEntries = 4096)
        : this(tokenizer, CompileGrammar(grammar), maxCacheEntries)
    {
    }

    /// <summary>
    /// Creates a grammar constraint from a pre-compiled grammar (for reuse across requests).
    /// </summary>
    internal GrammarConstraint(ITokenizer tokenizer, CompiledGrammar grammar, int maxCacheEntries = 4096)
    {
        _grammar = grammar;
        _simulator = new PdaSimulator(grammar);
        _tokenizer = tokenizer;
        _vocabSize = tokenizer.VocabSize;
        _eosTokenId = tokenizer.EosTokenId;
        _maskCache = new Dictionary<GrammarStateKey, TokenMask>();
        _maxCacheEntries = maxCacheEntries;
    }

    /// <summary>Copy constructor for <see cref="Clone"/>.</summary>
    private GrammarConstraint(GrammarConstraint source)
    {
        _simulator = source._simulator; // struct — copies by value (~268 bytes)
        _grammar = source._grammar;
        _tokenizer = source._tokenizer;
        _vocabSize = source._vocabSize;
        _eosTokenId = source._eosTokenId;
        _maskCache = source._maskCache; // shared cache
        _maxCacheEntries = source._maxCacheEntries;
    }

    /// <inheritdoc/>
    public void Advance(int tokenId)
    {
        if (tokenId == _eosTokenId)
            return;

        string text = _tokenizer.DecodeToken(tokenId);
        foreach (char c in text)
        {
            bool ok = _simulator.TryAdvance(c);
            Debug.Assert(ok, $"Constraint allowed token that advances to invalid state at char '{c}'");
        }
    }

    /// <inheritdoc/>
    public TokenMask GetAllowedTokens()
    {
        var stateKey = _simulator.GetEffectiveStateKey();
        if (_maskCache.TryGetValue(stateKey, out var cached))
            return cached;
        return BuildAndCacheMask(stateKey);
    }

    /// <inheritdoc/>
    public bool IsComplete() => _simulator.IsAccepting;

    /// <inheritdoc/>
    public IDecodingConstraint Clone() => new GrammarConstraint(this);

    /// <inheritdoc/>
    public void Reset()
    {
        _simulator.Reset();
        // Don't clear cache — it's still valid for the same grammar
    }

    private TokenMask BuildAndCacheMask(GrammarStateKey stateKey)
    {
        // Evict cache if full
        if (_maskCache.Count >= _maxCacheEntries)
            _maskCache.Clear();

        var mask = new TokenMask(_vocabSize);

        for (int tokenId = 0; tokenId < _vocabSize; tokenId++)
        {
            if (tokenId == _eosTokenId)
            {
                // EOS allowed only when grammar can accept
                if (_simulator.IsAccepting || _simulator.CanAccept())
                    mask.Allow(tokenId);
                continue;
            }

            if (IsTokenValid(tokenId))
                mask.Allow(tokenId);
        }

        _maskCache[stateKey] = mask;
        return mask;
    }

    private bool IsTokenValid(int tokenId)
    {
        string tokenText = _tokenizer.DecodeToken(tokenId);
        if (tokenText.Length == 0)
            return false;

        // Struct copy — zero heap allocations (PdaSimulator uses InlineArray stacks)
        var clone = _simulator;
        foreach (char c in tokenText)
        {
            if (!clone.TryAdvance(c))
                return false;
        }
        return true;
    }

    private static CompiledGrammar CompileGrammar(string grammar)
    {
        var ast = GbnfParser.Parse(grammar);
        return CompiledGrammar.Compile(ast);
    }
}
