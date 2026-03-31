using System.Diagnostics;
using DotLLM.Core.Constraints;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Constraints;

/// <summary>
/// Constrained decoding constraint that guarantees syntactically valid JSON output (RFC 8259).
/// Uses an FSM-based approach: at each decode step, iterates over the vocabulary to build
/// a <see cref="TokenMask"/> of tokens whose text is valid at the current parser state.
/// Masks are cached by effective state key for reuse.
/// </summary>
public sealed class JsonConstraint : IDecodingConstraint
{
    private JsonCharParser _parser;
    private readonly ITokenizer _tokenizer;
    private readonly int _vocabSize;
    private readonly int _eosTokenId;

    // Shared across clones. Not thread-safe — single-sequence use only.
    // See Step 35 (continuous batching) for concurrent access requirements.
    private readonly Dictionary<int, TokenMask> _maskCache;

    /// <summary>
    /// Creates a new JSON constraint for the given tokenizer's vocabulary.
    /// </summary>
    /// <param name="tokenizer">Tokenizer used to decode token IDs to text for FSM simulation.</param>
    public JsonConstraint(ITokenizer tokenizer)
    {
        _parser = new JsonCharParser();
        _tokenizer = tokenizer;
        _vocabSize = tokenizer.VocabSize;
        _eosTokenId = tokenizer.EosTokenId;
        _maskCache = new Dictionary<int, TokenMask>();
    }

    /// <summary>Copy constructor for <see cref="Clone"/>.</summary>
    private JsonConstraint(JsonConstraint source)
    {
        _parser = source._parser; // struct — copies by value, zero allocations
        _tokenizer = source._tokenizer;
        _vocabSize = source._vocabSize;
        _eosTokenId = source._eosTokenId;
        _maskCache = source._maskCache; // shared cache — masks are immutable once built
    }

    /// <inheritdoc/>
    public void Advance(int tokenId)
    {
        if (tokenId == _eosTokenId)
            return;

        string text = _tokenizer.DecodeToken(tokenId);
        foreach (char c in text)
        {
            bool ok = _parser.TryAdvance(c);
            Debug.Assert(ok, $"Constraint allowed token that advances to invalid state at char '{c}'");
        }
    }

    /// <inheritdoc/>
    public TokenMask GetAllowedTokens()
    {
        int stateKey = _parser.GetEffectiveStateKey();
        if (_maskCache.TryGetValue(stateKey, out var cached))
            return cached;
        return BuildAndCacheMask(stateKey);
    }

    /// <inheritdoc/>
    public bool IsComplete() => _parser.IsComplete;

    /// <inheritdoc/>
    public IDecodingConstraint Clone() => new JsonConstraint(this);

    /// <inheritdoc/>
    public void Reset()
    {
        _parser.Reset();
        // Don't clear cache — it's still valid for the same vocabulary
    }

    private TokenMask BuildAndCacheMask(int stateKey)
    {
        var mask = new TokenMask(_vocabSize);

        for (int tokenId = 0; tokenId < _vocabSize; tokenId++)
        {
            if (tokenId == _eosTokenId)
            {
                // EOS allowed only when JSON is complete
                if (_parser.IsComplete)
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

        // Struct copy — zero allocations (JsonCharParser is fully unmanaged via InlineArray)
        var clone = _parser;
        foreach (char c in tokenText)
        {
            if (!clone.TryAdvance(c))
                return false;
        }
        return true;
    }
}
