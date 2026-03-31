using System.Diagnostics;
using DotLLM.Core.Constraints;
using DotLLM.Engine.Constraints.Schema;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Constraints;

/// <summary>
/// Constrained decoding constraint that guarantees output conforms to a JSON Schema.
/// Layers schema tracking (<see cref="SchemaTracker"/>) on top of <see cref="JsonCharParser"/>
/// for syntactic validation. Token masks are lazily computed and cached by composite state key.
/// </summary>
public sealed class JsonSchemaConstraint : IDecodingConstraint
{
    private JsonCharParser _parser;
    private SchemaTracker _tracker;
    private readonly ITokenizer _tokenizer;
    private readonly int _vocabSize;
    private readonly int _eosTokenId;
    private readonly CompiledSchema _schema;

    // Shared across clones. Not thread-safe — single-sequence use only.
    private readonly Dictionary<SchemaStateKey, TokenMask> _maskCache;
    private readonly int _maxCacheEntries;

    /// <summary>
    /// Creates a new JSON Schema constraint.
    /// </summary>
    /// <param name="tokenizer">Tokenizer for decoding token text.</param>
    /// <param name="schemaJson">JSON Schema as a string.</param>
    /// <param name="maxCacheEntries">Maximum mask cache entries before eviction (default 4096).</param>
    public JsonSchemaConstraint(ITokenizer tokenizer, string schemaJson, int maxCacheEntries = 4096)
        : this(tokenizer, SchemaCompiler.Compile(schemaJson), maxCacheEntries)
    {
    }

    /// <summary>
    /// Creates a new JSON Schema constraint from a pre-compiled schema (for reuse across requests).
    /// </summary>
    /// <param name="tokenizer">Tokenizer for decoding token text.</param>
    /// <param name="schema">Pre-compiled schema.</param>
    /// <param name="maxCacheEntries">Maximum mask cache entries before eviction (default 4096).</param>
    internal JsonSchemaConstraint(ITokenizer tokenizer, CompiledSchema schema, int maxCacheEntries = 4096)
    {
        _schema = schema;
        _parser = new JsonCharParser();
        _tracker = new SchemaTracker(schema);
        _tokenizer = tokenizer;
        _vocabSize = tokenizer.VocabSize;
        _eosTokenId = tokenizer.EosTokenId;
        _maskCache = new Dictionary<SchemaStateKey, TokenMask>();
        _maxCacheEntries = maxCacheEntries;
    }

    /// <summary>Copy constructor for <see cref="Clone"/>.</summary>
    private JsonSchemaConstraint(JsonSchemaConstraint source)
    {
        _parser = source._parser;       // struct copy
        _tracker = source._tracker;     // struct copy
        _tokenizer = source._tokenizer;
        _vocabSize = source._vocabSize;
        _eosTokenId = source._eosTokenId;
        _schema = source._schema;       // immutable, shared
        _maskCache = source._maskCache; // shared — masks are immutable once built
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
            bool ok = _parser.TryAdvance(c);
            Debug.Assert(ok, $"Schema constraint allowed token that advances to invalid JSON state at char '{c}'");
            _tracker.OnCharAdvanced(c, in _parser);
        }
    }

    /// <inheritdoc/>
    public TokenMask GetAllowedTokens()
    {
        var stateKey = _tracker.GetSchemaStateKey(in _parser);
        if (_maskCache.TryGetValue(stateKey, out var cached))
            return cached;
        return BuildAndCacheMask(stateKey);
    }

    /// <inheritdoc/>
    public bool IsComplete() => _parser.IsComplete && _tracker.IsComplete(in _parser);

    /// <inheritdoc/>
    public IDecodingConstraint Clone() => new JsonSchemaConstraint(this);

    /// <inheritdoc/>
    public void Reset()
    {
        _parser.Reset();
        _tracker.Reset();
        // Don't clear cache — still valid for same schema + vocabulary
    }

    private TokenMask BuildAndCacheMask(SchemaStateKey stateKey)
    {
        // Evict if cache is full (simple clear strategy)
        if (_maskCache.Count >= _maxCacheEntries)
            _maskCache.Clear();

        var mask = new TokenMask(_vocabSize);

        for (int tokenId = 0; tokenId < _vocabSize; tokenId++)
        {
            if (tokenId == _eosTokenId)
            {
                if (_parser.IsComplete && _tracker.IsComplete(in _parser))
                    mask.Allow(tokenId);
                continue;
            }

            if (IsTokenValid(tokenId))
                mask.Allow(tokenId);
        }

        _maskCache[stateKey] = mask;
        return mask;
    }

    // TODO: Perf — SchemaTracker is ~1.3KB due to InlineArray stacks. Copying it for each of
    // the 128K vocab tokens (~160MB per cache miss) is measurable. Optimize by grouping tokens
    // by first character or walking a tokenizer vocab trie so clones only happen at branch points.
    private bool IsTokenValid(int tokenId)
    {
        string tokenText = _tokenizer.DecodeToken(tokenId);
        if (tokenText.Length == 0)
            return false;

        // Value copies — zero allocations (both are fully unmanaged structs via InlineArray)
        var parserClone = _parser;
        var trackerClone = _tracker;

        foreach (char c in tokenText)
        {
            // Schema restriction check (uses parser state for context)
            if (!trackerClone.IsCharAllowedBySchema(c, in parserClone))
                return false;

            // Syntactic check
            if (!parserClone.TryAdvance(c))
                return false;

            // Update schema tracker with the accepted char and new parser state
            trackerClone.OnCharAdvanced(c, in parserClone);
        }
        return true;
    }
}
