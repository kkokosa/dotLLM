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
/// <remarks>
/// Hot-path optimization — the vocabulary is partitioned at construction into per-first-char
/// buckets. On each mask build we iterate buckets, consulting the (readonly) tracker's schema
/// check for the bucket's first character before doing any struct clones. Typically only a
/// handful of first chars are allowed at any given schema state, so the overwhelming majority
/// of the ~128K-token vocab is eliminated without a single parser/tracker copy.
/// </remarks>
public sealed class JsonSchemaConstraint : IDecodingConstraint
{
    private JsonCharParser _parser;
    private SchemaTracker _tracker;
    private readonly ITokenizer _tokenizer;
    private readonly int _vocabSize;
    private readonly int _eosTokenId;
    private readonly CompiledSchema _schema;

    // Shared across clones — derived purely from tokenizer + schema, both immutable
    // for the lifetime of the constraint.
    private readonly LruCache<SchemaStateKey, TokenMask> _maskCache;
    private readonly FirstCharBuckets _firstCharBuckets;

    /// <summary>
    /// Creates a new JSON Schema constraint.
    /// </summary>
    /// <param name="tokenizer">Tokenizer for decoding token text.</param>
    /// <param name="schemaJson">JSON Schema as a string.</param>
    /// <param name="maxCacheEntries">Maximum mask cache entries before LRU eviction (default 4096).</param>
    public JsonSchemaConstraint(ITokenizer tokenizer, string schemaJson, int maxCacheEntries = 4096)
        : this(tokenizer, SchemaCompiler.Compile(schemaJson), maxCacheEntries)
    {
    }

    /// <summary>
    /// Creates a new JSON Schema constraint from a pre-compiled schema (for reuse across requests).
    /// </summary>
    /// <param name="tokenizer">Tokenizer for decoding token text.</param>
    /// <param name="schema">Pre-compiled schema.</param>
    /// <param name="maxCacheEntries">Maximum mask cache entries before LRU eviction (default 4096).</param>
    internal JsonSchemaConstraint(ITokenizer tokenizer, CompiledSchema schema, int maxCacheEntries = 4096)
    {
        _schema = schema;
        _parser = new JsonCharParser();
        _tracker = new SchemaTracker(schema);
        _tokenizer = tokenizer;
        _vocabSize = tokenizer.VocabSize;
        _eosTokenId = tokenizer.EosTokenId;
        _maskCache = new LruCache<SchemaStateKey, TokenMask>(maxCacheEntries);
        _firstCharBuckets = new FirstCharBuckets(tokenizer, _eosTokenId);
    }

    /// <summary>Copy constructor for <see cref="Clone"/>.</summary>
    private JsonSchemaConstraint(JsonSchemaConstraint source)
    {
        _parser = source._parser;           // struct copy
        _tracker = source._tracker;         // struct copy
        _tokenizer = source._tokenizer;
        _vocabSize = source._vocabSize;
        _eosTokenId = source._eosTokenId;
        _schema = source._schema;           // immutable, shared
        _maskCache = source._maskCache;     // shared — masks are immutable once built
        _firstCharBuckets = source._firstCharBuckets; // shared — depends only on tokenizer
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
        var mask = new TokenMask(_vocabSize);

        // EOS is allowed only when the schema is fully satisfied at this position.
        // The first-char bucketing path deliberately excludes EOS from every bucket.
        if (_parser.IsComplete && _tracker.IsComplete(in _parser))
            mask.Allow(_eosTokenId);

        // Walk first-char buckets: for each bucket, consult the tracker's readonly
        // IsCharAllowedBySchema before spending any clones. Typically ~10 of ~256
        // buckets survive at any given schema state, eliminating >95% of token clones.
        var buckets = _firstCharBuckets.Buckets;
        for (int b = 0; b < buckets.Length; b++)
        {
            var bucket = buckets[b];
            if (!_tracker.IsCharAllowedBySchema(bucket.Key, in _parser))
                continue;

            int[] tokenIds = bucket.Value;
            for (int i = 0; i < tokenIds.Length; i++)
            {
                int tokenId = tokenIds[i];
                if (IsTokenValidAfterBucketCheck(tokenId, bucket.Key))
                    mask.Allow(tokenId);
            }
        }

        _maskCache.Add(stateKey, mask);
        return mask;
    }

    /// <summary>
    /// Simulates consuming <paramref name="tokenId"/>'s text through a cloned parser + tracker,
    /// returning whether the token is fully accepted. The bucket iteration in
    /// <see cref="BuildAndCacheMask"/> has already verified that the schema allows
    /// <paramref name="bucketFirstChar"/>, so the first IsCharAllowedBySchema check is skipped.
    /// </summary>
    private bool IsTokenValidAfterBucketCheck(int tokenId, char bucketFirstChar)
    {
        string tokenText = _tokenizer.DecodeToken(tokenId);
        Debug.Assert(tokenText.Length > 0, "Bucketing excludes empty tokens");
        Debug.Assert(tokenText[0] == bucketFirstChar, "Bucket first char must match token first char");

        // Value copies — zero allocations (both are fully unmanaged structs via InlineArray)
        var parserClone = _parser;
        var trackerClone = _tracker;

        // First char: schema check already cleared by the bucket. Only need to advance.
        if (!parserClone.TryAdvance(bucketFirstChar))
            return false;
        trackerClone.OnCharAdvanced(bucketFirstChar, in parserClone);

        // Remaining chars: full schema + syntactic check.
        for (int i = 1; i < tokenText.Length; i++)
        {
            char c = tokenText[i];
            if (!trackerClone.IsCharAllowedBySchema(c, in parserClone))
                return false;
            if (!parserClone.TryAdvance(c))
                return false;
            trackerClone.OnCharAdvanced(c, in parserClone);
        }
        return true;
    }

    // ──────────────────── First-char bucketing ────────────────────

    /// <summary>
    /// Groups all non-EOS, non-empty vocabulary tokens by the first character of their
    /// decoded text. Built once per tokenizer+schema pair and shared across all clones.
    /// </summary>
    private sealed class FirstCharBuckets
    {
        public readonly KeyValuePair<char, int[]>[] Buckets;

        public FirstCharBuckets(ITokenizer tokenizer, int eosTokenId)
        {
            int vocabSize = tokenizer.VocabSize;
            var groups = new Dictionary<char, List<int>>();

            for (int tokenId = 0; tokenId < vocabSize; tokenId++)
            {
                if (tokenId == eosTokenId)
                    continue;

                string text = tokenizer.DecodeToken(tokenId);
                if (text.Length == 0)
                    continue;

                char first = text[0];
                if (!groups.TryGetValue(first, out var list))
                {
                    list = new List<int>();
                    groups[first] = list;
                }
                list.Add(tokenId);
            }

            Buckets = new KeyValuePair<char, int[]>[groups.Count];
            int i = 0;
            foreach (var (c, list) in groups)
                Buckets[i++] = new KeyValuePair<char, int[]>(c, list.ToArray());
        }
    }

    // ──────────────────── LRU cache ────────────────────

    /// <summary>
    /// Simple bounded LRU cache. Single-threaded by contract — callers must not mutate
    /// from multiple threads concurrently (matches the single-sequence decoding model).
    /// Evicts the least-recently-used entry when capacity is reached, avoiding the
    /// cache-cliff of a full flush on overflow.
    /// </summary>
    private sealed class LruCache<TKey, TValue> where TKey : notnull
    {
        private readonly int _capacity;
        private readonly Dictionary<TKey, LinkedListNode<Entry>> _map;
        private readonly LinkedList<Entry> _order;

        public LruCache(int capacity)
        {
            if (capacity <= 0)
                throw new ArgumentOutOfRangeException(nameof(capacity));
            _capacity = capacity;
            _map = new Dictionary<TKey, LinkedListNode<Entry>>(capacity);
            _order = new LinkedList<Entry>();
        }

        public int Count => _map.Count;

        public bool TryGetValue(TKey key, out TValue value)
        {
            if (_map.TryGetValue(key, out var node))
            {
                _order.Remove(node);
                _order.AddFirst(node);
                value = node.Value.Value;
                return true;
            }
            value = default!;
            return false;
        }

        public void Add(TKey key, TValue value)
        {
            if (_map.TryGetValue(key, out var existing))
            {
                _order.Remove(existing);
                existing.Value = new Entry(key, value);
                _order.AddFirst(existing);
                return;
            }

            if (_map.Count >= _capacity)
            {
                var lru = _order.Last!;
                _order.RemoveLast();
                _map.Remove(lru.Value.Key);
            }

            var node = new LinkedListNode<Entry>(new Entry(key, value));
            _order.AddFirst(node);
            _map[key] = node;
        }

        private readonly record struct Entry(TKey Key, TValue Value);
    }
}
