namespace DotLLM.Engine.PromptCache;

/// <summary>
/// Configuration for the cross-request prefix trie / advanced prompt cache
/// (Step 37). The trie sits in front of <see cref="DotLLM.Engine.KvCache.PagedKvCache"/>
/// and shares already-computed KV blocks across concurrent sequences that
/// have a common prompt prefix (e.g. a shared system prompt).
/// </summary>
public sealed record PrefixCacheConfig
{
    /// <summary>Whether the prefix trie is active. When false the engine
    /// behaves exactly as it did before Step 37.</summary>
    public bool Enabled { get; init; } = true;

    /// <summary>
    /// Maximum prefix depth in tokens that may be matched. Walking the trie
    /// stops once the matched length reaches this cap. Set to <c>0</c> to
    /// disable the cap (match arbitrarily long prefixes).
    /// </summary>
    /// <remarks>
    /// The cap protects against pathologically long prompts dominating the
    /// trie. A typical system prompt is &lt; 2048 tokens; values around
    /// 8192 are a safe upper bound for most chat workloads.
    /// </remarks>
    public int MaxPrefixDepth { get; init; }

    /// <summary>
    /// When the block pool is exhausted, may the prefix manager evict
    /// zero-refcount trie nodes (LRU first) to recover blocks?
    /// </summary>
    /// <remarks>
    /// The MVP (Step 37) ships with eviction enabled but limited to
    /// zero-refcount blocks. Active sequences are never preempted —
    /// that lives in Step 59 (advanced scheduling).
    /// </remarks>
    public bool EvictionEnabled { get; init; } = true;
}
