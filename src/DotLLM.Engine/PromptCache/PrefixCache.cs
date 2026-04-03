using System.Diagnostics;
using DotLLM.Core.Attention;

namespace DotLLM.Engine.PromptCache;

/// <summary>
/// LRU cache of prefix-matched KV-cache sessions for multi-turn conversations.
/// Stores live KV-cache instances with their associated token sequences.
/// On each generation call, finds the longest matching prefix to skip redundant prefill.
/// </summary>
public sealed class PrefixCache : IDisposable
{
    private readonly int _maxEntries;
    private readonly List<PrefixCacheEntry> _entries;
    private bool _disposed;

    /// <summary>Number of cached entries.</summary>
    public int EntryCount => _entries.Count;

    /// <summary>
    /// Creates a new prefix cache with the given maximum number of entries.
    /// </summary>
    /// <param name="maxEntries">Maximum cached sessions. When full, LRU entry is evicted.</param>
    public PrefixCache(int maxEntries = 4)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxEntries, 1);
        _maxEntries = maxEntries;
        _entries = new List<PrefixCacheEntry>(maxEntries);
    }

    /// <summary>
    /// Finds the entry with the longest common prefix match against the given prompt tokens.
    /// Returns the matched entry and the number of matched tokens, or (null, 0) on miss.
    /// </summary>
    internal (PrefixCacheEntry? Entry, int MatchedTokens) FindMatch(ReadOnlySpan<int> promptTokenIds)
    {
        PrefixCacheEntry? bestEntry = null;
        int bestMatch = 0;

        for (int i = 0; i < _entries.Count; i++)
        {
            int match = _entries[i].GetPrefixMatchLength(promptTokenIds);
            if (match > bestMatch)
            {
                bestMatch = match;
                bestEntry = _entries[i];
            }
        }

        if (bestEntry != null)
            bestEntry.LastAccessTicks = Stopwatch.GetTimestamp();

        return (bestEntry, bestMatch);
    }

    /// <summary>
    /// Stores a KV-cache with its associated token sequence.
    /// If the same KV-cache instance already exists, updates the token sequence.
    /// Evicts the LRU entry if at capacity.
    /// </summary>
    internal void Store(int[] tokenSequence, IKvCache kvCache)
    {
        // Check if this KV-cache is already stored (reuse case)
        for (int i = 0; i < _entries.Count; i++)
        {
            if (ReferenceEquals(_entries[i].KvCache, kvCache))
            {
                // Update token sequence and access time for existing entry
                _entries[i] = new PrefixCacheEntry(tokenSequence, kvCache);
                return;
            }
        }

        // Evict LRU if at capacity
        if (_entries.Count >= _maxEntries)
        {
            int lruIdx = 0;
            long lruTicks = _entries[0].LastAccessTicks;
            for (int i = 1; i < _entries.Count; i++)
            {
                if (_entries[i].LastAccessTicks < lruTicks)
                {
                    lruTicks = _entries[i].LastAccessTicks;
                    lruIdx = i;
                }
            }
            _entries[lruIdx].Dispose();
            _entries.RemoveAt(lruIdx);
        }

        _entries.Add(new PrefixCacheEntry(tokenSequence, kvCache));
    }

    /// <summary>
    /// Removes all entries and disposes their KV-caches.
    /// </summary>
    public void Clear()
    {
        for (int i = 0; i < _entries.Count; i++)
            _entries[i].Dispose();
        _entries.Clear();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Clear();
    }
}
