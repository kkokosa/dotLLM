using System.Diagnostics;
using DotLLM.Core.Attention;

namespace DotLLM.Engine.PromptCache;

/// <summary>
/// A single cached session: the token prefix and the live KV-cache populated by it.
/// The KV-cache is owned by this entry and disposed on eviction.
/// </summary>
internal sealed class PrefixCacheEntry : IDisposable
{
    /// <summary>Full token sequence (prompt + generated) stored in the KV-cache.</summary>
    public int[] TokenSequence { get; }

    /// <summary>Live KV-cache with state for <see cref="TokenSequence"/>.</summary>
    public IKvCache KvCache { get; }

    /// <summary>Timestamp of last access, for LRU eviction.</summary>
    public long LastAccessTicks { get; set; }

    public PrefixCacheEntry(int[] tokenSequence, IKvCache kvCache)
    {
        TokenSequence = tokenSequence;
        KvCache = kvCache;
        LastAccessTicks = Stopwatch.GetTimestamp();
    }

    /// <summary>
    /// Returns the number of leading tokens shared between this entry and the given prompt.
    /// </summary>
    public int GetPrefixMatchLength(ReadOnlySpan<int> promptTokenIds)
    {
        int maxLen = Math.Min(TokenSequence.Length, promptTokenIds.Length);
        for (int i = 0; i < maxLen; i++)
        {
            if (TokenSequence[i] != promptTokenIds[i])
                return i;
        }
        return maxLen;
    }

    public void Dispose() => KvCache.Dispose();
}
