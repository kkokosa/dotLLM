using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.PromptCache;

/// <summary>
/// Regression tests for the prefix-cache reuse contract in <see cref="TextGenerator"/>:
/// only cache types <c>ResolveKvCache</c> knows how to truncate-and-reuse may be stored,
/// and the type set must stay in sync between the resolve and store paths (issue #121, items 6 and 7).
/// </summary>
public sealed class PrefixCacheResolveTests
{
    [Fact]
    public void SupportsPrefixReuse_AcceptsSimpleKvCache()
    {
        using var cache = new SimpleKvCache(numLayers: 1, numKvHeads: 1, headDim: 4, maxSeqLen: 16);
        Assert.True(TextGenerator.SupportsPrefixReuse(cache));
    }

    [Fact]
    public void SupportsPrefixReuse_RejectsCustomCacheType()
    {
        // Quantized / GPU / user-factory caches that don't implement SetCurrentLength can't be
        // truncated to a matched prefix length — storing them in the prefix cache would pin memory
        // forever since they'd never satisfy the type switch in ResolveKvCache.
        using var custom = new StubKvCache(maxLength: 32);
        Assert.False(TextGenerator.SupportsPrefixReuse(custom));
    }

    /// <summary>Minimal IKvCache implementation outside the SimpleKvCache/PagedKvCache reuse switch.</summary>
    private sealed class StubKvCache(int maxLength) : IKvCache
    {
        public int CurrentLength { get; private set; }
        public int MaxLength { get; } = maxLength;

        public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex) { }
        public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex) { }
        public ITensor GetKeys(int layerIndex) => throw new NotSupportedException();
        public ITensor GetValues(int layerIndex) => throw new NotSupportedException();
        public TensorRef GetKeysRef(int layerIndex) => default;
        public TensorRef GetValuesRef(int layerIndex) => default;
        public void Rollback(int length) => CurrentLength = length;
        public void Dispose() { }
    }
}
