using BenchmarkDotNet.Attributes;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.HuggingFace;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;

namespace DotLLM.Benchmarks;

/// <summary>
/// Measures the per-decode-step saving from direct-to-cache K/V projection (#25 item 4):
/// the K and V GEMMs write straight into the cache slot, skipping the scratch + Update
/// memcpy. <see cref="LegacyUpdateOnlyCache"/> forces the pre-#278 baseline by returning
/// <c>false</c> from <see cref="IKvCache.TryReserveSlot"/>.
///
/// Per-decode-step saving on a 7B GQA-2 (kvStride = 1024 floats / 4 KiB) over 32 layers:
/// 2 × 4 KiB × 32 = 256 KiB of copy per token eliminated. The exact wall-time impact is
/// model and cache-size dependent — this benchmark prints the delta on the bundled
/// SmolLM-135M Q8_0 model (small but real).
/// </summary>
[SimpleJob(warmupCount: 2, iterationCount: 5)]
public class DirectKvWriteBenchmarks
{
    private GgufFile _gguf = null!;
    private TransformerModel _model = null!;
    private BpeTokenizer _tokenizer = null!;
    private int[] _promptIds = null!;
    private int[] _positions = null!;

    // Two pre-prefilled caches, reset to the same prefill state between iterations.
    private SimpleKvCache _baseCache = null!;

    [GlobalSetup]
    public void Setup()
    {
        const string Repo = "QuantFactory/SmolLM-135M-GGUF";
        const string FileName = "SmolLM-135M.Q8_0.gguf";
        string cacheDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "test-cache");
        string cachedPath = Path.Combine(cacheDir, Repo.Replace('/', Path.DirectorySeparatorChar), FileName);
        string path;
        if (File.Exists(cachedPath))
        {
            path = cachedPath;
        }
        else
        {
            using var downloader = new HuggingFaceDownloader();
            path = downloader.DownloadFileAsync(Repo, FileName, cacheDir).GetAwaiter().GetResult();
        }

        _gguf = GgufFile.Open(path);
        var cfg = GgufModelConfigExtractor.Extract(_gguf.Metadata);
        _model = TransformerModel.LoadFromGguf(_gguf, cfg);
        _tokenizer = GgufBpeTokenizerFactory.Load(_gguf.Metadata);

        _promptIds = _tokenizer.Encode("The capital of France is");
        _positions = new int[_promptIds.Length + 64];
        for (int i = 0; i < _positions.Length; i++) _positions[i] = i;

        _baseCache = NewPrefilledCache();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _baseCache.Dispose();
        _model.Dispose();
        _gguf.Dispose();
    }

    private SimpleKvCache NewPrefilledCache()
    {
        var cache = new SimpleKvCache(
            _model.Config.NumLayers, _model.Config.NumKvHeads, _model.Config.HeadDim,
            _positions.Length);
        using var _ = _model.Forward(_promptIds, _positions.AsSpan(0, _promptIds.Length), -1, cache);
        return cache;
    }

    /// <summary>One decode step with the direct-to-cache path enabled (the new default).</summary>
    [Benchmark(Baseline = false)]
    public int Decode_DirectToCache()
    {
        using var cache = NewPrefilledCache();
        int pos = _promptIds.Length;
        using var logits = _model.Forward([_promptIds[^1]], _positions.AsSpan(pos, 1), -1, cache);
        return cache.CurrentLength;
    }

    /// <summary>
    /// One decode step forced onto the legacy <see cref="IKvCache.Update"/> path
    /// via <see cref="LegacyUpdateOnlyCache"/>. The delta to
    /// <see cref="Decode_DirectToCache"/> is the K/V scratch→cache memcpy saved.
    /// </summary>
    [Benchmark(Baseline = true)]
    public int Decode_LegacyUpdate()
    {
        using var inner = NewPrefilledCache();
        using var legacy = new LegacyUpdateOnlyCache(inner);
        int pos = _promptIds.Length;
        using var logits = _model.Forward([_promptIds[^1]], _positions.AsSpan(pos, 1), -1, legacy);
        return legacy.CurrentLength;
    }

    /// <summary>
    /// IKvCache decorator that forces the legacy <see cref="IKvCache.Update"/> path by
    /// returning <c>false</c> from <see cref="IKvCache.TryReserveSlot"/>. Used to A/B
    /// the direct-to-cache optimisation against the pre-#278 baseline behaviour.
    /// </summary>
    private sealed class LegacyUpdateOnlyCache : IKvCache
    {
        private readonly IKvCache _inner;
        public LegacyUpdateOnlyCache(IKvCache inner) => _inner = inner;
        public int CurrentLength => _inner.CurrentLength;
        public int MaxLength => _inner.MaxLength;
        public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex) =>
            _inner.Update(keys, values, positions, layerIndex);
        public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex) =>
            _inner.Update(keys, values, positions, layerIndex);
        public ITensor GetKeys(int layerIndex) => _inner.GetKeys(layerIndex);
        public ITensor GetValues(int layerIndex) => _inner.GetValues(layerIndex);
        public TensorRef GetKeysRef(int layerIndex) => _inner.GetKeysRef(layerIndex);
        public TensorRef GetValuesRef(int layerIndex) => _inner.GetValuesRef(layerIndex);
        public void Rollback(int length) => _inner.Rollback(length);
        public bool TryReserveSlot(int layerIndex, ReadOnlySpan<int> positions, out Span<float> kDst, out Span<float> vDst)
        {
            kDst = default;
            vDst = default;
            return false;
        }
        public void CommitSlot(int layerIndex, ReadOnlySpan<int> positions) { }
        public void Dispose() { /* outer benchmark owns inner */ }
    }
}
