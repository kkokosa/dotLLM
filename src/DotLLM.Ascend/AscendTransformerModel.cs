using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;

namespace DotLLM.Ascend;

/// <summary>
/// Single-card dense transformer with FP32 weights/activations and FP16 KV using ACLNN kernels. Weights, activations
/// and KV stay on the NPU; the final token's logits return to host FP32 for sampling.
/// Forward results are non-owning views valid until the next forward or disposal.
/// </summary>
public sealed class AscendTransformerModel : IModel
{
    private readonly object _gate = new();
    private readonly AscendSession _session;
    private readonly AscendWeights _weights;
    private readonly Layer[] _layers;
    private readonly AscendTensor _embedding, _outputNorm, _output;
    private AscendForwardState? _state;
    private bool _disposed;
    private readonly record struct Projection(AscendTensor Weight, AscendTensor? Bias);
    private readonly record struct Layer(AscendTensor AttnNorm, AscendTensor FfnNorm,
        Projection Q, Projection K, Projection V, Projection O, Projection Gate, Projection Up, Projection Down);

    /// <inheritdoc/>
    public ModelConfig Config { get; }
    /// <summary>Logical NPU ordinal for this model.</summary>
    public int DeviceId => _session.DeviceId;
    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state?.Bytes ?? 0;
    /// <summary>Expanded FP32 device weight storage, including norms and biases.</summary>
    public long DeviceWeightBytes => _weights.Bytes;

    private AscendTransformerModel(ModelConfig config, AscendSession session, AscendWeights weights)
    {
        Config = config; _session = session; _weights = weights;
        _embedding = weights["token_embd.weight"];
        _outputNorm = weights["output_norm.weight"]; _output = weights["output.weight"];
        _layers = Enumerable.Range(0, config.NumLayers).Select(i =>
        {
            string p = $"blk.{i}.";
            Projection Get(string name) => new(weights[p + name + ".weight"], weights.Optional(p + name + ".bias"));
            return new Layer(weights[p + "attn_norm.weight"], weights[p + "ffn_norm.weight"],
                Get("attn_q"), Get("attn_k"), Get("attn_v"), Get("attn_output"),
                Get("ffn_gate"), Get("ffn_up"), Get("ffn_down"));
        }).ToArray();
    }

    /// <summary>
    /// Validates and expands F16/F32/Q8_0 GGUF weights to FP32 on one NPU.
    /// The caller owns the file and may dispose it after this method returns.
    /// </summary>
    public static AscendTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, int deviceId = 0)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        AscendWeights.Validate(gguf, config);
        var session = AscendSession.Create(deviceId);
        try
        {
            long required = checked(gguf.Tensors.Sum(t => checked(t.Shape.ElementCount * 4)));
            var (free, _) = session.GetMemoryInfo();
            // Reserve room for scratch, KV and transient ACLNN workspace.
            if (required > free - Math.Min(free / 4, 1L << 30))
                throw new OutOfMemoryException($"Ascend FP32 weights need {required} bytes; free device memory is {free}.");
            var weights = AscendWeights.Load(gguf, session);
            try { return new(config, session, weights); }
            catch { weights.Dispose(); throw; }
        }
        catch { session.Dispose(); throw; }
    }

    /// <summary>Allocates an independent contiguous FP16 cache owned by the caller.</summary>
    public AscendKvCache CreateKvCache(int maxLength)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return new(this, _session, Config, maxLength);
        }
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, null);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (deviceId != -1 && deviceId != DeviceId)
                throw new ArgumentException("Forward device differs from the model's NPU ordinal.", nameof(deviceId));
            if (tokenIds.IsEmpty || tokenIds.Length != positions.Length)
                throw new ArgumentException("Forward requires equally sized, nonempty token and position spans.");
            AscendKvCache? cache = null;
            if (kvCache is not null)
            {
                cache = kvCache as AscendKvCache;
                if (cache is null || !ReferenceEquals(cache.Owner, this))
                    throw new ArgumentException("KV cache must belong to this Ascend model.", nameof(kvCache));
            }
            lock (cache?.SyncRoot ?? _gate)
            {
                cache?.Check();
                int offset = cache?.CurrentLength ?? 0;
                int limit = cache?.MaxLength ?? Config.MaxSequenceLength;
                if (tokenIds.Length > limit - offset)
                    throw new ArgumentOutOfRangeException(nameof(tokenIds), "Forward exceeds KV/context capacity.");
                for (int i = 0; i < tokenIds.Length; i++)
                    if ((uint)tokenIds[i] >= (uint)Config.VocabSize || positions[i] != offset + i)
                        throw new ArgumentException("Token IDs must be in vocabulary; positions must append contiguously.");
                if (_state?.Rows != tokenIds.Length)
                {
                    _state?.Dispose(); _state = null;
                    _state = new(_session, Config, tokenIds.Length);
                }
                try
                {
                    var result = Execute(tokenIds, positions, cache, offset);
                    cache?.Commit(offset + tokenIds.Length);
                    return result;
                }
                catch { cache?.Abort(); throw; }
            }
        }
    }

    private unsafe ITensor Execute(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions, AscendKvCache? cache, int offset)
    {
        var s = _state!;
        int h = Config.HiddenSize;
        int rows = tokens.Length;
        for (int i = 0; i < rows; i++)
            _session.CopyFloat32(_embedding, (long)tokens[i] * h, s.Hidden, (long)i * h, h);
        var rope = Config.RoPEConfig!.Value;
        bool neox = rope.Type == RoPEType.NeoX;
        var cos = new Span<Half>((void*)s.HostCos.DataPointer, checked(rows * Config.HeadDim));
        var sin = new Span<Half>((void*)s.HostSin.DataPointer, cos.Length);
        for (int row = 0; row < rows; row++)
            for (int d = 0; d < Config.HeadDim; d++)
            {
                int pair = neox ? d % (Config.HeadDim / 2) : d / 2;
                double angle = positions[row] * Math.Pow(rope.Theta, -2.0 * pair / Config.HeadDim);
                cos[row * Config.HeadDim + d] = (Half)Math.Cos(angle);
                sin[row * Config.HeadDim + d] = (Half)Math.Sin(angle);
            }
        _session.Upload(s.Cos, cos); _session.Upload(s.Sin, sin);
        for (int layer = 0; layer < _layers.Length; layer++)
        {
            ref readonly var l = ref _layers[layer];
            _session.RmsNormFloat32(s.Hidden, l.AttnNorm, s.Norm, Config.NormEpsilon);
            Project(s.Norm, l.Q, s.Q, s.BiasH);
            Project(s.Norm, l.K, s.K, s.BiasKv);
            Project(s.Norm, l.V, s.V, s.BiasKv);
            _session.RopeFloat32(s.Q, s.Cos, s.Sin, s.QRot, Config.NumAttentionHeads, Config.HeadDim, Config.HeadDim, neox);
            _session.RopeFloat32(s.K, s.Cos, s.Sin, s.KRot, Config.NumKvHeads, Config.HeadDim, Config.HeadDim, neox);
            if (cache is not null)
            {
                _session.Cast(s.KRot, s.CacheK); _session.Cast(s.V, s.CacheV);
                cache.Write(s.CacheK, s.CacheV, offset, layer);
            }
            _session.AttentionFloat32(s.QRot, cache?.Keys(layer) ?? s.KRot, cache?.Values(layer) ?? s.V,
                s.Attention, Config.NumAttentionHeads, Config.NumKvHeads, Config.HeadDim, offset + rows, offset);
            Project(s.Attention, l.O, s.Projection, s.BiasH);
            _session.AddFloat32(s.Hidden, s.Projection, s.Residual);
            _session.RmsNormFloat32(s.Residual, l.FfnNorm, s.Norm, Config.NormEpsilon);
            Project(s.Norm, l.Gate, s.Gate, s.BiasF);
            Project(s.Norm, l.Up, s.Up, s.BiasF);
            _session.SwiGluFloat32(s.Gate, s.Up, s.Activated);
            Project(s.Activated, l.Down, s.Projection, s.BiasH);
            _session.AddFloat32(s.Residual, s.Projection, s.Hidden);
        }
        _session.CopyFloat32(s.Hidden, (long)(rows - 1) * h, s.Last, 0, h);
        _session.RmsNormFloat32(s.Last, _outputNorm, s.LastNorm, Config.NormEpsilon);
        _session.LinearFloat32(s.LastNorm, _output, s.Logits);
        var floats = new Span<float>((void*)s.HostLogits.DataPointer, Config.VocabSize);
        _session.Download(s.Logits, floats);
        for (int i = 0; i < floats.Length; i++)
        {
            if (!float.IsFinite(floats[i])) throw new ArithmeticException("Ascend forward produced non-finite logits.");
        }
        return new TensorView(new(1, Config.VocabSize), DType.Float32, -1, s.HostLogits.DataPointer);
    }

    private void Project(AscendTensor input, Projection projection, AscendTensor output, AscendTensor scratch)
    {
        _session.LinearFloat32(input, projection.Weight, output);
        if (projection.Bias is { } bias)
        {
            _session.AddFloat32(output, bias, scratch);
            _session.CopyFloat32(scratch, 0, output, 0, output.ElementCount);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            // Session owns remaining caller caches as well as model allocations.
            try { _state?.Dispose(); _weights.Dispose(); }
            finally { _session.Dispose(); }
        }
    }
}
