using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;

namespace DotLLM.Ascend;

internal sealed class AscendWeights : IDisposable
{
    private readonly Dictionary<string, AscendTensor> _tensors = [];
    internal AscendTensor this[string name] => _tensors[name];
    internal AscendTensor? Optional(string name) => _tensors.GetValueOrDefault(name);
    internal long Bytes => _tensors.Values.Distinct().Sum(t => t.ByteCount);

    // Validate every descriptor before allocating device storage or dereferencing mmap data.
    internal static void Validate(GgufFile file, ModelConfig c)
    {
        var expected = new HashSet<string>();
        string arch = file.Metadata.GetString("general.architecture");
        if (arch is not ("llama" or "qwen2") ||
            c.AttentionType != AttentionType.GQA || c.NormType != NormType.RMSNorm ||
            c.ActivationFunction != ActivationFunction.SiLU || c.SlidingWindowSize is not null ||
            c.MlaConfig is not null || c.PositionEncodingType != PositionEncodingType.RoPE ||
            c.RoPEConfig is not { ScalingType: RoPEScalingType.None, ScalingFactor: 1, AttnFactor: 1 } rope)
            throw new NotSupportedException("Ascend supports dense llama/qwen2 RMSNorm/SwiGLU with unscaled RoPE.");
        if (c.HiddenSize <= 0 || c.VocabSize <= 0 || c.IntermediateSize <= 0 || c.NumLayers <= 0 ||
            c.NumAttentionHeads <= 0 || c.NumKvHeads <= 0 || c.HeadDim <= 0 ||
            c.MaxSequenceLength <= 0 || c.NumAttentionHeads % c.NumKvHeads != 0 ||
            checked(c.NumAttentionHeads * c.HeadDim) != c.HiddenSize ||
            rope.DimensionCount != c.HeadDim || c.HeadDim % 2 != 0 ||
            !float.IsFinite(rope.Theta) || rope.Theta <= 0 ||
            !float.IsFinite(c.NormEpsilon) || c.NormEpsilon <= 0)
            throw new InvalidDataException("Invalid Ascend dense model dimensions or RoPE configuration.");
        if (c.Architecture != (arch == "llama" ? Architecture.Llama : Architecture.Qwen) ||
            rope.Type != (arch == "llama" ? RoPEType.Norm : RoPEType.NeoX))
            throw new NotSupportedException("Model configuration architecture/RoPE pairing differs from the GGUF weight layout.");
        string scaling = file.Metadata.GetStringOrDefault($"{arch}.rope.scaling.type", "none");
        if (scaling != "none" || file.Metadata.GetUInt32OrDefault($"{arch}.expert_count", 0) != 0)
            throw new NotSupportedException("Scaled RoPE and MoE are unsupported by the Ascend baseline.");
        if (file.TensorsByName.ContainsKey("rope_freqs.weight"))
            throw new NotSupportedException("Per-frequency RoPE factors are unsupported.");
        Check("token_embd.weight", new(c.HiddenSize, c.VocabSize));
        Check("output_norm.weight", new(c.HiddenSize));
        if (file.TensorsByName.ContainsKey("output.weight"))
            Check("output.weight", new(c.HiddenSize, c.VocabSize));
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"blk.{i}.";
            Check(p + "attn_norm.weight", new(c.HiddenSize));
            Check(p + "ffn_norm.weight", new(c.HiddenSize));
            foreach (var (name, input, output) in new[]
            {
                ("attn_q", c.HiddenSize, c.HiddenSize),
                ("attn_k", c.HiddenSize, checked(c.NumKvHeads * c.HeadDim)),
                ("attn_v", c.HiddenSize, checked(c.NumKvHeads * c.HeadDim)),
                ("attn_output", c.HiddenSize, c.HiddenSize),
                ("ffn_gate", c.HiddenSize, c.IntermediateSize),
                ("ffn_up", c.HiddenSize, c.IntermediateSize),
                ("ffn_down", c.IntermediateSize, c.HiddenSize),
            })
            {
                Check(p + name + ".weight", new(input, output));
                if (file.TensorsByName.ContainsKey(p + name + ".bias"))
                    Check(p + name + ".bias", new(output));
            }
        }
        // Never silently ignore tensors carrying additional architecture semantics.
        foreach (var t in file.Tensors)
            if (!expected.Contains(t.Name))
                throw new NotSupportedException($"Unsupported Ascend tensor '{t.Name}'.");
        return;

        void Check(string name, TensorShape shape)
        {
            if (!file.TensorsByName.TryGetValue(name, out var t) || t.Shape != shape)
                throw new InvalidDataException($"Ascend expected tensor '{name}' with GGUF shape {shape}.");
            if (t.QuantizationType is not (QuantizationType.F32 or QuantizationType.F16 or QuantizationType.Q8_0))
                throw new NotSupportedException($"Ascend tensor '{name}': only F32/F16/Q8_0 storage is supported.");
            if (t.QuantizationType == QuantizationType.Q8_0 && shape[0] % 32 != 0)
                throw new InvalidDataException($"Q8_0 row '{name}' must be divisible by 32.");
            expected.Add(name);
        }
    }

    internal static AscendWeights Load(GgufFile file, AscendSession session)
    {
        var result = new AscendWeights();
        try
        {
            foreach (var t in file.Tensors)
                result._tensors.Add(t.Name, Upload(t, file.DataBasePointer, session));
            if (!result._tensors.ContainsKey("output.weight"))
                result._tensors.Add("output.weight", result["token_embd.weight"]);
            return result;
        }
        catch { result.Dispose(); throw; }
    }

    private static unsafe AscendTensor Upload(GgufTensorDescriptor t, nint dataBase, AscendSession session)
    {
        int columns = t.Shape[0], rows = t.Shape.Rank == 1 ? 1 : t.Shape[1];
        var destination = session.AllocateFloat32(t.Shape.Rank == 1 ? new(columns) : new(rows, columns));
        try
        {
            // Bound host staging to ~1 MiB, including for multi-gigabyte embedding tables.
            int chunkRows = Math.Max(1, 262144 / columns);
            int chunkElements = checked(Math.Min(chunkRows, rows) * columns);
            using var floats = UnmanagedTensor.Allocate(new(chunkElements), DType.Float32);
            long rowBytes = Dequantize.RowByteSize(columns, t.QuantizationType);
            nint source = dataBase + checked((nint)t.DataOffset);
            for (int row = 0; row < rows; row += chunkRows)
            {
                int count = checked(Math.Min(chunkRows, rows - row) * columns);
                var f = new Span<float>((void*)floats.DataPointer, count);
                Dequantize.ToFloat32(source + checked((nint)(row * rowBytes)), count, t.QuantizationType, f);
                for (int j = 0; j < count; j++)
                    if (!float.IsFinite(f[j]))
                        throw new InvalidDataException($"Tensor '{t.Name}' contains non-finite weights.");
                session.UploadSlice(destination, (long)row * columns, f);
            }
            return destination;
        }
        catch { destination.Dispose(); throw; }
    }

    public void Dispose()
    {
        foreach (var t in _tensors.Values.Distinct()) t.Dispose();
        _tensors.Clear();
    }
}
