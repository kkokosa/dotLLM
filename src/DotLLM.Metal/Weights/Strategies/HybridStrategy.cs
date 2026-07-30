using DotLLM.Core.Configuration;

namespace DotLLM.Metal.Weights.Strategies;

/// <summary>Best of both worlds, ~1.5× RAM — for production servers.</summary>
public sealed class HybridStrategy : IWeightLoadStrategy
{
    private readonly DequantToFp16Strategy _fp16 = new();

    /// <inheritdoc/>
    public string Name => "Hybrid";

    /// <inheritdoc/>
    public long EstimateBytesFor(int totalElements, QuantizationType sourceFormat)
        => (long)totalElements * sizeof(ushort);

    /// <inheritdoc/>
    public LoadedWeight Load(MetalContext ctx, in TensorSource src)
    {
        LoadedWeight fp16Result = _fp16.Load(ctx, src);

        return fp16Result with {
            QuantizedPointer = src.MmapPointer,    // on garde aussi le mmap
            QuantizedFormat  = src.Format,
        };
    }
}
