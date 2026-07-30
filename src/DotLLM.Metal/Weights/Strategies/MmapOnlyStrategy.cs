using DotLLM.Core.Configuration;

namespace DotLLM.Metal.Weights.Strategies;

/// <summary>Smallest RAM, fastest decode — for chat on a Mac perso.</summary>
public sealed class MmapOnlyStrategy : IWeightLoadStrategy
{
    /// <inheritdoc/>
    public string Name => "MmapOnly";

    /// <inheritdoc/>
    public long EstimateBytesFor(int totalElements, QuantizationType sourceFormat)
        => 0;  // no allocation, mmap is shared with OS page cache

    /// <inheritdoc/>
    public LoadedWeight Load(MetalContext ctx, in TensorSource src)
        => new(
            QuantizedPointer: src.MmapPointer,    // direct mmap reference
            QuantizedFormat: src.Format,
            Fp16Pointer: 0,
            OwnsFp16Buffer: false,
            OutputDim: src.OutputDim,
            InputDim: src.InputDim);
}
