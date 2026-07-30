using DotLLM.Core.Configuration;

namespace DotLLM.Metal.Weights;

/// <summary>
/// Decides what to do with each quantized weight tensor when loading
/// a model: keep the original quantized bytes, dequantize to FP16,
/// or both. The forward pass reads the resulting <see cref="LoadedWeight"/>
/// and picks the appropriate kernel automatically.
/// </summary>
public interface IWeightLoadStrategy
{
    /// <summary>Human-readable name for diagnostics ("MmapOnly", "DequantToFp16", "Hybrid").</summary>
    string Name { get; }

    /// <summary>
    /// Estimated bytes this strategy will allocate per tensor of the given size.
    /// Used by the loader to warn the user before loading a 70B model on a 16 GB Mac.
    /// </summary>
    long EstimateBytesFor(int totalElements, QuantizationType sourceFormat);

    /// <summary>
    /// Called once per quantized weight tensor during loading.
    /// Implementations may read the mmap, allocate fresh buffers, run dequant kernels.
    /// </summary>
    LoadedWeight Load(MetalContext ctx, in TensorSource source);
}

/// <summary>Where to find the original quantized data in the mmap'd GGUF.</summary>
public readonly ref struct TensorSource
{
    /// <summary>Creates a tensor source descriptor pointing into the mmap'd GGUF.</summary>
    public TensorSource(
        nint mmapPointer, long byteLength, QuantizationType format,
        int outputDim, int inputDim)
    {
        MmapPointer = mmapPointer;
        ByteLength  = byteLength;
        Format      = format;
        OutputDim   = outputDim;
        InputDim    = inputDim;
    }

    /// <summary>Absolute pointer to the start of the tensor's bytes in the mmap.</summary>
    public readonly nint             MmapPointer;

    /// <summary>Total bytes occupied by the tensor (depends on format).</summary>
    public readonly long             ByteLength;

    /// <summary>Storage / quantization format of the source bytes.</summary>
    public readonly QuantizationType Format;

    /// <summary>Output dimension (rows). For projections: outputDim of the linear layer.</summary>
    public readonly int              OutputDim;

    /// <summary>Input dimension (cols). 1 for 1-D tensors (norms, biases).</summary>
    public readonly int              InputDim;
}

/// <summary>
/// Result handed back to <see cref="MetalWeights"/>. Exactly one of
/// (<see cref="QuantizedPointer"/>, <see cref="Fp16Pointer"/>) may be 0,
/// or both may be non-zero in hybrid mode.
/// </summary>
public readonly record struct LoadedWeight(
    nint QuantizedPointer,
    QuantizationType QuantizedFormat,
    nint Fp16Pointer,
    bool OwnsFp16Buffer,
    int OutputDim,
    int InputDim);
