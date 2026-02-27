using DotLLM.Core.Tensors;

namespace DotLLM.Engine;

/// <summary>
/// A loaded LoRA adapter with its weight tensors.
/// </summary>
public record LoraAdapter
{
    /// <summary>Unique name identifying this adapter.</summary>
    public required string Name { get; init; }

    /// <summary>LoRA rank (r).</summary>
    public required int Rank { get; init; }

    /// <summary>LoRA alpha scaling factor.</summary>
    public required float Alpha { get; init; }

    /// <summary>Target module names (e.g., "q_proj", "v_proj").</summary>
    public required string[] TargetModules { get; init; }

    /// <summary>Per-layer A and B weight tensors keyed by layer name.</summary>
    public required Dictionary<string, (ITensor A, ITensor B)> Layers { get; init; }
}
