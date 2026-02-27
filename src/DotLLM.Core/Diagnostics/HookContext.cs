namespace DotLLM.Core.Diagnostics;

/// <summary>
/// Context provided to an inference hook when an activation is intercepted.
/// </summary>
/// <param name="LayerIndex">Index of the transformer layer (0-based). -1 for non-layer hooks.</param>
/// <param name="TokenPosition">Position of the token in the sequence.</param>
/// <param name="SequenceId">Identifier for the current sequence/request.</param>
/// <param name="CurrentStep">Current decoding step (0-based).</param>
public readonly record struct HookContext(
    int LayerIndex,
    int TokenPosition,
    int SequenceId,
    int CurrentStep);
