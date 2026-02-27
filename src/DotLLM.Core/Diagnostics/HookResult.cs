namespace DotLLM.Core.Diagnostics;

/// <summary>
/// Result returned by an inference hook, indicating whether the activation should be
/// passed through unchanged or replaced with modified values.
/// </summary>
public abstract record HookResult
{
    private HookResult() { }

    /// <summary>
    /// Continue with the original activation unchanged. Used for read-only inspection.
    /// </summary>
    public static HookResult Continue { get; } = new ContinueResult();

    /// <summary>
    /// Replace the activation with new values. Used for interventions, steering, and ablation.
    /// </summary>
    /// <param name="activation">Replacement activation data. Must match the original shape.</param>
    /// <returns>A result that replaces the activation.</returns>
    public static HookResult Replace(float[] activation) => new ReplaceResult(activation);

    /// <summary>Read-only pass-through result.</summary>
    public sealed record ContinueResult : HookResult;

    /// <summary>Activation replacement result.</summary>
    /// <param name="Activation">Replacement activation data.</param>
    public sealed record ReplaceResult(float[] Activation) : HookResult;
}
