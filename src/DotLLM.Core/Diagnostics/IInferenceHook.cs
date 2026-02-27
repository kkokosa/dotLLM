namespace DotLLM.Core.Diagnostics;

/// <summary>
/// Hook for intercepting activations at specific points in the transformer forward pass.
/// Zero cost when no hooks are registered — callers null-check before invoking.
/// </summary>
public interface IInferenceHook
{
    /// <summary>The point in the forward pass where this hook should be called.</summary>
    HookPoint HookPoint { get; }

    /// <summary>
    /// Called when the activation at the configured hook point is available.
    /// </summary>
    /// <param name="activation">The current activation values (read-only view).</param>
    /// <param name="context">Context about the current layer, position, and step.</param>
    /// <returns>Whether to continue with the original activation or replace it.</returns>
    HookResult OnActivation(ReadOnlySpan<float> activation, HookContext context);
}
