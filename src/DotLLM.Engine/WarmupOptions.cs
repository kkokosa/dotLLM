namespace DotLLM.Engine;

/// <summary>
/// Configuration for the warm-up pass that runs at startup to trigger JIT compilation
/// and CUDA kernel loading before the first real request.
/// </summary>
public sealed record WarmupOptions
{
    /// <summary>Default warm-up configuration (enabled, 3 iterations).</summary>
    public static WarmupOptions Default => new();

    /// <summary>Disabled warm-up.</summary>
    public static WarmupOptions Disabled => new() { Enabled = false };

    /// <summary>Whether warm-up is enabled. Default: true.</summary>
    public bool Enabled { get; init; } = true;

    /// <summary>
    /// Dummy prompt used for warm-up inference passes.
    /// Should be short but representative (exercises tokenizer encode, prefill, decode).
    /// </summary>
    public string DummyPrompt { get; init; } = "The quick brown fox jumps over the lazy dog.";

    /// <summary>
    /// Maximum tokens to generate per warm-up iteration.
    /// Kept small to minimize warm-up time while ensuring the full decode path is exercised.
    /// </summary>
    public int MaxTokens { get; init; } = 16;

    /// <summary>
    /// Number of warm-up iterations. Multiple iterations trigger .NET Tier-1 JIT promotion
    /// with Dynamic PGO profile data, producing optimized machine code for hot paths.
    /// Default: 3 (sufficient for Tier-0 → Tier-1 promotion of critical methods).
    /// </summary>
    public int Iterations { get; init; } = 3;
}
