namespace DotLLM.Benchmarks.Lora;

/// <summary>
/// Resolves a local GGUF checkpoint for the LoRA macro-bench, in priority order:
/// <list type="number">
/// <item>The <c>DOTLLM_BENCH_MODEL_PATH</c> env var (manual override).</item>
/// <item>A TinyLlama GGUF anywhere under <c>~/.dotllm/test-cache/</c>.</item>
/// <item>Llama-3.2-1B-Instruct Q8_0 GGUF (the closest available stand-in).</item>
/// <item>SmolLM-135M Q8_0 GGUF (smallest fallback).</item>
/// </list>
/// We deliberately do NOT trigger downloads here — Phase 4d.3 is a measurement
/// follow-up, not a fixture provisioner, and the parent agent is offline-tolerant.
/// </summary>
internal static class LoraMacroBenchFixture
{
    /// <summary>
    /// Searches the local test-cache for a usable checkpoint. Returns the
    /// resolved path + a short label, or <c>null</c> + a skip reason.
    /// </summary>
    public static string? ResolveCheckpoint(out string label, out string? skipReason)
    {
        // 1. Manual override.
        var envPath = Environment.GetEnvironmentVariable("DOTLLM_BENCH_MODEL_PATH");
        if (!string.IsNullOrEmpty(envPath) && File.Exists(envPath))
        {
            label = Path.GetFileNameWithoutExtension(envPath);
            skipReason = null;
            return envPath;
        }

        string cacheRoot = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "test-cache");

        if (!Directory.Exists(cacheRoot))
        {
            label = string.Empty;
            skipReason = $"test-cache directory does not exist: {cacheRoot}";
            return null;
        }

        // 2. TinyLlama GGUF — preferred per Phase 4d.3 spec.
        //    Probe common filename stems used by HF mirrors.
        foreach (var pat in TinyLlamaPatterns)
        {
            foreach (var hit in Directory.EnumerateFiles(cacheRoot, pat, SearchOption.AllDirectories))
            {
                label = "TinyLlama";
                skipReason = null;
                return hit;
            }
        }

        // 3. Llama-3.2-1B Q8_0 — same scale class, exercises the same forward
        //    path on the same architecture family.
        string l32 = Path.Combine(cacheRoot,
            "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf");
        if (File.Exists(l32))
        {
            label = "Llama32_1B";
            skipReason = null;
            return l32;
        }

        // 4. SmolLM-135M Q8_0 — smallest fallback. Useful for CI smoke runs but
        //    note its absolute tok/s aren't representative of TinyLlama-class
        //    bandwidth pressure — the delta % vs base is still meaningful.
        string smol = Path.Combine(cacheRoot,
            "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");
        if (File.Exists(smol))
        {
            label = "SmolLM_135M";
            skipReason = null;
            return smol;
        }

        // SmolLM2 alt path (we saw both Q8 variants on disk).
        string smol2 = Path.Combine(cacheRoot,
            "bartowski", "SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q8_0.gguf");
        if (File.Exists(smol2))
        {
            label = "SmolLM2_135M";
            skipReason = null;
            return smol2;
        }

        label = string.Empty;
        skipReason =
            "no usable GGUF checkpoint found under ~/.dotllm/test-cache/. "
          + "Set DOTLLM_BENCH_MODEL_PATH=<path-to-gguf> or place a TinyLlama / "
          + "Llama-3.2-1B / SmolLM-135M GGUF in the test-cache.";
        return null;
    }

    private static readonly string[] TinyLlamaPatterns =
    [
        "*TinyLlama*Q8_0*.gguf",
        "*tinyllama*q8_0*.gguf",
        "*TinyLlama*Q4_K_M*.gguf",
        "*tinyllama*q4_k_m*.gguf",
        "*TinyLlama*.gguf",
        "*tinyllama*.gguf",
    ];
}
