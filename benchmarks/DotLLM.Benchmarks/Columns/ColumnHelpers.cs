using BenchmarkDotNet.Running;
using DotLLM.Benchmarks.Lora;

namespace DotLLM.Benchmarks.Columns;

/// <summary>
/// Shared helpers for custom inference metric columns.
/// </summary>
internal static class ColumnHelpers
{
    /// <summary>
    /// Extracts the <see cref="BenchmarkModel"/> parameter from a benchmark case, if present.
    /// </summary>
    public static BenchmarkModel? TryGetModel(BenchmarkCase benchmarkCase)
    {
        if (!benchmarkCase.HasParameters)
            return null;

        var items = benchmarkCase.Parameters.Items;
        foreach (var item in items)
        {
            if (item.Value is BenchmarkModel model)
                return model;
        }

        return null;
    }

    /// <summary>
    /// Returns the metrics key for a benchmark case. Resolution order:
    /// <list type="number">
    /// <item><see cref="LoraMacroBenchmarks"/> cases — composite key from
    /// (model-label, variant, scenario), matching what the bench writes.</item>
    /// <item><c>DOTLLM_BENCH_MODEL_PATH</c> env var — filename stem.</item>
    /// <item><see cref="BenchmarkModel"/> param — the enum name.</item>
    /// </list>
    /// </summary>
    public static string? TryGetMetricsKey(BenchmarkCase benchmarkCase)
    {
        // LoRA macro-bench cases write a composite key — match it here so the
        // shared Prefill / Decode columns surface the right value per case
        // even though it carries no BenchmarkModel parameter.
        var loraKey = TryGetLoraMacroKey(benchmarkCase);
        if (loraKey is not null)
            return loraKey;

        var envPath = Environment.GetEnvironmentVariable("DOTLLM_BENCH_MODEL_PATH");
        if (!string.IsNullOrEmpty(envPath))
            return Path.GetFileNameWithoutExtension(envPath);

        var model = TryGetModel(benchmarkCase);
        return model?.ToString();
    }

    /// <summary>
    /// Composes the <see cref="LoraMacroBenchmarks"/> metrics key from a benchmark
    /// case's parameters. Returns <c>null</c> when the case is not a LoRA macro-bench
    /// case (i.e. doesn't carry both <see cref="LoraVariant"/> and <see cref="LoraScenario"/>).
    /// </summary>
    private static string? TryGetLoraMacroKey(BenchmarkCase benchmarkCase)
    {
        if (!benchmarkCase.HasParameters) return null;

        LoraVariant? variant = null;
        LoraScenario? scenario = null;
        foreach (var item in benchmarkCase.Parameters.Items)
        {
            if (item.Value is LoraVariant v) variant = v;
            else if (item.Value is LoraScenario s) scenario = s;
        }
        if (variant is null || scenario is null) return null;

        // The fixture label is determined at runtime; we cannot recover it
        // from BDN params. Probe the on-disk metrics dir for the first key
        // matching the expected suffix — there will be one per (variant, scenario).
        string suffix = $"_{variant.Value}_{scenario.Value}";
        string dir = Path.Combine(Path.GetTempPath(), "dotllm-bdn-metrics");
        if (!Directory.Exists(dir)) return null;

        foreach (var file in Directory.EnumerateFiles(dir, "Lora_*.json"))
        {
            string stem = Path.GetFileNameWithoutExtension(file);
            if (stem.EndsWith(suffix, StringComparison.Ordinal))
                return stem;
        }
        return null;
    }
}
