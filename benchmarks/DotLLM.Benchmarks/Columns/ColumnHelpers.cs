using BenchmarkDotNet.Running;

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
    /// Returns the metrics key for a benchmark case. When <c>DOTLLM_BENCH_MODEL_PATH</c> is set,
    /// uses the filename stem; otherwise falls back to the <see cref="BenchmarkModel"/> enum name.
    /// </summary>
    public static string? TryGetMetricsKey(BenchmarkCase benchmarkCase)
    {
        var envPath = Environment.GetEnvironmentVariable("DOTLLM_BENCH_MODEL_PATH");
        string? baseKey = !string.IsNullOrEmpty(envPath)
            ? Path.GetFileNameWithoutExtension(envPath)
            : TryGetModel(benchmarkCase)?.ToString();

        if (baseKey is null)
            return null;

        // MultiRuntimeConfig names each job after its target framework, so a job id of the
        // form "netN.M" identifies the runtime and selects that job's metrics file. Any other
        // id (BDN's generated "Job-XXXX") means a single-runtime run — use the plain key.
        string? jobId = benchmarkCase.Job?.Id;
        return IsTfm(jobId)
            ? InferenceMetricsFile.ComposeKey(baseKey, jobId!)
            : baseKey;
    }

    private static bool IsTfm(string? id) =>
        id is not null && id.StartsWith("net", StringComparison.Ordinal)
        && id.Length > 3 && char.IsDigit(id[3]);
}
