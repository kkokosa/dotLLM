using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;

namespace DotLLM.Benchmarks.Columns;

/// <summary>
/// Custom BDN column that displays median prefill tokens/sec from the file-based metrics bridge.
/// </summary>
internal sealed class PrefillTokPerSecColumn : IColumn
{
    public string Id => nameof(PrefillTokPerSecColumn);
    public string ColumnName => "Prefill tok/s";
    public bool AlwaysShow => true;
    public ColumnCategory Category => ColumnCategory.Custom;
    public int PriorityInCategory => 0;
    public bool IsNumeric => true;
    public UnitType UnitType => UnitType.Dimensionless;
    public string Legend => "Median prefill throughput (tokens per second)";

    public string GetValue(Summary summary, BenchmarkCase benchmarkCase)
        => GetValue(summary, benchmarkCase, SummaryStyle.Default);

    public string GetValue(Summary summary, BenchmarkCase benchmarkCase, SummaryStyle style)
    {
        var key = ColumnHelpers.TryGetMetricsKey(benchmarkCase);
        if (key is null)
            return "N/A";

        if (!InferenceMetricsFile.TryRead(key, out var data) || data is null)
            return "N/A";

        return data.MedianPrefillTokPerSec.ToString("F1", style.CultureInfo);
    }

    public bool IsAvailable(Summary summary) => true;

    public bool IsDefault(Summary summary, BenchmarkCase benchmarkCase) => true;
}
