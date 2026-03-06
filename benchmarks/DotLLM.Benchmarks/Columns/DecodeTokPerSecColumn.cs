using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;

namespace DotLLM.Benchmarks.Columns;

/// <summary>
/// Custom BDN column that displays median decode tokens/sec from the file-based metrics bridge.
/// </summary>
internal sealed class DecodeTokPerSecColumn : IColumn
{
    public string Id => nameof(DecodeTokPerSecColumn);
    public string ColumnName => "Decode tok/s";
    public bool AlwaysShow => true;
    public ColumnCategory Category => ColumnCategory.Custom;
    public int PriorityInCategory => 1;
    public bool IsNumeric => true;
    public UnitType UnitType => UnitType.Dimensionless;
    public string Legend => "Median decode throughput (tokens per second)";

    public string GetValue(Summary summary, BenchmarkCase benchmarkCase)
        => GetValue(summary, benchmarkCase, SummaryStyle.Default);

    public string GetValue(Summary summary, BenchmarkCase benchmarkCase, SummaryStyle style)
    {
        var key = ColumnHelpers.TryGetMetricsKey(benchmarkCase);
        if (key is null)
            return "N/A";

        if (!InferenceMetricsFile.TryRead(key, out var data) || data is null)
            return "N/A";

        return data.MedianDecodeTokPerSec.ToString("F1", style.CultureInfo);
    }

    public bool IsAvailable(Summary summary) => true;

    public bool IsDefault(Summary summary, BenchmarkCase benchmarkCase) => true;
}
