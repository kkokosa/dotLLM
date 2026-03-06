using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotLLM.Benchmarks.Columns;

/// <summary>
/// Metrics written by the BDN subprocess during [GlobalCleanup] and read by custom IColumn in the host process.
/// File-based bridge to cross the BDN subprocess boundary.
/// </summary>
internal sealed record InferenceMetricsFile(
    [property: JsonPropertyName("medianPrefillTokPerSec")] double MedianPrefillTokPerSec,
    [property: JsonPropertyName("medianDecodeTokPerSec")] double MedianDecodeTokPerSec,
    [property: JsonPropertyName("medianPrefillMs")] double MedianPrefillMs,
    [property: JsonPropertyName("medianDecodeMs")] double MedianDecodeMs,
    [property: JsonPropertyName("prefillTokenCount")] int PrefillTokenCount,
    [property: JsonPropertyName("decodeTokenCount")] int DecodeTokenCount,
    [property: JsonPropertyName("iterations")] int Iterations)
{
    private static readonly JsonSerializerOptions s_jsonOptions = new()
    {
        WriteIndented = true
    };

    private static string GetMetricsDir() =>
        Path.Combine(Path.GetTempPath(), "dotllm-bdn-metrics");

    private static string GetFilePath(string key) =>
        Path.Combine(GetMetricsDir(), $"{key}.json");

    private static string GetFilePath(BenchmarkModel model) =>
        GetFilePath(model.ToString());

    /// <summary>
    /// Writes metrics to a temp JSON file keyed by model enum name.
    /// </summary>
    public static void Write(BenchmarkModel model, InferenceMetricsFile data)
        => Write(model.ToString(), data);

    /// <summary>
    /// Writes metrics to a temp JSON file keyed by an arbitrary string key.
    /// Used when <c>DOTLLM_BENCH_MODEL_PATH</c> overrides the model enum.
    /// </summary>
    public static void Write(string modelKey, InferenceMetricsFile data)
    {
        string dir = GetMetricsDir();
        Directory.CreateDirectory(dir);
        string path = GetFilePath(modelKey);
        string json = JsonSerializer.Serialize(data, s_jsonOptions);
        File.WriteAllText(path, json);
    }

    /// <summary>
    /// Attempts to read metrics from the temp JSON file for the given model enum.
    /// </summary>
    public static bool TryRead(BenchmarkModel model, out InferenceMetricsFile? data)
        => TryRead(model.ToString(), out data);

    /// <summary>
    /// Attempts to read metrics from the temp JSON file for the given string key.
    /// </summary>
    public static bool TryRead(string modelKey, out InferenceMetricsFile? data)
    {
        string path = GetFilePath(modelKey);
        if (!File.Exists(path))
        {
            data = null;
            return false;
        }

        try
        {
            string json = File.ReadAllText(path);
            data = JsonSerializer.Deserialize<InferenceMetricsFile>(json);
            return data is not null;
        }
        catch
        {
            data = null;
            return false;
        }
    }
}
