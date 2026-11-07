using System.Text.Json;
using BenchmarkDotNet.Attributes;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server;
using DotLLM.Server.Models;

namespace DotLLM.Benchmarks;

/// <summary>Measures stop-list conversion while retaining the complete options allocation.</summary>
[MemoryDiagnoser]
public class RequestConversionBenchmarks
{
    private readonly SamplingDefaults _defaults = new();
    private readonly ThreadingConfig _threading = ThreadingConfig.SingleThreaded;
    private readonly string[] _chatStops = ["eos", "chat-end", "tool-end", "im-end", "eot", "end"];
    private CompletionRequest _request = null!;
    private ChatCompletionRequest _chatRequest = null!;
    private JsonElement? _stop;

    /// <summary>Number of default chat-template stop strings.</summary>
    [Params(0, 6)]
    public int DefaultStopCount { get; set; }

    /// <summary>Optional request stop field.</summary>
    [Params("none", "string", "array")]
    public string StopKind { get; set; } = "none";

    /// <summary>Creates pre-parsed request data outside the timed region.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _stop = StopKind switch
        {
            "string" => Parse("\"stop\""),
            "array" => Parse("[\"stop\",\"finish\"]"),
            _ => null,
        };
        _request = new CompletionRequest { Prompt = "hello", Stop = _stop };
        _chatRequest = new ChatCompletionRequest { Messages = [], Stop = _stop };
    }

    /// <summary>Historical list-based conversion, including growth when defaults are present.</summary>
    [Benchmark(Baseline = true)]
    public InferenceOptions ListBased()
    {
        var stops = new List<string>(Defaults);
        if (_stop is { } stop)
        {
            if (stop.ValueKind == JsonValueKind.String)
            {
                string? value = stop.GetString();
                if (!string.IsNullOrEmpty(value)) stops.Add(value);
            }
            else if (stop.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in stop.EnumerateArray())
                {
                    string? value = item.GetString();
                    if (!string.IsNullOrEmpty(value)) stops.Add(value);
                }
            }
        }
        return Options(stops);
    }

    /// <summary>Converts through the public API with an empty/default fast path and reserved list capacity.</summary>
    [Benchmark]
    public InferenceOptions CapacityAware()
    {
        if (DefaultStopCount == 0)
            return RequestConverter.ToInferenceOptions(_request, _defaults, _threading);
        return RequestConverter.ToInferenceOptions(
            _chatRequest, Defaults, _defaults, _threading);
    }

    private IReadOnlyList<string> Defaults => DefaultStopCount == 0 ? Array.Empty<string>() : _chatStops;

    private InferenceOptions Options(IReadOnlyList<string> stops) => new()
    {
        Temperature = _defaults.Temperature,
        TopK = _defaults.TopK,
        TopP = _defaults.TopP,
        MinP = _defaults.MinP,
        RepetitionPenalty = _defaults.RepetitionPenalty,
        MaxTokens = _defaults.MaxTokens,
        Seed = _defaults.Seed,
        StopSequences = stops,
        Threading = _threading,
    };

    private static JsonElement Parse(string json)
    {
        using var document = JsonDocument.Parse(json);
        return document.RootElement.Clone();
    }
}
