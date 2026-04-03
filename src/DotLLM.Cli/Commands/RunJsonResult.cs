using System.Text.Json.Serialization;

namespace DotLLM.Cli.Commands;

/// <summary>
/// JSON output DTO for <c>dotllm run --json</c>.
/// Replaces anonymous type to enable AOT-compatible source-generated serialization.
/// </summary>
internal sealed record RunJsonResult
{
    public required string Text { get; init; }
    public required string Prompt { get; init; }
    public required string Model { get; init; }
    public required string Architecture { get; init; }
    public required string FinishReason { get; init; }
    public RunToolCallDto[]? ToolCalls { get; init; }
    public required RunUsageDto Usage { get; init; }
    public required RunTimingsDto Timings { get; init; }
    public required RunMemoryDto Memory { get; init; }
}

/// <summary>Detected tool call in generated output.</summary>
internal sealed record RunToolCallDto
{
    public required string Id { get; init; }
    public required string FunctionName { get; init; }
    public required string Arguments { get; init; }
}

/// <summary>Token usage statistics.</summary>
internal sealed record RunUsageDto
{
    public int PromptTokens { get; init; }
    public int GeneratedTokens { get; init; }
}

/// <summary>Inference timing breakdown.</summary>
internal sealed record RunTimingsDto
{
    public double LoadMs { get; init; }
    public double PrefillMs { get; init; }
    public double DecodeMs { get; init; }
    public double SamplingMs { get; init; }
    public double TotalMs { get; init; }
    public double PrefillTokS { get; init; }
    public double DecodeTokS { get; init; }
}

/// <summary>Memory usage breakdown.</summary>
internal sealed record RunMemoryDto
{
    public long WeightsBytes { get; init; }
    public long ComputeBytes { get; init; }
    public long KvCacheBytes { get; init; }
    public long TotalBytes { get; init; }
}
