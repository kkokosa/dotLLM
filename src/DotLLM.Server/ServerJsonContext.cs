using System.Text.Json.Serialization;
using DotLLM.Server.Models;

namespace DotLLM.Server;

/// <summary>
/// Source-generated JSON serializer context for all server DTOs.
/// Eliminates reflection overhead and enables AOT compilation.
/// </summary>
[JsonSerializable(typeof(ChatCompletionRequest))]
[JsonSerializable(typeof(ChatCompletionResponse))]
[JsonSerializable(typeof(ChatCompletionChunk))]
[JsonSerializable(typeof(CompletionRequest))]
[JsonSerializable(typeof(CompletionResponse))]
[JsonSerializable(typeof(CompletionChunk))]
[JsonSerializable(typeof(TokenizeRequest))]
[JsonSerializable(typeof(TokenizeResponse))]
[JsonSerializable(typeof(DetokenizeRequest))]
[JsonSerializable(typeof(DetokenizeResponse))]
[JsonSerializable(typeof(ModelListResponse))]
[JsonSerializable(typeof(PropsResponse))]
[JsonSerializable(typeof(SamplingDefaultsDto))]
[JsonSerializable(typeof(AvailableModelsResponse))]
[JsonSerializable(typeof(ModelLoadRequest))]
[JsonSerializable(typeof(ModelLoadResponse))]
[JsonSerializable(typeof(TimingsDto))]
[JsonSerializable(typeof(ModelInspectResponse))]
[JsonSerializable(typeof(ErrorResponse))]
[JsonSerializable(typeof(StatusResponse))]
[JsonSourceGenerationOptions(
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
internal partial class ServerJsonContext : JsonSerializerContext;
