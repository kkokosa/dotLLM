using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;

namespace DotLLM.Server;

/// <summary>
/// Converts between OpenAI-compatible DTOs and dotLLM engine types.
/// </summary>
public static class RequestConverter
{
    /// <summary>
    /// Converts chat message DTOs to engine ChatMessage records.
    /// </summary>
    public static ChatMessage[] ToMessages(ChatMessageDto[] dtos) =>
        dtos.Select(d => new ChatMessage
        {
            Role = d.Role,
            Content = d.Content ?? "",
            ToolCalls = d.ToolCalls?.Select(tc => new ToolCall(
                tc.Id, tc.Function.Name, tc.Function.Arguments)).ToArray(),
            ToolCallId = d.ToolCallId,
        }).ToArray();

    /// <summary>
    /// Converts tool definition DTOs to engine ToolDefinition records.
    /// </summary>
    public static ToolDefinition[]? ToTools(ToolDefinitionDto[]? dtos) =>
        dtos?.Select(d => new ToolDefinition(
            d.Function.Name,
            d.Function.Description ?? "",
            d.Function.Parameters?.GetRawText() ?? "{}")).ToArray();

    /// <summary>
    /// Builds <see cref="InferenceOptions"/> from a chat completion request.
    /// </summary>
    public static InferenceOptions ToInferenceOptions(ChatCompletionRequest request,
        IReadOnlyList<string> stopSequences, ThreadingConfig threading)
    {
        var allStops = new List<string>(stopSequences);
        AddRequestStopSequences(allStops, request.Stop);

        return new InferenceOptions
        {
            Temperature = request.Temperature ?? 0.7f,
            TopK = request.TopK ?? 0,
            TopP = request.TopP ?? 1.0f,
            MinP = request.MinP ?? 0f,
            RepetitionPenalty = request.RepetitionPenalty ?? 1.0f,
            MaxTokens = request.MaxTokens ?? 2048,
            Seed = request.Seed,
            StopSequences = allStops,
            ResponseFormat = ParseResponseFormat(request.ResponseFormat),
            Threading = threading,
        };
    }

    /// <summary>
    /// Builds <see cref="InferenceOptions"/> from a raw completion request.
    /// </summary>
    public static InferenceOptions ToInferenceOptions(CompletionRequest request,
        ThreadingConfig threading)
    {
        var stops = new List<string>();
        AddRequestStopSequences(stops, request.Stop);

        return new InferenceOptions
        {
            Temperature = request.Temperature ?? 0.7f,
            TopK = request.TopK ?? 0,
            TopP = request.TopP ?? 1.0f,
            MinP = request.MinP ?? 0f,
            RepetitionPenalty = request.RepetitionPenalty ?? 1.0f,
            MaxTokens = request.MaxTokens ?? 2048,
            Seed = request.Seed,
            StopSequences = stops,
            ResponseFormat = ParseResponseFormat(request.ResponseFormat),
            Threading = threading,
        };
    }

    /// <summary>
    /// Parses the tool_choice JSON element into a <see cref="ToolChoice"/> record.
    /// </summary>
    public static ToolChoice ParseToolChoice(JsonElement? element)
    {
        if (element is null || element.Value.ValueKind == JsonValueKind.Undefined)
            return new ToolChoice.Auto();

        if (element.Value.ValueKind == JsonValueKind.String)
        {
            return element.Value.GetString() switch
            {
                "none" => new ToolChoice.None(),
                "required" => new ToolChoice.Required(),
                _ => new ToolChoice.Auto(),
            };
        }

        if (element.Value.ValueKind == JsonValueKind.Object &&
            element.Value.TryGetProperty("function", out var funcProp) &&
            funcProp.TryGetProperty("name", out var nameProp))
        {
            return new ToolChoice.Function(nameProp.GetString()!);
        }

        return new ToolChoice.Auto();
    }

    /// <summary>
    /// Parses the response_format JSON element into a <see cref="ResponseFormat"/> record.
    /// </summary>
    public static ResponseFormat? ParseResponseFormat(JsonElement? element)
    {
        if (element is null || element.Value.ValueKind != JsonValueKind.Object)
            return null;

        if (!element.Value.TryGetProperty("type", out var typeProp))
            return null;

        return typeProp.GetString() switch
        {
            "json_object" => new ResponseFormat.JsonObject(),
            "json_schema" when element.Value.TryGetProperty("json_schema", out var schemaProp) =>
                new ResponseFormat.JsonSchema
                {
                    Schema = schemaProp.TryGetProperty("schema", out var s)
                        ? s.GetRawText()
                        : schemaProp.GetRawText(),
                    Name = schemaProp.TryGetProperty("name", out var n) ? n.GetString() : null,
                },
            _ => null,
        };
    }

    /// <summary>
    /// Converts a <see cref="FinishReason"/> to the OpenAI-compatible string.
    /// </summary>
    public static string ToFinishReasonString(FinishReason reason) => reason switch
    {
        FinishReason.Stop => "stop",
        FinishReason.Length => "length",
        FinishReason.ToolCalls => "tool_calls",
        _ => "stop",
    };

    /// <summary>
    /// Converts engine <see cref="ToolCall"/> records to DTOs.
    /// </summary>
    public static ToolCallDto[] ToToolCallDtos(ToolCall[] toolCalls) =>
        toolCalls.Select(tc => new ToolCallDto
        {
            Id = tc.Id,
            Function = new ToolCallFunctionDto
            {
                Name = tc.FunctionName,
                Arguments = tc.Arguments,
            }
        }).ToArray();

    /// <summary>
    /// Generates a unique request ID in the OpenAI format.
    /// </summary>
    public static string GenerateRequestId() => $"chatcmpl-{Guid.NewGuid():N}";

    private static void AddRequestStopSequences(List<string> target, JsonElement? stopElement)
    {
        if (stopElement is null || stopElement.Value.ValueKind == JsonValueKind.Undefined)
            return;

        if (stopElement.Value.ValueKind == JsonValueKind.String)
        {
            string? s = stopElement.Value.GetString();
            if (!string.IsNullOrEmpty(s))
                target.Add(s);
        }
        else if (stopElement.Value.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in stopElement.Value.EnumerateArray())
            {
                string? s = item.GetString();
                if (!string.IsNullOrEmpty(s))
                    target.Add(s);
            }
        }
    }
}
