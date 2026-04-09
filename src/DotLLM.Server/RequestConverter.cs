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
    public static ChatMessage[] ToMessages(ChatMessageDto[] dtos)
    {
        var result = new ChatMessage[dtos.Length];
        for (int i = 0; i < dtos.Length; i++)
        {
            var d = dtos[i];
            ToolCall[]? toolCalls = null;
            if (d.ToolCalls is { Length: > 0 })
            {
                toolCalls = new ToolCall[d.ToolCalls.Length];
                for (int j = 0; j < d.ToolCalls.Length; j++)
                {
                    var tc = d.ToolCalls[j];
                    toolCalls[j] = new ToolCall(tc.Id, tc.Function.Name, tc.Function.Arguments);
                }
            }

            result[i] = new ChatMessage
            {
                Role = d.Role,
                Content = d.Content ?? "",
                ToolCalls = toolCalls,
                ToolCallId = d.ToolCallId,
            };
        }
        return result;
    }

    /// <summary>
    /// Converts tool definition DTOs to engine ToolDefinition records.
    /// </summary>
    public static ToolDefinition[]? ToTools(ToolDefinitionDto[]? dtos)
    {
        if (dtos is null) return null;
        var result = new ToolDefinition[dtos.Length];
        for (int i = 0; i < dtos.Length; i++)
        {
            var d = dtos[i];
            result[i] = new ToolDefinition(
                d.Function.Name,
                d.Function.Description ?? "",
                d.Function.Parameters?.GetRawText() ?? "{}");
        }
        return result;
    }

    /// <summary>
    /// Builds <see cref="InferenceOptions"/> from a chat completion request.
    /// </summary>
    public static InferenceOptions ToInferenceOptions(ChatCompletionRequest request,
        IReadOnlyList<string> stopSequences, SamplingDefaults defaults, ThreadingConfig threading)
    {
        var allStops = new List<string>(stopSequences);
        AddRequestStopSequences(allStops, request.Stop);

        return new InferenceOptions
        {
            Temperature = request.Temperature ?? defaults.Temperature,
            TopK = request.TopK ?? defaults.TopK,
            TopP = request.TopP ?? defaults.TopP,
            MinP = request.MinP ?? defaults.MinP,
            RepetitionPenalty = request.RepetitionPenalty ?? defaults.RepetitionPenalty,
            MaxTokens = request.MaxTokens ?? defaults.MaxTokens,
            Seed = request.Seed ?? defaults.Seed,
            StopSequences = allStops,
            ResponseFormat = ParseResponseFormat(request.ResponseFormat),
            Logprobs = request.Logprobs ?? false,
            TopLogprobs = Math.Clamp(request.TopLogprobs ?? 0, 0, 20),
            Threading = threading,
        };
    }

    /// <summary>
    /// Builds <see cref="InferenceOptions"/> from a raw completion request.
    /// </summary>
    public static InferenceOptions ToInferenceOptions(CompletionRequest request,
        SamplingDefaults defaults, ThreadingConfig threading)
    {
        var stops = new List<string>();
        AddRequestStopSequences(stops, request.Stop);

        return new InferenceOptions
        {
            Temperature = request.Temperature ?? defaults.Temperature,
            TopK = request.TopK ?? defaults.TopK,
            TopP = request.TopP ?? defaults.TopP,
            MinP = request.MinP ?? defaults.MinP,
            RepetitionPenalty = request.RepetitionPenalty ?? defaults.RepetitionPenalty,
            MaxTokens = request.MaxTokens ?? defaults.MaxTokens,
            Seed = request.Seed ?? defaults.Seed,
            StopSequences = stops,
            ResponseFormat = ParseResponseFormat(request.ResponseFormat),
            Logprobs = request.Logprobs ?? false,
            TopLogprobs = Math.Clamp(request.TopLogprobs ?? 0, 0, 20),
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
    public static ToolCallDto[] ToToolCallDtos(ToolCall[] toolCalls)
    {
        var result = new ToolCallDto[toolCalls.Length];
        for (int i = 0; i < toolCalls.Length; i++)
        {
            var tc = toolCalls[i];
            result[i] = new ToolCallDto
            {
                Id = tc.Id,
                Function = new ToolCallFunctionDto
                {
                    Name = tc.FunctionName,
                    Arguments = tc.Arguments,
                }
            };
        }
        return result;
    }

    /// <summary>
    /// Generates a unique request ID in the OpenAI format.
    /// </summary>
    public static string GenerateRequestId() => $"chatcmpl-{Guid.NewGuid():N}";

    /// <summary>
    /// Converts a single engine <see cref="TokenLogprobInfo"/> to a DTO for one streaming token.
    /// </summary>
    public static LogprobsDto ToLogprobsDto(TokenLogprobInfo info)
    {
        return new LogprobsDto
        {
            Content = [ToTokenLogprobDto(info)]
        };
    }

    /// <summary>
    /// Converts an array of engine <see cref="TokenLogprobInfo"/> to a DTO for a non-streaming response.
    /// </summary>
    public static LogprobsDto ToLogprobsDto(TokenLogprobInfo[] infos)
    {
        var content = new TokenLogprobDto[infos.Length];
        for (int i = 0; i < infos.Length; i++)
            content[i] = ToTokenLogprobDto(infos[i]);
        return new LogprobsDto { Content = content };
    }

    private static TokenLogprobDto ToTokenLogprobDto(TokenLogprobInfo info)
    {
        TopLogprobDto[] topLogprobs;
        if (info.TopLogprobs is { Length: > 0 })
        {
            topLogprobs = new TopLogprobDto[info.TopLogprobs.Length];
            for (int i = 0; i < info.TopLogprobs.Length; i++)
            {
                var t = info.TopLogprobs[i];
                topLogprobs[i] = new TopLogprobDto
                {
                    Token = t.Token,
                    Logprob = t.Logprob,
                    Bytes = t.Bytes,
                };
            }
        }
        else
        {
            topLogprobs = [];
        }

        return new TokenLogprobDto
        {
            Token = info.Token,
            Logprob = info.Logprob,
            Bytes = info.Bytes,
            TopLogprobs = topLogprobs,
        };
    }

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
