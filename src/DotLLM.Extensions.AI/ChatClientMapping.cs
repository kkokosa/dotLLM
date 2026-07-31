using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Tokenizers;
using Microsoft.Extensions.AI;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;
using EngineChatMessage = DotLLM.Tokenizers.ChatMessage;

namespace DotLLM.Extensions.AI;

/// <summary>
/// Pure translation between <c>Microsoft.Extensions.AI</c> types and dotLLM engine
/// types. Kept free of engine state so it is independently unit-testable; the
/// stateful generation lives in <see cref="DotLLMChatClient"/>.
/// </summary>
public static class ChatClientMapping
{
    // --- Input: Microsoft.Extensions.AI -> engine ---------------------------

    /// <summary>
    /// Flattens a Microsoft.Extensions.AI message list into the engine's linear
    /// <see cref="EngineChatMessage"/> list. Text blocks are concatenated;
    /// <see cref="FunctionCallContent"/> becomes assistant <see cref="ToolCall"/>s;
    /// <see cref="FunctionResultContent"/> becomes a <c>tool</c>-role message keyed
    /// by its call id.
    /// </summary>
    public static EngineChatMessage[] ToEngineMessages(IEnumerable<ChatMessage> messages)
    {
        var result = new List<EngineChatMessage>();

        foreach (var message in messages)
        {
            string role = ToRole(message.Role);
            var text = new StringBuilder();
            List<ToolCall>? toolCalls = null;

            foreach (var content in message.Contents)
            {
                switch (content)
                {
                    case TextContent tc when !string.IsNullOrEmpty(tc.Text):
                        if (text.Length > 0) text.Append('\n');
                        text.Append(tc.Text);
                        break;

                    case FunctionCallContent fc:
                        (toolCalls ??= []).Add(new ToolCall(
                            fc.CallId ?? "", fc.Name ?? "", SerializeArguments(fc.Arguments)));
                        break;

                    case FunctionResultContent fr:
                        // Tool results map to a dedicated tool-role message.
                        result.Add(new EngineChatMessage
                        {
                            Role = "tool",
                            Content = ResultToString(fr.Result),
                            ToolCallId = fr.CallId,
                        });
                        break;
                }
            }

            if (text.Length > 0 || toolCalls is not null)
            {
                result.Add(new EngineChatMessage
                {
                    Role = role,
                    Content = text.ToString(),
                    ToolCalls = toolCalls?.ToArray(),
                });
            }
        }

        return [.. result];
    }

    /// <summary>
    /// Maps a Microsoft.Extensions.AI role to the engine role string. Non-standard roles
    /// (e.g. <c>developer</c>) are passed through verbatim rather than coerced to
    /// <c>user</c>, so the chat template — not this adapter — decides how to render them.
    /// </summary>
    public static string ToRole(ChatRole role) =>
        role == ChatRole.System ? "system"
        : role == ChatRole.Assistant ? "assistant"
        : role == ChatRole.Tool ? "tool"
        : role == ChatRole.User ? "user"
        : string.IsNullOrWhiteSpace(role.Value) ? "user"
        : role.Value;

    /// <summary>Builds <see cref="InferenceOptions"/> from <see cref="ChatOptions"/>.</summary>
    public static InferenceOptions ToInferenceOptions(ChatOptions? options)
    {
        var defaults = new InferenceOptions();
        if (options is null)
            return defaults;

        var stops = options.StopSequences is { Count: > 0 }
            ? new List<string>(options.StopSequences)
            : null;

        return defaults with
        {
            Temperature = options.Temperature ?? defaults.Temperature,
            TopP = options.TopP ?? defaults.TopP,
            TopK = options.TopK ?? defaults.TopK,
            MaxTokens = options.MaxOutputTokens ?? defaults.MaxTokens,
            Seed = options.Seed.HasValue ? unchecked((int)options.Seed.Value) : defaults.Seed,
            StopSequences = stops ?? defaults.StopSequences,
            ResponseFormat = ToResponseFormat(options.ResponseFormat),
            // dotLLM-specific knobs without a first-class ChatOptions field can be
            // passed through AdditionalProperties for full fidelity.
            MinP = ReadFloat(options.AdditionalProperties, "min_p") ?? defaults.MinP,
            RepetitionPenalty =
                ReadFloat(options.AdditionalProperties, "repetition_penalty") ?? defaults.RepetitionPenalty,
        };
    }

    /// <summary>Maps a <see cref="ChatResponseFormat"/> to the engine constraint, if any.</summary>
    public static ResponseFormat? ToResponseFormat(ChatResponseFormat? format)
    {
        if (format is not ChatResponseFormatJson json)
            return null;

        if (json.Schema is JsonElement schema)
            return new ResponseFormat.JsonSchema { Schema = schema.GetRawText(), Name = json.SchemaName };

        return new ResponseFormat.JsonObject();
    }

    /// <summary>Extracts <see cref="ToolDefinition"/>s from the request's tools.</summary>
    public static ToolDefinition[]? ToToolDefinitions(ChatOptions? options)
    {
        if (options?.Tools is not { Count: > 0 } tools)
            return null;

        var result = new List<ToolDefinition>(tools.Count);
        foreach (var tool in tools)
        {
            if (tool is AIFunction fn)
                result.Add(new ToolDefinition(fn.Name, fn.Description ?? "", fn.JsonSchema.GetRawText()));
        }
        return result.Count > 0 ? [.. result] : null;
    }

    // --- Output: engine -> Microsoft.Extensions.AI --------------------------

    /// <summary>
    /// Builds the assistant message content. When tool calls are present the text is
    /// dropped and <see cref="FunctionCallContent"/> blocks are emitted (matching the
    /// OpenAI/Anthropic server endpoints).
    /// </summary>
    public static List<AIContent> ToResponseContents(string text, ToolCall[]? toolCalls)
    {
        var contents = new List<AIContent>();
        if (toolCalls is { Length: > 0 })
        {
            foreach (var tc in toolCalls)
            {
                contents.Add(new FunctionCallContent(
                    string.IsNullOrEmpty(tc.Id) ? Guid.NewGuid().ToString("N") : tc.Id,
                    tc.FunctionName,
                    ParseArguments(tc.Arguments)));
            }
        }
        else if (!string.IsNullOrEmpty(text))
        {
            contents.Add(new TextContent(text));
        }
        return contents;
    }

    /// <summary>Maps an engine <see cref="FinishReason"/> to a <see cref="ChatFinishReason"/>.</summary>
    public static ChatFinishReason ToChatFinishReason(FinishReason reason) => reason switch
    {
        FinishReason.Length => ChatFinishReason.Length,
        FinishReason.ToolCalls => ChatFinishReason.ToolCalls,
        _ => ChatFinishReason.Stop,
    };

    /// <summary>Builds token <see cref="UsageDetails"/> from prompt/completion counts.</summary>
    public static UsageDetails ToUsageDetails(int promptTokens, int completionTokens) => new()
    {
        InputTokenCount = promptTokens,
        OutputTokenCount = completionTokens,
        TotalTokenCount = promptTokens + completionTokens,
    };

    // --- Helpers ------------------------------------------------------------

    // AOT-safe JSON (no reflection-based JsonSerializer) — tool-call arguments are
    // simple JSON objects whose values are typically JsonElement.
    private static string SerializeArguments(IDictionary<string, object?>? arguments)
    {
        if (arguments is null || arguments.Count == 0)
            return "{}";

        var buffer = new ArrayBufferWriter<byte>();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            foreach (var (key, value) in arguments)
            {
                writer.WritePropertyName(key);
                WriteValue(writer, value);
            }
            writer.WriteEndObject();
        }
        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    private static void WriteValue(Utf8JsonWriter writer, object? value)
    {
        switch (value)
        {
            case null: writer.WriteNullValue(); break;
            case JsonElement je: je.WriteTo(writer); break;
            case string s: writer.WriteStringValue(s); break;
            case bool b: writer.WriteBooleanValue(b); break;
            case int i: writer.WriteNumberValue(i); break;
            case long l: writer.WriteNumberValue(l); break;
            case double d: writer.WriteNumberValue(d); break;
            case float f: writer.WriteNumberValue(f); break;
            default: writer.WriteStringValue(value.ToString()); break;
        }
    }

    private static IDictionary<string, object?>? ParseArguments(string? arguments)
    {
        if (string.IsNullOrWhiteSpace(arguments))
            return null;
        try
        {
            using var doc = JsonDocument.Parse(arguments);
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
                return null;

            var dict = new Dictionary<string, object?>();
            foreach (var prop in doc.RootElement.EnumerateObject())
                dict[prop.Name] = prop.Value.Clone();
            return dict;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static string ResultToString(object? result) => result switch
    {
        null => "",
        string s => s,
        JsonElement je => je.ValueKind == JsonValueKind.String ? je.GetString() ?? "" : je.GetRawText(),
        _ => result.ToString() ?? "",
    };

    private static float? ReadFloat(AdditionalPropertiesDictionary? props, string key)
    {
        if (props is null || !props.TryGetValue(key, out var value) || value is null)
            return null;

        return value switch
        {
            float f => f,
            double d => (float)d,
            int i => i,
            long l => l,
            JsonElement je when je.ValueKind == JsonValueKind.Number => je.GetSingle(),
            // Invariant culture: these values are JSON/config-adjacent, so "0.9" must parse
            // identically on a machine whose current culture uses ',' as the decimal separator.
            string s when float.TryParse(
                s, NumberStyles.Float, CultureInfo.InvariantCulture, out var r) => r,
            _ => null,
        };
    }
}
