using System.Text;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;

namespace DotLLM.Server;

/// <summary>
/// Converts between Anthropic Messages API DTOs and dotLLM engine types.
/// The engine, chat template, sampler and tool-calling pipeline are shared
/// verbatim with the OpenAI surface — this type only reshapes the wire format.
/// </summary>
public static class AnthropicConverter
{
    /// <summary>
    /// Flattens an Anthropic request (top-level <c>system</c> + per-message
    /// content blocks) into the engine's linear <see cref="ChatMessage"/> list.
    /// </summary>
    /// <remarks>
    /// <list type="bullet">
    /// <item>A top-level <c>system</c> string/array becomes a leading <c>system</c> message.</item>
    /// <item>String message content maps 1:1.</item>
    /// <item><c>text</c> blocks are concatenated; <c>tool_use</c> blocks become <see cref="ToolCall"/>s;
    /// <c>tool_result</c> blocks become separate <c>tool</c>-role messages keyed by <c>tool_use_id</c>.</item>
    /// </list>
    /// </remarks>
    public static ChatMessage[] ToMessages(AnthropicMessagesRequest request)
    {
        var result = new List<ChatMessage>(request.Messages.Length + 1);

        string? systemText = ExtractText(request.System);
        if (!string.IsNullOrEmpty(systemText))
            result.Add(new ChatMessage { Role = "system", Content = systemText });

        foreach (var msg in request.Messages)
            AppendMessage(result, msg);

        return result.ToArray();
    }

    private static void AppendMessage(List<ChatMessage> target, AnthropicMessageDto msg)
    {
        var content = msg.Content;

        if (content.ValueKind == JsonValueKind.String)
        {
            target.Add(new ChatMessage { Role = msg.Role, Content = content.GetString() ?? "" });
            return;
        }

        if (content.ValueKind != JsonValueKind.Array)
        {
            target.Add(new ChatMessage { Role = msg.Role, Content = "" });
            return;
        }

        var textBuilder = new StringBuilder();
        List<ToolCall>? toolCalls = null;

        foreach (var block in content.EnumerateArray())
        {
            if (block.ValueKind != JsonValueKind.Object)
                continue;

            string? blockType = block.TryGetProperty("type", out var t) ? t.GetString() : null;
            switch (blockType)
            {
                case "text":
                    if (block.TryGetProperty("text", out var txt) && txt.ValueKind == JsonValueKind.String)
                    {
                        if (textBuilder.Length > 0) textBuilder.Append('\n');
                        textBuilder.Append(txt.GetString());
                    }
                    break;

                case "tool_use":
                    string tuId = block.TryGetProperty("id", out var idp) ? idp.GetString() ?? "" : "";
                    string tuName = block.TryGetProperty("name", out var np) ? np.GetString() ?? "" : "";
                    string tuInput = block.TryGetProperty("input", out var ip) ? ip.GetRawText() : "{}";
                    (toolCalls ??= []).Add(new ToolCall(tuId, tuName, tuInput));
                    break;

                case "tool_result":
                    // Emit any pending text/tool_use for this message, then the tool result.
                    FlushPending(target, msg.Role, textBuilder, ref toolCalls);
                    string trId = block.TryGetProperty("tool_use_id", out var tup) ? tup.GetString() ?? "" : "";
                    target.Add(new ChatMessage
                    {
                        Role = "tool",
                        Content = ExtractToolResultContent(block),
                        ToolCallId = trId,
                    });
                    break;
            }
        }

        FlushPending(target, msg.Role, textBuilder, ref toolCalls);
    }

    private static void FlushPending(
        List<ChatMessage> target, string role, StringBuilder textBuilder, ref List<ToolCall>? toolCalls)
    {
        if (textBuilder.Length == 0 && toolCalls is null)
            return;

        target.Add(new ChatMessage
        {
            Role = role,
            Content = textBuilder.ToString(),
            ToolCalls = toolCalls?.ToArray(),
        });
        textBuilder.Clear();
        toolCalls = null;
    }

    private static string ExtractToolResultContent(JsonElement block)
    {
        if (!block.TryGetProperty("content", out var c))
            return "";
        if (c.ValueKind == JsonValueKind.String)
            return c.GetString() ?? "";
        if (c.ValueKind == JsonValueKind.Array)
            return ConcatTextBlocks(c);
        return "";
    }

    /// <summary>
    /// Extracts plain text from an Anthropic <c>system</c> value: a string, or an
    /// array of <c>{"type":"text","text":"..."}</c> blocks. Returns null when absent.
    /// </summary>
    public static string? ExtractText(JsonElement? element)
    {
        if (element is null)
            return null;
        var e = element.Value;
        if (e.ValueKind == JsonValueKind.String)
            return e.GetString();
        if (e.ValueKind == JsonValueKind.Array)
            return ConcatTextBlocks(e);
        return null;
    }

    private static string ConcatTextBlocks(JsonElement array)
    {
        var sb = new StringBuilder();
        foreach (var item in array.EnumerateArray())
        {
            if (item.ValueKind == JsonValueKind.Object &&
                item.TryGetProperty("text", out var t) && t.ValueKind == JsonValueKind.String)
            {
                if (sb.Length > 0) sb.Append('\n');
                sb.Append(t.GetString());
            }
        }
        return sb.ToString();
    }

    /// <summary>Converts Anthropic tool definitions to engine <see cref="ToolDefinition"/>s.</summary>
    public static ToolDefinition[]? ToTools(AnthropicToolDto[]? dtos)
    {
        if (dtos is null)
            return null;
        var result = new ToolDefinition[dtos.Length];
        for (int i = 0; i < dtos.Length; i++)
        {
            var d = dtos[i];
            result[i] = new ToolDefinition(d.Name, d.Description ?? "", d.InputSchema?.GetRawText() ?? "{}");
        }
        return result;
    }

    /// <summary>
    /// Parses Anthropic <c>tool_choice</c> into the engine <see cref="ToolChoice"/>:
    /// <c>auto</c>→Auto, <c>any</c>→Required, <c>none</c>→None, <c>tool</c>→Function.
    /// </summary>
    public static ToolChoice ParseToolChoice(JsonElement? element)
    {
        if (element is null || element.Value.ValueKind != JsonValueKind.Object)
            return new ToolChoice.Auto();

        var e = element.Value;
        if (!e.TryGetProperty("type", out var typeProp))
            return new ToolChoice.Auto();

        return typeProp.GetString() switch
        {
            "any" => new ToolChoice.Required(),
            "none" => new ToolChoice.None(),
            "tool" when e.TryGetProperty("name", out var n) && n.ValueKind == JsonValueKind.String =>
                new ToolChoice.Function(n.GetString()!),
            _ => new ToolChoice.Auto(),
        };
    }

    /// <summary>Builds <see cref="InferenceOptions"/> from an Anthropic request.</summary>
    public static InferenceOptions ToInferenceOptions(
        AnthropicMessagesRequest request, IReadOnlyList<string> commonStops,
        SamplingDefaults defaults, ThreadingConfig threading)
    {
        var allStops = new List<string>(commonStops);
        if (request.StopSequences is { Length: > 0 })
        {
            foreach (var s in request.StopSequences)
                if (!string.IsNullOrEmpty(s))
                    allStops.Add(s);
        }

        return new InferenceOptions
        {
            Temperature = request.Temperature ?? defaults.Temperature,
            TopK = request.TopK ?? defaults.TopK,
            TopP = request.TopP ?? defaults.TopP,
            MinP = defaults.MinP,
            RepetitionPenalty = defaults.RepetitionPenalty,
            MaxTokens = request.MaxTokens ?? defaults.MaxTokens,
            Seed = defaults.Seed,
            StopSequences = allStops,
            Threading = threading,
        };
    }

    /// <summary>
    /// Maps an engine <see cref="FinishReason"/> to an Anthropic <c>stop_reason</c>.
    /// </summary>
    /// <param name="reason">The engine finish reason.</param>
    /// <param name="matchedStopSequence">
    /// True when generation stopped because a caller-supplied stop sequence matched
    /// (reported as <c>stop_sequence</c> rather than <c>end_turn</c>).
    /// </param>
    public static string ToStopReason(FinishReason reason, bool matchedStopSequence) => reason switch
    {
        FinishReason.ToolCalls => "tool_use",
        FinishReason.Length => "max_tokens",
        FinishReason.Stop => matchedStopSequence ? "stop_sequence" : "end_turn",
        _ => "end_turn",
    };

    /// <summary>Converts detected engine tool calls to Anthropic <c>tool_use</c> content blocks.</summary>
    public static AnthropicContentBlockDto[] ToToolUseBlocks(ToolCall[] toolCalls)
    {
        var blocks = new AnthropicContentBlockDto[toolCalls.Length];
        for (int i = 0; i < toolCalls.Length; i++)
        {
            var tc = toolCalls[i];
            blocks[i] = new AnthropicContentBlockDto
            {
                Type = "tool_use",
                Id = string.IsNullOrEmpty(tc.Id) ? GenerateToolUseId() : tc.Id,
                Name = tc.FunctionName,
                Input = ParseInput(tc.Arguments),
            };
        }
        return blocks;
    }

    /// <summary>
    /// Parses a tool-call argument JSON string into a JSON object element.
    /// </summary>
    /// <remarks>
    /// Anthropic requires <c>tool_use.input</c> to be an object, so a non-object root
    /// (a bare string, array or number emitted by a model that ignored the schema)
    /// collapses to <c>{}</c> rather than being passed through as an invalid wire shape
    /// that clients would reject or mis-handle.
    /// </remarks>
    public static JsonElement ParseInput(string? arguments)
    {
        if (string.IsNullOrWhiteSpace(arguments))
            return EmptyObject;
        try
        {
            using var doc = JsonDocument.Parse(arguments);
            return doc.RootElement.ValueKind == JsonValueKind.Object
                ? doc.RootElement.Clone()
                : EmptyObject;
        }
        catch (JsonException)
        {
            return EmptyObject;
        }
    }

    // A cloned JsonElement is detached and immutable, so one instance can be shared
    // across every empty/invalid tool input instead of reparsing "{}" per tool call.
    private static readonly JsonElement EmptyObject = ParseEmptyObject();

    private static JsonElement ParseEmptyObject()
    {
        using var doc = JsonDocument.Parse("{}");
        return doc.RootElement.Clone();
    }

    /// <summary>Generates an Anthropic-style message id (<c>msg_...</c>).</summary>
    public static string GenerateMessageId() => $"msg_{Guid.NewGuid():N}";

    /// <summary>Generates an Anthropic-style tool-use block id (<c>toolu_...</c>).</summary>
    public static string GenerateToolUseId() => $"toolu_{Guid.NewGuid():N}";
}
