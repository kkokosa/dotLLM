using System.Text.Json;

namespace DotLLM.Tokenizers.ChatTemplates;

/// <summary>
/// IChatTemplate implementation backed by a Jinja2-subset interpreter.
/// Parses the template once at construction and evaluates per Apply() call.
/// </summary>
public sealed class JinjaChatTemplate : IChatTemplate
{
    private readonly JinjaTemplate _ast;
    private readonly string _bosToken;
    private readonly string _eosToken;

    /// <summary>
    /// Creates a new Jinja2 chat template.
    /// </summary>
    /// <param name="templateSource">Jinja2 template string (from GGUF metadata or HuggingFace config).</param>
    /// <param name="bosToken">Beginning-of-sequence token string.</param>
    /// <param name="eosToken">End-of-sequence token string.</param>
    public JinjaChatTemplate(string templateSource, string bosToken, string eosToken)
    {
        _bosToken = bosToken;
        _eosToken = eosToken;

        var lexer = new JinjaLexer(templateSource);
        var tokens = lexer.Tokenize();
        var parser = new JinjaParser(tokens);
        _ast = parser.Parse();
    }

    /// <inheritdoc/>
    public string Apply(IReadOnlyList<ChatMessage> messages, ChatTemplateOptions options)
    {
        var context = BuildContext(messages, options);
        var evaluator = new JinjaEvaluator(context);
        return evaluator.Evaluate(_ast);
    }

    private Dictionary<string, object?> BuildContext(IReadOnlyList<ChatMessage> messages, ChatTemplateOptions options)
    {
        // Convert ChatMessage[] to List<Dict> matching HuggingFace Jinja template convention
        var messageList = new List<object?>();
        foreach (var msg in messages)
        {
            var dict = new Dictionary<string, object?>
            {
                ["role"] = msg.Role,
                ["content"] = msg.Content,
            };

            if (msg.ToolCalls is { Length: > 0 })
            {
                var toolCalls = new List<object?>();
                foreach (var tc in msg.ToolCalls)
                {
                    var tcDict = new Dictionary<string, object?>
                    {
                        ["id"] = tc.Id,
                        ["type"] = "function",
                        ["function"] = new Dictionary<string, object?>
                        {
                            ["name"] = tc.FunctionName,
                            ["arguments"] = tc.Arguments,
                        }
                    };
                    toolCalls.Add(tcDict);
                }
                dict["tool_calls"] = toolCalls;
            }

            if (msg.ToolCallId is not null)
                dict["tool_call_id"] = msg.ToolCallId;

            messageList.Add(dict);
        }

        var context = new Dictionary<string, object?>
        {
            ["messages"] = messageList,
            ["add_generation_prompt"] = options.AddGenerationPrompt,
            ["bos_token"] = _bosToken,
            ["eos_token"] = _eosToken,
        };

        // Add tool definitions if present
        if (options.Tools is { Length: > 0 })
        {
            var tools = new List<object?>();
            foreach (var tool in options.Tools)
            {
                var toolDict = new Dictionary<string, object?>
                {
                    ["type"] = "function",
                    ["function"] = new Dictionary<string, object?>
                    {
                        ["name"] = tool.Name,
                        ["description"] = tool.Description,
                        ["parameters"] = ParseJsonToDict(tool.ParametersSchema),
                    }
                };
                tools.Add(toolDict);
            }
            context["tools"] = tools;
        }

        return context;
    }

    /// <summary>
    /// Parses a JSON string into nested Dictionary/List structures
    /// that the Jinja evaluator can traverse.
    /// </summary>
    private static object? ParseJsonToDict(string json)
    {
        if (string.IsNullOrEmpty(json))
            return null;

        try
        {
            using var doc = JsonDocument.Parse(json);
            return ConvertJsonElement(doc.RootElement);
        }
        catch
        {
            return json; // fallback to raw string
        }
    }

    private static object? ConvertJsonElement(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.Object => ConvertJsonObject(element),
        JsonValueKind.Array => ConvertJsonArray(element),
        JsonValueKind.String => element.GetString(),
        JsonValueKind.Number => element.TryGetInt32(out int i) ? i : element.GetDouble(),
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        JsonValueKind.Null => null,
        _ => element.ToString()
    };

    private static Dictionary<string, object?> ConvertJsonObject(JsonElement element)
    {
        var dict = new Dictionary<string, object?>();
        foreach (var prop in element.EnumerateObject())
            dict[prop.Name] = ConvertJsonElement(prop.Value);
        return dict;
    }

    private static List<object?> ConvertJsonArray(JsonElement element)
    {
        var list = new List<object?>();
        foreach (var item in element.EnumerateArray())
            list.Add(ConvertJsonElement(item));
        return list;
    }
}
