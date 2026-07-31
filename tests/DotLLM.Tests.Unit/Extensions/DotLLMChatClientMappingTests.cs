using System.Globalization;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Extensions.AI;
using Microsoft.Extensions.AI;
using Xunit;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;
using EngineToolCall = DotLLM.Tokenizers.ToolCall;

namespace DotLLM.Tests.Unit.Extensions;

/// <summary>
/// Unit tests for the Microsoft.Extensions.AI ↔ dotLLM translation layer
/// (<see cref="ChatClientMapping"/>). Pure mapping — no model load required.
/// </summary>
public sealed class DotLLMChatClientMappingTests
{
    // --- Message flattening -------------------------------------------------

    [Fact]
    public void ToEngineMessages_TextRoles_MapOneToOne()
    {
        ChatMessage[] messages =
        [
            new(ChatRole.System, "Be terse."),
            new(ChatRole.User, "Hello"),
        ];

        var engine = ChatClientMapping.ToEngineMessages(messages);

        Assert.Equal(2, engine.Length);
        Assert.Equal("system", engine[0].Role);
        Assert.Equal("Be terse.", engine[0].Content);
        Assert.Equal("user", engine[1].Role);
        Assert.Equal("Hello", engine[1].Content);
    }

    [Fact]
    public void ToEngineMessages_AssistantFunctionCall_BecomesToolCall()
    {
        var call = new FunctionCallContent("call_1", "get_weather",
            new Dictionary<string, object?> { ["city"] = "Paris" });
        ChatMessage[] messages = [new(ChatRole.Assistant, [call])];

        var engine = ChatClientMapping.ToEngineMessages(messages);

        var msg = Assert.Single(engine);
        Assert.Equal("assistant", msg.Role);
        Assert.NotNull(msg.ToolCalls);
        Assert.Equal("call_1", msg.ToolCalls![0].Id);
        Assert.Equal("get_weather", msg.ToolCalls[0].FunctionName);
        Assert.Contains("Paris", msg.ToolCalls[0].Arguments);
    }

    [Fact]
    public void ToEngineMessages_FunctionResult_BecomesToolMessage()
    {
        ChatMessage[] messages =
            [new(ChatRole.Tool, [new FunctionResultContent("call_1", "22C and sunny")])];

        var engine = ChatClientMapping.ToEngineMessages(messages);

        var msg = Assert.Single(engine);
        Assert.Equal("tool", msg.Role);
        Assert.Equal("call_1", msg.ToolCallId);
        Assert.Equal("22C and sunny", msg.Content);
    }

    [Theory]
    [InlineData("system")]
    [InlineData("user")]
    [InlineData("assistant")]
    [InlineData("tool")]
    public void ToRole_RoundTrips(string role)
    {
        Assert.Equal(role, ChatClientMapping.ToRole(new ChatRole(role)));
    }

    [Fact]
    public void ToRole_UnknownRole_IsPreservedNotCoercedToUser()
    {
        Assert.Equal("developer", ChatClientMapping.ToRole(new ChatRole("developer")));
    }

    // --- Options ------------------------------------------------------------

    [Fact]
    public void ToInferenceOptions_Null_ReturnsDefaults()
    {
        var io = ChatClientMapping.ToInferenceOptions(null);
        Assert.Equal(new InferenceOptions().Temperature, io.Temperature);
    }

    [Fact]
    public void ToInferenceOptions_MapsSamplingFields()
    {
        var options = new ChatOptions
        {
            Temperature = 0.3f,
            TopP = 0.8f,
            TopK = 40,
            MaxOutputTokens = 128,
            Seed = 1234,
            StopSequences = ["\n\n"],
        };

        var io = ChatClientMapping.ToInferenceOptions(options);

        Assert.Equal(0.3f, io.Temperature);
        Assert.Equal(0.8f, io.TopP);
        Assert.Equal(40, io.TopK);
        Assert.Equal(128, io.MaxTokens);
        Assert.Equal(1234, io.Seed);
        Assert.Contains("\n\n", io.StopSequences);
    }

    [Fact]
    public void ToInferenceOptions_AdditionalProperties_MapDotLLMKnobs()
    {
        var options = new ChatOptions
        {
            AdditionalProperties = new AdditionalPropertiesDictionary
            {
                ["min_p"] = 0.05f,
                ["repetition_penalty"] = 1.1f,
            },
        };

        var io = ChatClientMapping.ToInferenceOptions(options);

        Assert.Equal(0.05f, io.MinP, 3);
        Assert.Equal(1.1f, io.RepetitionPenalty, 3);
    }

    [Fact]
    public void ToInferenceOptions_StringKnobs_ParseUnderCommaDecimalCulture()
    {
        var original = CultureInfo.CurrentCulture;
        CultureInfo.CurrentCulture = new CultureInfo("pl-PL"); // ',' decimal separator
        try
        {
            var options = new ChatOptions
            {
                AdditionalProperties = new AdditionalPropertiesDictionary
                {
                    ["min_p"] = "0.05",
                    ["repetition_penalty"] = "1.1",
                },
            };

            var io = ChatClientMapping.ToInferenceOptions(options);

            Assert.Equal(0.05f, io.MinP, 3);
            Assert.Equal(1.1f, io.RepetitionPenalty, 3);
        }
        finally
        {
            CultureInfo.CurrentCulture = original;
        }
    }

    [Fact]
    public void ToResponseFormat_Json_MapsToJsonObject()
    {
        Assert.IsType<ResponseFormat.JsonObject>(
            ChatClientMapping.ToResponseFormat(ChatResponseFormat.Json));
    }

    [Fact]
    public void ToResponseFormat_JsonSchema_MapsToJsonSchema()
    {
        using var schema = JsonDocument.Parse("""{"type":"object","properties":{"x":{"type":"number"}}}""");
        var format = ChatResponseFormat.ForJsonSchema(schema.RootElement.Clone(), schemaName: "MySchema");

        var result = ChatClientMapping.ToResponseFormat(format);

        var js = Assert.IsType<ResponseFormat.JsonSchema>(result);
        Assert.Contains("properties", js.Schema);
        Assert.Equal("MySchema", js.Name);
    }

    [Fact]
    public void ToResponseFormat_Text_ReturnsNull()
    {
        Assert.Null(ChatClientMapping.ToResponseFormat(ChatResponseFormat.Text));
        Assert.Null(ChatClientMapping.ToResponseFormat(null));
    }

    [Fact]
    public void ToToolDefinitions_MapsAIFunctions()
    {
        var fn = AIFunctionFactory.Create(
            (string city) => $"weather in {city}", "get_weather", "Get the weather");
        var options = new ChatOptions { Tools = [fn] };

        var tools = ChatClientMapping.ToToolDefinitions(options);

        Assert.NotNull(tools);
        var tool = Assert.Single(tools!);
        Assert.Equal("get_weather", tool.Name);
        Assert.Equal("Get the weather", tool.Description);
        Assert.Contains("city", tool.ParametersSchema);
    }

    [Fact]
    public void ToToolDefinitions_NoTools_ReturnsNull()
    {
        Assert.Null(ChatClientMapping.ToToolDefinitions(null));
        Assert.Null(ChatClientMapping.ToToolDefinitions(new ChatOptions()));
    }

    // --- Output mapping -----------------------------------------------------

    [Fact]
    public void ToResponseContents_TextOnly_EmitsTextContent()
    {
        var contents = ChatClientMapping.ToResponseContents("Hi there", null);
        var text = Assert.IsType<TextContent>(Assert.Single(contents));
        Assert.Equal("Hi there", text.Text);
    }

    [Fact]
    public void ToResponseContents_ToolCalls_EmitsFunctionCallAndDropsText()
    {
        EngineToolCall[] calls = [new("call_9", "get_weather", """{"city":"Paris"}""")];

        var contents = ChatClientMapping.ToResponseContents("raw tool text", calls);

        var fc = Assert.IsType<FunctionCallContent>(Assert.Single(contents));
        Assert.Equal("call_9", fc.CallId);
        Assert.Equal("get_weather", fc.Name);
        Assert.NotNull(fc.Arguments);
        var city = Assert.IsType<JsonElement>(fc.Arguments!["city"]);
        Assert.Equal("Paris", city.GetString());
    }

    [Fact]
    public void ToResponseContents_ToolCallMissingId_GeneratesId()
    {
        EngineToolCall[] calls = [new("", "fn", "{}")];
        var contents = ChatClientMapping.ToResponseContents("", calls);
        var fc = Assert.IsType<FunctionCallContent>(Assert.Single(contents));
        Assert.False(string.IsNullOrEmpty(fc.CallId));
    }

    [Theory]
    [InlineData(FinishReason.Stop, "stop")]
    [InlineData(FinishReason.Length, "length")]
    [InlineData(FinishReason.ToolCalls, "tool_calls")]
    public void ToChatFinishReason_Maps(FinishReason reason, string expected)
    {
        Assert.Equal(expected, ChatClientMapping.ToChatFinishReason(reason).Value);
    }

    [Fact]
    public void ToUsageDetails_SumsTokens()
    {
        var usage = ChatClientMapping.ToUsageDetails(promptTokens: 10, completionTokens: 5);
        Assert.Equal(10, usage.InputTokenCount);
        Assert.Equal(5, usage.OutputTokenCount);
        Assert.Equal(15, usage.TotalTokenCount);
    }
}
