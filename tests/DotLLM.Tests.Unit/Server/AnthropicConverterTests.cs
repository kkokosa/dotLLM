using System.Text.Json;
using DotLLM.Engine;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Unit tests for the Anthropic Messages API translation layer
/// (<see cref="AnthropicConverter"/> and <see cref="MessagesEndpoint"/> validation).
/// These exercise pure request/response reshaping — no model load required.
/// </summary>
public sealed class AnthropicConverterTests
{
    private static AnthropicMessagesRequest Parse(string json) =>
        JsonSerializer.Deserialize(json, ServerJsonContext.Default.AnthropicMessagesRequest)!;

    // --- Message flattening -------------------------------------------------

    [Fact]
    public void ToMessages_StringContent_MapsOneToOne()
    {
        var req = Parse("""
        {"model":"m","max_tokens":16,"messages":[{"role":"user","content":"Hello"}]}
        """);

        var messages = AnthropicConverter.ToMessages(req);

        Assert.Single(messages);
        Assert.Equal("user", messages[0].Role);
        Assert.Equal("Hello", messages[0].Content);
    }

    [Fact]
    public void ToMessages_SystemString_BecomesLeadingSystemMessage()
    {
        var req = Parse("""
        {"model":"m","max_tokens":16,"system":"Be terse.","messages":[{"role":"user","content":"hi"}]}
        """);

        var messages = AnthropicConverter.ToMessages(req);

        Assert.Equal(2, messages.Length);
        Assert.Equal("system", messages[0].Role);
        Assert.Equal("Be terse.", messages[0].Content);
        Assert.Equal("user", messages[1].Role);
    }

    [Fact]
    public void ToMessages_SystemBlockArray_IsConcatenated()
    {
        var req = Parse("""
        {"model":"m","max_tokens":16,
         "system":[{"type":"text","text":"Line 1"},{"type":"text","text":"Line 2"}],
         "messages":[{"role":"user","content":"hi"}]}
        """);

        var messages = AnthropicConverter.ToMessages(req);

        Assert.Equal("system", messages[0].Role);
        Assert.Equal("Line 1\nLine 2", messages[0].Content);
    }

    [Fact]
    public void ToMessages_TextBlockArray_IsConcatenated()
    {
        var req = Parse("""
        {"model":"m","max_tokens":16,"messages":[
          {"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}
        """);

        var messages = AnthropicConverter.ToMessages(req);

        Assert.Single(messages);
        Assert.Equal("a\nb", messages[0].Content);
    }

    [Fact]
    public void ToMessages_AssistantToolUseBlock_BecomesToolCall()
    {
        var req = Parse("""
        {"model":"m","max_tokens":16,"messages":[
          {"role":"assistant","content":[
            {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}]}]}
        """);

        var messages = AnthropicConverter.ToMessages(req);

        Assert.Single(messages);
        Assert.Equal("assistant", messages[0].Role);
        Assert.NotNull(messages[0].ToolCalls);
        var call = messages[0].ToolCalls![0];
        Assert.Equal("toolu_1", call.Id);
        Assert.Equal("get_weather", call.FunctionName);
        Assert.Contains("Paris", call.Arguments);
    }

    [Fact]
    public void ToMessages_ToolResultBlock_BecomesToolRoleMessage()
    {
        var req = Parse("""
        {"model":"m","max_tokens":16,"messages":[
          {"role":"user","content":[
            {"type":"tool_result","tool_use_id":"toolu_1","content":"22C and sunny"}]}]}
        """);

        var messages = AnthropicConverter.ToMessages(req);

        Assert.Single(messages);
        Assert.Equal("tool", messages[0].Role);
        Assert.Equal("toolu_1", messages[0].ToolCallId);
        Assert.Equal("22C and sunny", messages[0].Content);
    }

    [Fact]
    public void ToMessages_ToolResultBlockArrayContent_ConcatenatesText()
    {
        var req = Parse("""
        {"model":"m","max_tokens":16,"messages":[
          {"role":"user","content":[
            {"type":"tool_result","tool_use_id":"toolu_2","content":[{"type":"text","text":"ok"}]}]}]}
        """);

        var messages = AnthropicConverter.ToMessages(req);

        Assert.Equal("tool", messages[0].Role);
        Assert.Equal("ok", messages[0].Content);
    }

    // --- Tool choice --------------------------------------------------------

    [Theory]
    [InlineData("""{"type":"auto"}""", typeof(DotLLM.Core.Configuration.ToolChoice.Auto))]
    [InlineData("""{"type":"any"}""", typeof(DotLLM.Core.Configuration.ToolChoice.Required))]
    [InlineData("""{"type":"none"}""", typeof(DotLLM.Core.Configuration.ToolChoice.None))]
    public void ParseToolChoice_MapsTypes(string json, Type expected)
    {
        using var doc = JsonDocument.Parse(json);
        var choice = AnthropicConverter.ParseToolChoice(doc.RootElement);
        Assert.IsType(expected, choice);
    }

    [Fact]
    public void ParseToolChoice_Tool_MapsToFunction()
    {
        using var doc = JsonDocument.Parse("""{"type":"tool","name":"get_weather"}""");
        var choice = AnthropicConverter.ParseToolChoice(doc.RootElement);
        var fn = Assert.IsType<DotLLM.Core.Configuration.ToolChoice.Function>(choice);
        Assert.Equal("get_weather", fn.Name);
    }

    [Fact]
    public void ParseToolChoice_Null_DefaultsToAuto()
    {
        Assert.IsType<DotLLM.Core.Configuration.ToolChoice.Auto>(AnthropicConverter.ParseToolChoice(null));
    }

    // --- Stop reason mapping ------------------------------------------------

    [Theory]
    [InlineData(FinishReason.Stop, false, "end_turn")]
    [InlineData(FinishReason.Stop, true, "stop_sequence")]
    [InlineData(FinishReason.Length, false, "max_tokens")]
    [InlineData(FinishReason.ToolCalls, false, "tool_use")]
    public void ToStopReason_MapsFinishReason(FinishReason reason, bool matched, string expected)
    {
        Assert.Equal(expected, AnthropicConverter.ToStopReason(reason, matched));
    }

    // --- Tool use blocks ----------------------------------------------------

    [Fact]
    public void ToToolUseBlocks_ParsesArgumentsIntoJsonObject()
    {
        var blocks = AnthropicConverter.ToToolUseBlocks(
            [new ToolCall("toolu_x", "get_weather", """{"city":"Paris"}""")]);

        var block = Assert.Single(blocks);
        Assert.Equal("tool_use", block.Type);
        Assert.Equal("toolu_x", block.Id);
        Assert.Equal("get_weather", block.Name);
        Assert.NotNull(block.Input);
        Assert.Equal("Paris", block.Input!.Value.GetProperty("city").GetString());
    }

    [Fact]
    public void ToToolUseBlocks_MissingId_GeneratesAnthropicId()
    {
        var blocks = AnthropicConverter.ToToolUseBlocks([new ToolCall("", "f", "{}")]);
        Assert.StartsWith("toolu_", blocks[0].Id);
    }

    [Fact]
    public void ParseInput_InvalidJson_ReturnsEmptyObject()
    {
        var el = AnthropicConverter.ParseInput("not json");
        Assert.Equal(JsonValueKind.Object, el.ValueKind);
        Assert.False(el.EnumerateObject().MoveNext());
    }

    [Theory]
    [InlineData("\"hi\"")]
    [InlineData("[]")]
    [InlineData("[1,2]")]
    [InlineData("42")]
    [InlineData("null")]
    [InlineData("true")]
    public void ParseInput_NonObjectRoot_ReturnsEmptyObject(string arguments)
    {
        // Anthropic requires tool_use.input to be an object — a model that emits a bare
        // scalar or array must not produce an invalid wire shape.
        var el = AnthropicConverter.ParseInput(arguments);
        Assert.Equal(JsonValueKind.Object, el.ValueKind);
        Assert.False(el.EnumerateObject().MoveNext());
    }

    [Fact]
    public void ParseInput_ObjectRoot_PassesThrough()
    {
        var el = AnthropicConverter.ParseInput("""{"city":"Paris"}""");
        Assert.Equal(JsonValueKind.Object, el.ValueKind);
        Assert.Equal("Paris", el.GetProperty("city").GetString());
    }

    // --- Request validation -------------------------------------------------

    [Fact]
    public void ValidateRequest_EmptyMessages_Fails()
    {
        var req = new AnthropicMessagesRequest { Messages = [], MaxTokens = 16 };
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req, requireMaxTokens: true));
    }

    [Fact]
    public void ValidateRequest_MissingMaxTokens_FailsWhenRequired()
    {
        var req = Parse("""{"model":"m","messages":[{"role":"user","content":"hi"}]}""");
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req, requireMaxTokens: true));
        // count_tokens does not require max_tokens.
        Assert.Null(MessagesEndpoint.ValidateRequest(req, requireMaxTokens: false));
    }

    [Fact]
    public void ValidateRequest_NonPositiveMaxTokens_Fails()
    {
        var req = new AnthropicMessagesRequest
        {
            Messages = [new AnthropicMessageDto { Role = "user", Content = default }],
            MaxTokens = 0,
        };
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req, requireMaxTokens: true));
    }

    [Theory]
    [InlineData("system")]
    [InlineData("tool")]
    [InlineData("developer")]
    [InlineData("")]
    public void ValidateRequest_UnsupportedRole_Fails(string role)
    {
        // Roles flow straight into the chat template; only user/assistant are addressable
        // by a client (the system prompt is the top-level `system` field).
        var req = Parse($$"""
        {"model":"m","max_tokens":16,"messages":[{"role":"{{role}}","content":"hi"}]}
        """);
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req, requireMaxTokens: true));
    }

    [Theory]
    [InlineData("123")]
    [InlineData("null")]
    [InlineData("true")]
    [InlineData("""{"type":"text"}""")]
    public void ValidateRequest_NonStringNonArrayContent_Fails(string content)
    {
        var req = Parse($$"""
        {"model":"m","max_tokens":16,"messages":[{"role":"user","content":{{content}}}]}
        """);
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req, requireMaxTokens: true));
    }

    [Fact]
    public void ValidateRequest_MissingContent_Fails()
    {
        var req = Parse("""{"model":"m","max_tokens":16,"messages":[{"role":"user"}]}""");
        Assert.NotNull(MessagesEndpoint.ValidateRequest(req, requireMaxTokens: true));
    }

    [Fact]
    public void ValidateRequest_BlockArrayContent_ReturnsNull()
    {
        var req = Parse("""
        {"model":"m","max_tokens":16,"messages":[
          {"role":"user","content":[{"type":"text","text":"hi"}]},
          {"role":"assistant","content":"yes"}]}
        """);
        Assert.Null(MessagesEndpoint.ValidateRequest(req, requireMaxTokens: true));
    }

    [Fact]
    public void ValidateRequest_Valid_ReturnsNull()
    {
        var req = Parse("""
        {"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}
        """);
        Assert.Null(MessagesEndpoint.ValidateRequest(req, requireMaxTokens: true));
    }

    // --- Response serialization shape --------------------------------------

    [Fact]
    public void MessageResponse_SerializesAnthropicShape()
    {
        var response = new AnthropicMessageResponse
        {
            Id = "msg_1",
            Model = "m",
            Content = [new AnthropicContentBlockDto { Type = "text", Text = "Hi" }],
            StopReason = "end_turn",
            StopSequence = null,
            Usage = new AnthropicUsageDto { InputTokens = 5, OutputTokens = 2 },
        };

        string json = JsonSerializer.Serialize(response, ServerJsonContext.Default.AnthropicMessageResponse);

        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        Assert.Equal("message", root.GetProperty("type").GetString());
        Assert.Equal("assistant", root.GetProperty("role").GetString());
        Assert.Equal("end_turn", root.GetProperty("stop_reason").GetString());
        // stop_sequence must be present even when null (Anthropic wire format).
        Assert.Equal(JsonValueKind.Null, root.GetProperty("stop_sequence").ValueKind);
        Assert.Equal(5, root.GetProperty("usage").GetProperty("input_tokens").GetInt32());
        Assert.Equal("text", root.GetProperty("content")[0].GetProperty("type").GetString());
        Assert.Equal("Hi", root.GetProperty("content")[0].GetProperty("text").GetString());
    }

    [Fact]
    public void CountTokensResponse_SerializesInputTokens()
    {
        string json = JsonSerializer.Serialize(
            new AnthropicCountTokensResponse { InputTokens = 42 },
            ServerJsonContext.Default.AnthropicCountTokensResponse);
        using var doc = JsonDocument.Parse(json);
        Assert.Equal(42, doc.RootElement.GetProperty("input_tokens").GetInt32());
    }

    [Fact]
    public void ErrorResponse_SerializesAnthropicEnvelope()
    {
        string json = JsonSerializer.Serialize(
            new AnthropicErrorResponse
            {
                Error = new AnthropicErrorBody { Type = "invalid_request_error", Message = "bad" },
            },
            ServerJsonContext.Default.AnthropicErrorResponse);
        using var doc = JsonDocument.Parse(json);
        Assert.Equal("error", doc.RootElement.GetProperty("type").GetString());
        Assert.Equal("invalid_request_error", doc.RootElement.GetProperty("error").GetProperty("type").GetString());
    }
}
