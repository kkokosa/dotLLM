// dotLLM Tool Calling Sample
//
// Demonstrates the tool calling pipeline:
// 1. Define tools with JSON Schema parameters
// 2. Apply chat template with tool definitions
// 3. Generate model response
// 4. Detect and parse tool calls
// 5. Feed tool results back into the conversation
// 6. Generate final response
//
// Usage: dotnet run -- <model.gguf>
//
// This sample requires a model with tool calling support (e.g., Llama 3.1 Instruct, Qwen2.5).

using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.Constraints;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;

// --- 1. Define tools ---

var tools = new ToolDefinition[]
{
    new("get_weather", "Get current weather for a location", """
        {
            "type": "object",
            "properties": {
                "location": { "type": "string", "description": "City name" },
                "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
            },
            "required": ["location"]
        }
        """),
    new("get_time", "Get current time in a timezone", """
        {
            "type": "object",
            "properties": {
                "timezone": { "type": "string", "description": "IANA timezone (e.g., Europe/London)" }
            },
            "required": ["timezone"]
        }
        """),
};

// --- Demo without a model (shows the pipeline structure) ---

Console.WriteLine("=== dotLLM Tool Calling Pipeline Demo ===");
Console.WriteLine();

// Show tool definitions
Console.WriteLine("Available tools:");
foreach (var tool in tools)
    Console.WriteLine($"  - {tool.Name}: {tool.Description}");
Console.WriteLine();

// Show schema constraint generation
Console.WriteLine("--- Schema Constraint (for tool_choice=required) ---");
string schema = ToolCallSchemaBuilder.BuildForRequired(tools);
Console.WriteLine(schema);
Console.WriteLine();

// Show parallel call schema
Console.WriteLine("--- Parallel Calls Schema ---");
string parallelSchema = ToolCallSchemaBuilder.BuildForParallelCalls(tools);
Console.WriteLine(parallelSchema);
Console.WriteLine();

// --- Demo the parsing pipeline ---

Console.WriteLine("--- Tool Call Detection Examples ---");
Console.WriteLine();

// Llama format
var llamaParser = new DotLLM.Tokenizers.ToolCallParsers.LlamaToolCallParser();
string llamaOutput = """<|python_tag|>{"name": "get_weather", "parameters": {"location": "Paris", "unit": "celsius"}}""";
var llamaCalls = llamaParser.TryParse(llamaOutput);
Console.WriteLine($"Llama format: {llamaCalls?.Length ?? 0} call(s) detected");
if (llamaCalls is not null)
    foreach (var tc in llamaCalls)
        Console.WriteLine($"  [{tc.Id}] {tc.FunctionName}({tc.Arguments})");
Console.WriteLine();

// Hermes format (parallel calls)
var hermesParser = new DotLLM.Tokenizers.ToolCallParsers.HermesToolCallParser();
string hermesOutput = """
    <tool_call>{"name": "get_weather", "arguments": {"location": "London"}}</tool_call>
    <tool_call>{"name": "get_time", "arguments": {"timezone": "Europe/London"}}</tool_call>
    """;
var hermesCalls = hermesParser.TryParse(hermesOutput);
Console.WriteLine($"Hermes format (parallel): {hermesCalls?.Length ?? 0} call(s) detected");
if (hermesCalls is not null)
    foreach (var tc in hermesCalls)
        Console.WriteLine($"  [{tc.Id}] {tc.FunctionName}({tc.Arguments})");
Console.WriteLine();

// Mistral format
var mistralParser = new DotLLM.Tokenizers.ToolCallParsers.MistralToolCallParser();
string mistralOutput = """[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "Tokyo"}}]""";
var mistralCalls = mistralParser.TryParse(mistralOutput);
Console.WriteLine($"Mistral format: {mistralCalls?.Length ?? 0} call(s) detected");
if (mistralCalls is not null)
    foreach (var tc in mistralCalls)
        Console.WriteLine($"  [{tc.Id}] {tc.FunctionName}({tc.Arguments})");
Console.WriteLine();

// --- Demo ToolCallDetector ---

Console.WriteLine("--- ToolCallDetector ---");
var response = new InferenceResponse
{
    GeneratedTokenIds = [],
    Text = """<tool_call>{"name": "get_weather", "arguments": {"location": "Berlin"}}</tool_call>""",
    FinishReason = FinishReason.Stop,
    PromptTokenCount = 100,
    GeneratedTokenCount = 20
};

var enriched = ToolCallDetector.DetectToolCalls(response, hermesParser);
Console.WriteLine($"Before: FinishReason={response.FinishReason}, ToolCalls={response.ToolCalls?.Length ?? 0}");
Console.WriteLine($"After:  FinishReason={enriched.FinishReason}, ToolCalls={enriched.ToolCalls?.Length ?? 0}");
Console.WriteLine();

// --- Demo multi-turn conversation structure ---

Console.WriteLine("--- Multi-Turn Tool Use Conversation ---");

var messages = new List<ChatMessage>
{
    new() { Role = "system", Content = "You are a helpful assistant with access to tools." },
    new() { Role = "user", Content = "What's the weather in London?" },
    // Simulated assistant response with tool call
    new()
    {
        Role = "assistant", Content = "",
        ToolCalls = [new ToolCall("call_0", "get_weather", """{"location":"London"}""")]
    },
    // Simulated tool result
    new() { Role = "tool", Content = """{"temperature": 15, "condition": "cloudy"}""", ToolCallId = "call_0" },
    // Simulated final response
    new() { Role = "assistant", Content = "The weather in London is currently 15 degrees and cloudy." },
};

Console.WriteLine("Conversation history:");
foreach (var msg in messages)
{
    string display = msg.Role switch
    {
        "tool" => $"  [tool:{msg.ToolCallId}] {msg.Content}",
        "assistant" when msg.ToolCalls is { Length: > 0 } =>
            $"  [assistant] Tool calls: {string.Join(", ", msg.ToolCalls.Select(tc => $"{tc.FunctionName}({tc.Arguments})"))}",
        _ => $"  [{msg.Role}] {msg.Content}"
    };
    Console.WriteLine(display);
}
Console.WriteLine();

// Demo template rendering (with a ChatML-style template)
const string chatMlTemplate =
    "{% for message in messages %}" +
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}" +
    "{% endfor %}" +
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

var template = new JinjaChatTemplate(chatMlTemplate, "<s>", "</s>");
string rendered = template.Apply(messages[..2], new ChatTemplateOptions
{
    AddGenerationPrompt = true,
    Tools = tools
});
Console.WriteLine("Rendered prompt (first 2 messages + tools in context):");
Console.WriteLine(rendered);

Console.WriteLine();
Console.WriteLine("=== Tool calling pipeline complete ===");
Console.WriteLine();
Console.WriteLine("To use with a real model:");
Console.WriteLine("  dotllm chat <model.gguf> --tools @tools.json");
Console.WriteLine("  dotllm chat <model.gguf> --tools @tools.json --tool-choice required");
