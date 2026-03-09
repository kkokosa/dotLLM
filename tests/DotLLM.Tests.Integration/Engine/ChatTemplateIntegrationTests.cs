using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Integration tests for the Jinja2 chat template engine with real instruct models.
/// Verifies that chat templates parse correctly, format prompts properly, and
/// produce coherent first-turn responses with greedy decoding.
/// </summary>
public class ChatTemplateIntegrationTests
{
    /// <summary>
    /// Loads a GGUF model and returns all components needed for chat inference.
    /// </summary>
    private static (LlamaModel Model, GgufFile Gguf, BpeTokenizer Tokenizer, IChatTemplate Template, List<string> StopSequences)
        LoadChatModel(string filePath)
    {
        var gguf = GgufFile.Open(filePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        var model = LlamaModel.LoadFromGguf(gguf, config);

        // Create chat template from GGUF metadata, fallback to ChatML
        string bosToken = tokenizer.DecodeToken(tokenizer.BosTokenId);
        string eosToken = tokenizer.DecodeToken(tokenizer.EosTokenId);
        var jinjaTemplate = GgufChatTemplateFactory.TryCreate(gguf.Metadata, tokenizer);
        IChatTemplate template = jinjaTemplate
            ?? new JinjaChatTemplate(DefaultChatMlTemplate, bosToken, eosToken);

        // Common end-of-turn stop sequences
        var stopSequences = new List<string>();
        foreach (var marker in new[] { "<|im_end|>", "<|eot_id|>", "<|end|>", "</s>" })
        {
            if (marker != eosToken)
                stopSequences.Add(marker);
        }

        return (model, gguf, tokenizer, template, stopSequences);
    }

    private const string DefaultChatMlTemplate =
        "{% for message in messages %}" +
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}" +
        "{% endfor %}" +
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    /// <summary>
    /// Generates a single-turn chat response using the chat template pipeline.
    /// </summary>
    private static string GenerateChatResponse(
        string filePath, string systemPrompt, string userMessage, int maxTokens = 64)
    {
        var (model, gguf, tokenizer, template, stopSequences) = LoadChatModel(filePath);
        using var _ = gguf;
        using var __ = model;

        // Build conversation
        var history = new List<ChatMessage>();
        if (!string.IsNullOrEmpty(systemPrompt))
            history.Add(new ChatMessage { Role = "system", Content = systemPrompt });
        history.Add(new ChatMessage { Role = "user", Content = userMessage });

        // Apply template
        string prompt = template.Apply(history, new ChatTemplateOptions { AddGenerationPrompt = true });

        // Generate
        var options = new InferenceOptions
        {
            Temperature = 0f,
            MaxTokens = maxTokens,
            StopSequences = stopSequences,
        };

        var generator = new TextGenerator(model, tokenizer);
        var response = generator.Generate(prompt, options);

        // Strip stop sequence suffixes
        string text = response.Text;
        foreach (var seq in stopSequences)
        {
            if (text.EndsWith(seq, StringComparison.Ordinal))
                text = text[..^seq.Length];
        }

        return text.Trim();
    }

    /// <summary>
    /// SmolLM2-135M-Instruct with ChatML template: system prompt sets the assistant's name,
    /// and the model responds accordingly on the first turn.
    /// </summary>
    [Collection("SmolLM2Instruct")]
    public class SmolLM2InstructChatTests
    {
        private readonly SmolLM2InstructFixture _fixture;

        public SmolLM2InstructChatTests(SmolLM2InstructFixture fixture)
        {
            _fixture = fixture;
        }

        [Fact]
        public void ChatTemplate_ParsesAndFormatsCorrectly()
        {
            var (model, gguf, tokenizer, template, _) = LoadChatModel(_fixture.FilePath);
            using var _ = gguf;
            using var __ = model;

            var messages = new List<ChatMessage>
            {
                new() { Role = "system", Content = "You are a helpful assistant." },
                new() { Role = "user", Content = "Hello!" },
            };

            string prompt = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

            // ChatML format should contain the role markers
            Assert.Contains("<|im_start|>system", prompt);
            Assert.Contains("<|im_start|>user", prompt);
            Assert.Contains("<|im_start|>assistant", prompt);
            Assert.Contains("You are a helpful assistant.", prompt);
            Assert.Contains("Hello!", prompt);
        }

        [Fact]
        public void FirstTurn_ProducesCoherentResponse()
        {
            string response = GenerateChatResponse(
                _fixture.FilePath,
                systemPrompt: "You are a helpful assistant, your name is Robocop.",
                userMessage: "Hello!");

            // The model should produce a non-empty response
            Assert.False(string.IsNullOrEmpty(response), "Chat response should not be empty.");

            // SmolLM2-135M-Instruct with greedy decoding and this system prompt
            // consistently mentions "Robocop" in the greeting
            Assert.Contains("Robocop", response, StringComparison.OrdinalIgnoreCase);
        }
    }

    /// <summary>
    /// Llama-3.2-1B-Instruct with its complex Jinja2 template: dict literals, slicing,
    /// strftime_now, and tool-use formatting all exercise the template engine.
    /// </summary>
    [Collection("Llama32Instruct")]
    public class Llama32InstructChatTests
    {
        private readonly Llama32InstructFixture _fixture;

        public Llama32InstructChatTests(Llama32InstructFixture fixture)
        {
            _fixture = fixture;
        }

        [Fact]
        public void ChatTemplate_ParsesComplexLlamaTemplate()
        {
            var (model, gguf, tokenizer, template, _) = LoadChatModel(_fixture.FilePath);
            using var _ = gguf;
            using var __ = model;

            var messages = new List<ChatMessage>
            {
                new() { Role = "system", Content = "You are a helpful assistant." },
                new() { Role = "user", Content = "Hello!" },
            };

            string prompt = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

            // Llama 3.2 format uses header tokens
            Assert.Contains("<|start_header_id|>system<|end_header_id|>", prompt);
            Assert.Contains("<|start_header_id|>user<|end_header_id|>", prompt);
            Assert.Contains("<|start_header_id|>assistant<|end_header_id|>", prompt);
            Assert.Contains("You are a helpful assistant.", prompt);
            Assert.Contains("Hello!", prompt);
            // Should include the date preamble
            Assert.Contains("Cutting Knowledge Date:", prompt);
        }

        [Fact]
        public void FirstTurn_ProducesCoherentResponse()
        {
            string response = GenerateChatResponse(
                _fixture.FilePath,
                systemPrompt: "You are a helpful assistant.",
                userMessage: "Hello!");

            // The model should produce a non-empty, coherent response
            Assert.False(string.IsNullOrEmpty(response), "Chat response should not be empty.");

            // Llama-3.2-1B-Instruct with greedy decoding should produce a greeting
            // that contains common response patterns
            bool hasGreeting = response.Contains("help", StringComparison.OrdinalIgnoreCase)
                            || response.Contains("assist", StringComparison.OrdinalIgnoreCase)
                            || response.Contains("Hello", StringComparison.OrdinalIgnoreCase)
                            || response.Contains("Hi", StringComparison.OrdinalIgnoreCase)
                            || response.Contains("welcome", StringComparison.OrdinalIgnoreCase);

            Assert.True(hasGreeting,
                $"Expected a greeting/help response, got: \"{response[..Math.Min(200, response.Length)]}\"");
        }
    }
}
