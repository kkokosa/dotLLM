using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

public class JinjaChatTemplateTests
{
    // Normalize line endings: raw string literals on Windows contain \r\n,
    // but real GGUF templates always use \n.
    private static string Normalize(string s) => s.Replace("\r\n", "\n");

    // ── Real template strings from popular models ──

    // ChatML format used by Qwen2, many models
    private const string ChatMlTemplate =
        """
        {% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
        """;

    // Llama 3.1 Instruct template (simplified but representative)
    private const string Llama3Template =
        """
        {{- bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>

        {{ message['content'] }}<|eot_id|>{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>

        {{ message['content'] }}<|eot_id|>{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>

        {{ message['content'] }}<|eot_id|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

        {% endif %}
        """;

    // Mistral Instruct template
    private const string MistralTemplate =
        """
        {{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %} {{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}
        """;

    // SmolLM-style template (ChatML variant)
    private const string SmolLmTemplate =
        """
        {% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
        You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
        {% endif %}<|im_start|>{{ message['role'] }}
        {{ message['content'] }}<|im_end|>
        {% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
        {% endif %}
        """;

    // Template with namespace() for loop state (Llama 3.1 pattern)
    private const string NamespaceTemplate =
        """
        {%- set ns = namespace(has_system=false) -%}
        {%- for message in messages -%}
        {%- if message['role'] == 'system' -%}
        {%- set ns.has_system = true -%}
        {%- endif -%}
        {%- endfor -%}
        {%- if not ns.has_system -%}
        <|system|>You are a helpful assistant.<|end|>
        {% endif -%}
        {%- for message in messages -%}
        <|{{ message['role'] }}|>{{ message['content'] }}<|end|>
        {% endfor -%}
        {%- if add_generation_prompt -%}
        <|assistant|>
        {%- endif -%}
        """;

    // ── ChatML tests ──

    [Fact]
    public void ChatML_UserAssistant_SimpleConversation()
    {
        var template = new JinjaChatTemplate(Normalize(ChatMlTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hello!" },
            new() { Role = "assistant", Content = "Hi there!" },
            new() { Role = "user", Content = "How are you?" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.Contains("<|im_start|>user\nHello!<|im_end|>", result);
        Assert.Contains("<|im_start|>assistant\nHi there!<|im_end|>", result);
        Assert.Contains("<|im_start|>user\nHow are you?<|im_end|>", result);
        Assert.EndsWith("<|im_start|>assistant\n", result);
    }

    [Fact]
    public void ChatML_NoGenerationPrompt()
    {
        var template = new JinjaChatTemplate(Normalize(ChatMlTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hello!" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = false });

        Assert.Contains("<|im_start|>user\nHello!<|im_end|>", result);
        Assert.DoesNotContain("<|im_start|>assistant", result);
    }

    [Fact]
    public void ChatML_WithSystemMessage()
    {
        var template = new JinjaChatTemplate(Normalize(ChatMlTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are helpful." },
            new() { Role = "user", Content = "Hello!" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.Contains("<|im_start|>system\nYou are helpful.<|im_end|>", result);
        Assert.Contains("<|im_start|>user\nHello!<|im_end|>", result);
    }

    // ── Llama 3.1 tests ──

    [Fact]
    public void Llama3_BasicConversation()
    {
        var template = new JinjaChatTemplate(Normalize(Llama3Template), "<|begin_of_text|>", "<|eot_id|>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are helpful." },
            new() { Role = "user", Content = "What is 2+2?" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.StartsWith("<|begin_of_text|>", result);
        Assert.Contains("<|start_header_id|>system<|end_header_id|>", result);
        Assert.Contains("You are helpful.", result);
        Assert.Contains("<|start_header_id|>user<|end_header_id|>", result);
        Assert.Contains("What is 2+2?", result);
        Assert.Contains("<|start_header_id|>assistant<|end_header_id|>", result);
    }

    [Fact]
    public void Llama3_MultiTurn()
    {
        var template = new JinjaChatTemplate(Normalize(Llama3Template), "<|begin_of_text|>", "<|eot_id|>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hi" },
            new() { Role = "assistant", Content = "Hello!" },
            new() { Role = "user", Content = "Bye" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.StartsWith("<|begin_of_text|>", result);
        // All messages should appear in order
        int userHi = result.IndexOf("Hi", StringComparison.Ordinal);
        int assistantHello = result.IndexOf("Hello!", StringComparison.Ordinal);
        int userBye = result.IndexOf("Bye", StringComparison.Ordinal);
        Assert.True(userHi < assistantHello);
        Assert.True(assistantHello < userBye);
    }

    // ── Mistral tests ──

    [Fact]
    public void Mistral_UserAssistant()
    {
        var template = new JinjaChatTemplate(Normalize(MistralTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What is AI?" },
            new() { Role = "assistant", Content = "AI is artificial intelligence." },
            new() { Role = "user", Content = "Tell me more." },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = false });

        Assert.StartsWith("<s>", result);
        Assert.Contains("[INST] What is AI? [/INST]", result);
        Assert.Contains("AI is artificial intelligence.</s>", result);
        Assert.Contains("[INST] Tell me more. [/INST]", result);
    }

    // ── SmolLM tests ──

    [Fact]
    public void SmolLM_DefaultSystemMessage()
    {
        var template = new JinjaChatTemplate(Normalize(SmolLmTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hello!" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        // When no system message, SmolLM inserts a default system message
        Assert.Contains("<|im_start|>system\nYou are a helpful AI assistant named SmolLM", result);
        Assert.Contains("<|im_start|>user\nHello!", result);
        Assert.Contains("<|im_start|>assistant\n", result);
    }

    [Fact]
    public void SmolLM_WithExplicitSystemMessage()
    {
        var template = new JinjaChatTemplate(Normalize(SmolLmTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are a poet." },
            new() { Role = "user", Content = "Write a haiku." },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        // Should NOT have the default system message
        Assert.DoesNotContain("SmolLM", result);
        Assert.Contains("<|im_start|>system\nYou are a poet.", result);
    }

    // ── Namespace pattern tests ──

    [Fact]
    public void Namespace_DetectsSystemMessage()
    {
        var template = new JinjaChatTemplate(Normalize(NamespaceTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "Custom system." },
            new() { Role = "user", Content = "Hello!" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        // Should NOT have default system because we provided one
        Assert.DoesNotContain("You are a helpful assistant.", result);
        Assert.Contains("<|system|>Custom system.<|end|>", result);
        Assert.Contains("<|user|>Hello!<|end|>", result);
        Assert.Contains("<|assistant|>", result);
    }

    [Fact]
    public void Namespace_InjectsDefaultSystemMessage()
    {
        var template = new JinjaChatTemplate(Normalize(NamespaceTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hello!" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.Contains("<|system|>You are a helpful assistant.<|end|>", result);
        Assert.Contains("<|user|>Hello!<|end|>", result);
    }

    // ── Edge cases ──

    [Fact]
    public void EmptyMessages()
    {
        var template = new JinjaChatTemplate(Normalize(ChatMlTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>();

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.Equal("<|im_start|>assistant\n", result);
    }

    [Fact]
    public void SingleUserMessage()
    {
        var template = new JinjaChatTemplate(Normalize(ChatMlTemplate), "<s>", "</s>");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hi" },
        };

        var result = template.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = true });

        Assert.Equal("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", result);
    }

    [Fact]
    public void BosAndEosTokens_Accessible()
    {
        // Template that uses bos_token and eos_token
        const string template = "{{ bos_token }}hello{{ eos_token }}";
        var tmpl = new JinjaChatTemplate(template, "<BOS>", "<EOS>");
        var result = tmpl.Apply(new List<ChatMessage>(), new ChatTemplateOptions());

        Assert.Equal("<BOS>hello<EOS>", result);
    }

    // ── Tool definitions ──

    [Fact]
    public void Tools_AvailableInContext()
    {
        const string template = "{% if tools %}Tools: {{ tools | length }}{% else %}No tools{% endif %}";
        var tmpl = new JinjaChatTemplate(template, "<s>", "</s>");

        var withTools = tmpl.Apply(
            new List<ChatMessage>(),
            new ChatTemplateOptions
            {
                Tools = [new ToolDefinition("get_weather", "Get weather info", """{"type":"object"}""")]
            });
        Assert.Equal("Tools: 1", withTools);

        var withoutTools = tmpl.Apply(
            new List<ChatMessage>(),
            new ChatTemplateOptions());
        Assert.Equal("No tools", withoutTools);
    }

    // ── Whitespace control in real templates ──

    [Fact]
    public void WhitespaceControl_InForLoop()
    {
        const string template = "{%- for msg in messages -%}[{{ msg.role }}]{{ msg.content }}{%- endfor -%}";
        var tmpl = new JinjaChatTemplate(template, "", "");
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "a" },
            new() { Role = "assistant", Content = "b" },
        };

        var result = tmpl.Apply(messages, new ChatTemplateOptions { AddGenerationPrompt = false });
        Assert.Equal("[user]a[assistant]b", result);
    }

    // ── Filter chains in templates ──

    [Fact]
    public void FilterChain_TrimThenTojson()
    {
        const string template = "{{ '  hello  ' | trim | tojson }}";
        var tmpl = new JinjaChatTemplate(template, "", "");
        var result = tmpl.Apply(new List<ChatMessage>(), new ChatTemplateOptions());
        Assert.Equal("\"hello\"", result);
    }

    // ── Not in operator ──

    [Fact]
    public void NotInOperator()
    {
        const string template = "{% if 'x' not in items %}missing{% else %}found{% endif %}";
        var tmpl = new JinjaChatTemplate(template, "", "");

        // Can't easily set custom vars through Apply, so test through evaluator directly
        var lexer = new JinjaLexer(template);
        var tokens = lexer.Tokenize();
        var parser = new JinjaParser(tokens);
        var ast = parser.Parse();

        var eval = new JinjaEvaluator(new Dictionary<string, object?>
        {
            ["items"] = new List<object?> { "a", "b" },
            ["messages"] = new List<object?>(),
            ["add_generation_prompt"] = false,
        });
        var result = eval.Evaluate(ast);
        Assert.Equal("missing", result);
    }
}
