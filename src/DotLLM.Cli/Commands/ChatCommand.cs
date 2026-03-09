using System.ComponentModel;
using System.Diagnostics;
using System.Text.RegularExpressions;
using DotLLM.Cli.Helpers;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Interactive multi-turn chat REPL: load model → apply chat template → stream tokens.
/// Maintains conversation history across turns with Jinja2 template formatting.
/// </summary>
internal sealed class ChatCommand : Command<ChatCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        /// <summary>Path to a GGUF file or HuggingFace repo ID.</summary>
        [CommandArgument(0, "<model>")]
        [Description("Path to a GGUF file or HuggingFace repo ID (e.g., QuantFactory/SmolLM-135M-GGUF).")]
        public string Model { get; set; } = string.Empty;

        /// <summary>System prompt for the conversation.</summary>
        [CommandOption("--system|-s")]
        [Description("System prompt for the conversation.")]
        public string? SystemPrompt { get; set; }

        /// <summary>Maximum tokens per response.</summary>
        [CommandOption("--max-tokens|-n")]
        [Description("Maximum number of tokens to generate per response.")]
        [DefaultValue(512)]
        public int MaxTokens { get; set; } = 512;

        /// <summary>Sampling temperature.</summary>
        [CommandOption("--temp|-t")]
        [Description("Sampling temperature. 0 = greedy (default).")]
        [DefaultValue(0f)]
        public float Temperature { get; set; }

        /// <summary>Top-K sampling.</summary>
        [CommandOption("--top-k")]
        [Description("Top-K sampling. 0 = disabled.")]
        [DefaultValue(0)]
        public int TopK { get; set; }

        /// <summary>Top-P sampling.</summary>
        [CommandOption("--top-p")]
        [Description("Top-P (nucleus) sampling threshold.")]
        [DefaultValue(1.0f)]
        public float TopP { get; set; } = 1.0f;

        /// <summary>Min-P sampling.</summary>
        [CommandOption("--min-p")]
        [Description("Min-P sampling threshold. 0 = disabled.")]
        [DefaultValue(0f)]
        public float MinP { get; set; }

        /// <summary>Repetition penalty.</summary>
        [CommandOption("--repeat-penalty")]
        [Description("Repetition penalty factor. 1.0 = disabled.")]
        [DefaultValue(1.0f)]
        public float RepeatPenalty { get; set; } = 1.0f;

        /// <summary>Repetition penalty lookback window.</summary>
        [CommandOption("--repeat-last-n")]
        [Description("Number of recent tokens for repetition penalty lookback. 0 = full history.")]
        [DefaultValue(0)]
        public int RepeatLastN { get; set; }

        /// <summary>Random seed for reproducibility.</summary>
        [CommandOption("--seed")]
        [Description("Random seed for reproducible sampling. Omit for non-deterministic.")]
        public int? Seed { get; set; }

        /// <summary>CPU thread count.</summary>
        [CommandOption("--threads")]
        [Description("Number of CPU threads for inference. 0 = auto/all cores (default).")]
        [DefaultValue(0)]
        public int Threads { get; set; }

        /// <summary>Quantization filter.</summary>
        [CommandOption("--quant|-q")]
        [Description("Quantization filter when multiple GGUF files exist (e.g., Q4_K_M, Q8_0).")]
        public string? Quant { get; set; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        var resolvedPath = GgufFileResolver.Resolve(settings.Model, settings.Quant);
        if (resolvedPath is null)
            return 1;

        GgufFile? gguf = null;
        ModelConfig? config = null;
        Tokenizers.Bpe.BpeTokenizer? tokenizer = null;
        LlamaModel? model = null;

        AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .Start("Loading model...", ctx =>
            {
                ctx.Status("Opening GGUF file...");
                gguf = GgufFile.Open(resolvedPath);

                ctx.Status("Extracting model config...");
                config = GgufModelConfigExtractor.Extract(gguf.Metadata);

                ctx.Status("Loading tokenizer...");
                tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

                var threading = new ThreadingConfig(settings.Threads);
                ctx.Status($"Loading {config.Architecture} model ({config.NumLayers} layers, {config.HiddenSize} hidden, {threading.EffectiveThreadCount} threads)...");
                model = LlamaModel.LoadFromGguf(gguf, config, threading);
            });

        // Create chat template from GGUF metadata, fallback to ChatML
        string bosTokenStr = tokenizer!.DecodeToken(tokenizer.BosTokenId);
        string eosTokenStr = tokenizer.DecodeToken(tokenizer.EosTokenId);
        IChatTemplate chatTemplate;
        var jinjaTemplate = GgufChatTemplateFactory.TryCreate(gguf!.Metadata, tokenizer);
        chatTemplate = jinjaTemplate ?? new JinjaChatTemplate(DefaultChatMlTemplate, bosTokenStr, eosTokenStr);

        // Common end-of-turn markers used by chat templates.
        // The EOS stop condition handles eos_token_id, but the end-of-turn marker
        // may be a different token (e.g., <|im_end|> in ChatML, <|eot_id|> in Llama 3).
        var stopSequences = new List<string>();
        foreach (var marker in new[] { "<|im_end|>", "<|eot_id|>", "<|end|>", "</s>" })
        {
            if (marker != eosTokenStr) // avoid duplicate with EOS stop condition
                stopSequences.Add(marker);
        }

        var inferenceOptions = new InferenceOptions
        {
            Temperature = settings.Temperature,
            TopK = settings.TopK,
            TopP = settings.TopP,
            MinP = settings.MinP,
            RepetitionPenalty = settings.RepeatPenalty,
            RepetitionPenaltyWindow = settings.RepeatLastN,
            MaxTokens = settings.MaxTokens,
            Seed = settings.Seed,
            StopSequences = stopSequences,
            Threading = new ThreadingConfig(settings.Threads)
        };

        // Print header
        var threadingInfo = new ThreadingConfig(settings.Threads);
        var quantLabel = InferQuantLabel(resolvedPath, settings.Quant);
        var samplingLabel = BuildSamplingLabel(settings);
        var segments = $"{config!.Architecture} {config.NumLayers}L/{config.HiddenSize}H | {quantLabel} | {threadingInfo.EffectiveThreadCount} threads | {samplingLabel}";
        AnsiConsole.Write(new Rule($"[grey]dotllm chat | {Markup.Escape(segments)}[/]").LeftJustified());
        AnsiConsole.MarkupLine("[dim]Type /exit to quit, /clear to reset history, /system <text> to set system prompt.[/]");
        AnsiConsole.WriteLine();

        // Initialize conversation
        var history = new List<ChatMessage>();
        if (!string.IsNullOrEmpty(settings.SystemPrompt))
            history.Add(new ChatMessage { Role = "system", Content = settings.SystemPrompt });

        var generator = new TextGenerator(model!, tokenizer);

        try
        {
            RunRepl(generator, chatTemplate, inferenceOptions, history, tokenizer, settings);
        }
        finally
        {
            model?.Dispose();
            gguf?.Dispose();
        }

        return 0;
    }

    private static void RunRepl(
        TextGenerator generator,
        IChatTemplate chatTemplate,
        InferenceOptions options,
        List<ChatMessage> history,
        Tokenizers.Bpe.BpeTokenizer tokenizer,
        Settings settings)
    {
        while (true)
        {
            Console.Write(">>> ");
            string? input = Console.ReadLine();

            if (input is null)
                break; // EOF / Ctrl+C

            input = input.Trim();
            if (string.IsNullOrEmpty(input))
                continue;

            // Handle special commands
            if (input.Equals("/exit", StringComparison.OrdinalIgnoreCase) ||
                input.Equals("/quit", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            if (input.Equals("/clear", StringComparison.OrdinalIgnoreCase))
            {
                // Keep system prompt if present
                var systemMsg = history.Find(m => m.Role == "system");
                history.Clear();
                if (systemMsg != null)
                    history.Add(systemMsg);
                AnsiConsole.MarkupLine("[dim]History cleared.[/]");
                continue;
            }

            if (input.StartsWith("/system ", StringComparison.OrdinalIgnoreCase))
            {
                string systemText = input[8..].Trim();
                // Remove existing system message and add new one
                history.RemoveAll(m => m.Role == "system");
                history.Insert(0, new ChatMessage { Role = "system", Content = systemText });
                AnsiConsole.MarkupLine($"[dim]System prompt set.[/]");
                continue;
            }

            // Add user message
            history.Add(new ChatMessage { Role = "user", Content = input });

            // Apply chat template to full history
            string prompt = chatTemplate.Apply(history, new ChatTemplateOptions { AddGenerationPrompt = true });

            // Generate response
            var sw = Stopwatch.StartNew();
            int tokenCount = 0;
            long firstTokenTicks = 0;

            var response = generator.Generate(prompt, options,
                onTokenGenerated: tokenId =>
                {
                    if (tokenCount == 0)
                        firstTokenTicks = sw.ElapsedTicks;
                    Console.Write(tokenizer.DecodeToken(tokenId));
                    tokenCount++;
                });

            sw.Stop();
            Console.WriteLine();

            // Strip any remaining stop sequence suffixes from the response text
            string assistantText = response.Text;
            foreach (var seq in options.StopSequences)
            {
                if (assistantText.EndsWith(seq, StringComparison.Ordinal))
                    assistantText = assistantText[..^seq.Length];
            }

            // Add assistant response to history
            history.Add(new ChatMessage { Role = "assistant", Content = assistantText.TrimEnd() });

            // Print timing info
            double ttftMs = firstTokenTicks > 0 ? firstTokenTicks * 1000.0 / Stopwatch.Frequency : 0;
            int promptTokens = response.PromptTokenCount;
            var timings = response.Timings;
            double prefillTokSec = timings.PrefillTokensPerSec;
            double decodeTokSec = timings.DecodeTokensPerSec;
            AnsiConsole.MarkupLine(
                $"[dim][[{promptTokens} prompt tokens, {tokenCount} generated tokens, " +
                $"{ttftMs:F0} ms TTFT, {prefillTokSec:F1} prefill tok/s, {decodeTokSec:F1} decode tok/s]][/]");
            Console.WriteLine();
        }
    }

    private static string InferQuantLabel(string resolvedPath, string? quantFlag)
    {
        if (!string.IsNullOrEmpty(quantFlag))
            return quantFlag;

        var match = Regex.Match(Path.GetFileName(resolvedPath), @"\.(Q[\w]+)\.gguf$", RegexOptions.IgnoreCase);
        return match.Success ? match.Groups[1].Value : "unknown";
    }

    private static string BuildSamplingLabel(Settings settings)
    {
        if (settings.Temperature <= 0)
            return "greedy";

        var parts = new List<string> { $"temp={settings.Temperature:F1}" };
        if (settings.TopK > 0) parts.Add($"top-k={settings.TopK}");
        if (settings.TopP < 1.0f) parts.Add($"top-p={settings.TopP:F2}");
        if (settings.MinP > 0f) parts.Add($"min-p={settings.MinP:F2}");
        if (settings.RepeatPenalty != 1.0f) parts.Add($"rep={settings.RepeatPenalty:F2}");
        if (settings.Seed.HasValue) parts.Add($"seed={settings.Seed.Value}");
        return string.Join(", ", parts);
    }

    // Default ChatML template used as fallback when GGUF has no chat_template
    private const string DefaultChatMlTemplate =
        "{% for message in messages %}" +
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}" +
        "{% endfor %}" +
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
}
