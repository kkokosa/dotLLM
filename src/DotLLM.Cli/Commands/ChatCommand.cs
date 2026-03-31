using System.ComponentModel;
using System.Diagnostics;
using System.Text;
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
internal sealed class ChatCommand : AsyncCommand<ChatCommand.Settings>
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

        /// <summary>Decode thread count.</summary>
        [CommandOption("--decode-threads")]
        [Description("Number of threads for decode. 0 = auto (caps at memory channel count).")]
        [DefaultValue(0)]
        public int DecodeThreads { get; set; }

        /// <summary>NUMA pinning.</summary>
        [CommandOption("--numa-pin")]
        [Description("Pin workers to NUMA-local cores on multi-socket systems.")]
        [DefaultValue(false)]
        public bool NumaPin { get; set; }

        /// <summary>P-core pinning.</summary>
        [CommandOption("--pcore-only")]
        [Description("Pin workers to P-cores only (Intel hybrid architectures).")]
        [DefaultValue(false)]
        public bool PCoreOnly { get; set; }

        /// <summary>Compute device.</summary>
        [CommandOption("--device|-d")]
        [Description("Compute device: 'cpu' (default), 'gpu', 'gpu:0', 'gpu:1'.")]
        [DefaultValue("cpu")]
        public string Device { get; set; } = "cpu";

        /// <summary>Number of GPU layers for hybrid offloading.</summary>
        [CommandOption("--gpu-layers")]
        [Description("Number of transformer layers to offload to GPU. 0 = CPU only. " +
                     "Omit for default (0 with --device cpu, all with --device gpu).")]
        public int? GpuLayers { get; set; }

        /// <summary>Quantization filter.</summary>
        [CommandOption("--quant|-q")]
        [Description("Quantization filter when multiple GGUF files exist (e.g., Q4_K_M, Q8_0).")]
        public string? Quant { get; set; }

        /// <summary>Constrain model output format.</summary>
        [CommandOption("--response-format")]
        [Description("Constrain model output format: 'text' (default) or 'json_object' (valid JSON).")]
        [DefaultValue("text")]
        public string ResponseFormat { get; set; } = "text";

        /// <summary>KV-cache key quantization type.</summary>
        [CommandOption("--cache-type-k")]
        [Description("KV-cache key quantization: f32 (default), q8_0, q4_0.")]
        [DefaultValue("f32")]
        public string CacheTypeK { get; set; } = "f32";

        /// <summary>KV-cache value quantization type.</summary>
        [CommandOption("--cache-type-v")]
        [Description("KV-cache value quantization: f32 (default), q8_0, q4_0.")]
        [DefaultValue("f32")]
        public string CacheTypeV { get; set; } = "f32";

        /// <summary>Mixed-precision window size for KV-cache quantization.</summary>
        [CommandOption("--cache-window")]
        [Description("Mixed-precision window: recent N tokens in full precision (0 = all quantized). Only used when --cache-type-k or --cache-type-v is set.")]
        [DefaultValue(0)]
        public int CacheWindow { get; set; }
    }

    /// <inheritdoc/>
    public override async Task<int> ExecuteAsync(CommandContext context, Settings settings)
    {
        var resolvedPath = GgufFileResolver.Resolve(settings.Model, settings.Quant);
        if (resolvedPath is null)
            return 1;

        GgufFile? gguf = null;
        ModelConfig? config = null;
        Tokenizers.Bpe.BpeTokenizer? tokenizer = null;
        IModel? model = null;

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

                int gpuLayers = ResolveGpuLayers(settings, config);
                if (gpuLayers <= 0)
                {
                    var threading = new ThreadingConfig(settings.Threads, settings.DecodeThreads, settings.NumaPin, settings.PCoreOnly);
                    ctx.Status($"Loading {config.Architecture} model ({config.NumLayers} layers, {threading.EffectiveThreadCount} threads)...");
                    model = TransformerModel.LoadFromGguf(gguf, config, threading);
                }
                else if (gpuLayers >= config.NumLayers)
                {
                    int gpuId = settings.Device.IndexOf(':') is int ci and > 0
                        ? int.Parse(settings.Device.AsSpan(ci + 1))
                        : 0;
                    ctx.Status($"Loading {config.Architecture} model on GPU {gpuId}...");
                    model = DotLLM.Cuda.CudaTransformerModel.LoadFromGguf(gguf, config, gpuId);
                }
                else
                {
                    int gpuId = settings.Device.IndexOf(':') is int ci2 and > 0
                        ? int.Parse(settings.Device.AsSpan(ci2 + 1))
                        : 0;
                    var threading = new ThreadingConfig(settings.Threads, settings.DecodeThreads, settings.NumaPin, settings.PCoreOnly);
                    ctx.Status($"Loading {config.Architecture} model ({gpuLayers} GPU + {config.NumLayers - gpuLayers} CPU layers)...");
                    model = DotLLM.Cuda.HybridTransformerModel.LoadFromGguf(gguf, config, gpuLayers, gpuId, threading);
                }
            });

        // Display VRAM warning after spinner completes (so it stays visible)
        string? vramWarning = (model as DotLLM.Cuda.CudaTransformerModel)?.VramWarning
                           ?? (model as DotLLM.Cuda.HybridTransformerModel)?.VramWarning;
        if (vramWarning is not null)
            AnsiConsole.MarkupLine($"[yellow]WARNING: {Markup.Escape(vramWarning)}[/]");

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

        var responseFormat = settings.ResponseFormat.ToLowerInvariant() == "json_object"
            ? (Core.Configuration.ResponseFormat)new Core.Configuration.ResponseFormat.JsonObject()
            : null;
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
            ResponseFormat = responseFormat,
            Threading = new ThreadingConfig(settings.Threads, settings.DecodeThreads, settings.NumaPin, settings.PCoreOnly)
        };

        // Print header
        var threadingInfo = new ThreadingConfig(settings.Threads, settings.DecodeThreads, settings.NumaPin, settings.PCoreOnly);
        var quantLabel = InferQuantLabel(resolvedPath, settings.Quant);
        var samplingLabel = BuildSamplingLabel(settings);
        var deviceLabel = model is DotLLM.Cuda.CudaTransformerModel
            ? DotLLM.Cuda.CudaDevice.GetDevice(settings.Device.IndexOf(':') is int ci and > 0
                ? int.Parse(settings.Device.AsSpan(ci + 1)) : 0).ToString()
            : $"{threadingInfo.EffectiveThreadCount} threads";
        var segments = $"{config!.Architecture} {config.NumLayers}L/{config.HiddenSize}H | {quantLabel} | {deviceLabel} | {samplingLabel}";
        AnsiConsole.Write(new Rule($"[grey]dotllm chat | {Markup.Escape(segments)}[/]").LeftJustified());
        AnsiConsole.MarkupLine("[dim]Type /exit to quit, /clear to reset history, /system <text> to set system prompt.[/]");
        AnsiConsole.WriteLine();

        // Initialize conversation
        var history = new List<ChatMessage>();
        if (!string.IsNullOrEmpty(settings.SystemPrompt))
            history.Add(new ChatMessage { Role = "system", Content = settings.SystemPrompt });

        var kvConfig = new KvCacheConfig(
            KvCacheConfig.ParseDType(settings.CacheTypeK),
            KvCacheConfig.ParseDType(settings.CacheTypeV),
            settings.CacheWindow);

        Func<ModelConfig, int, DotLLM.Core.Attention.IKvCache>? kvFactory = null;
        if (model is DotLLM.Cuda.CudaTransformerModel cudaModel)
        {
            kvFactory = kvConfig.IsQuantized
                ? (cfg, size) => cudaModel.CreateKvCache(size, kvConfig)
                : (cfg, size) => cudaModel.CreateKvCache(size);
        }
        else if (model is DotLLM.Cuda.HybridTransformerModel hybridModel)
            kvFactory = (cfg, size) => hybridModel.CreateKvCache(size);
        else if (kvConfig.IsQuantized)
        {
            kvFactory = (cfg, size) => new DotLLM.Engine.KvCache.QuantizedKvCache(
                cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, size,
                kvConfig.KeyDType, kvConfig.ValueDType, kvConfig.MixedPrecisionWindowSize);
        }

        var generator = new TextGenerator(model!, tokenizer!, kvFactory);

        try
        {
            await RunRepl(generator, chatTemplate, inferenceOptions, history, settings);
        }
        finally
        {
            model?.Dispose();
            gguf?.Dispose();
        }

        return 0;
    }

    private static async Task RunRepl(
        TextGenerator generator,
        IChatTemplate chatTemplate,
        InferenceOptions options,
        List<ChatMessage> history,
        Settings settings)
    {
        while (true)
        {
            string input;
            try
            {
                input = AnsiConsole.Prompt(new TextPrompt<string>(">>>").AllowEmpty());
            }
            catch (InvalidOperationException)
            {
                break; // stdin redirected or closed
            }

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

            // Generate response via streaming
            var sw = Stopwatch.StartNew();
            int tokenCount = 0;
            long firstTokenTicks = 0;
            var sb = new StringBuilder();
            FinishReason finishReason = FinishReason.Length;
            InferenceTimings timings = default;
            int promptTokenCount = 0;

            await foreach (var token in generator.GenerateStreamingTokensAsync(prompt, options))
            {
                if (tokenCount == 0 && token.Text.Length > 0)
                    firstTokenTicks = sw.ElapsedTicks;
                Console.Write(token.Text);
                sb.Append(token.Text);
                if (token.FinishReason is null || token.Text.Length > 0)
                    tokenCount++;
                if (token.FinishReason.HasValue)
                {
                    finishReason = token.FinishReason.Value;
                    timings = token.Timings ?? default;
                    promptTokenCount = timings.PrefillTokenCount;
                }
            }

            sw.Stop();
            Console.WriteLine();

            // Strip any remaining stop sequence suffixes from the response text
            string assistantText = sb.ToString();
            foreach (var seq in options.StopSequences)
            {
                if (assistantText.EndsWith(seq, StringComparison.Ordinal))
                    assistantText = assistantText[..^seq.Length];
            }

            // Add assistant response to history
            history.Add(new ChatMessage { Role = "assistant", Content = assistantText.TrimEnd() });

            // Print timing info
            double ttftMs = firstTokenTicks > 0 ? firstTokenTicks * 1000.0 / Stopwatch.Frequency : 0;
            double prefillTokSec = timings.PrefillTokensPerSec;
            double decodeTokSec = timings.DecodeTokensPerSec;
            AnsiConsole.MarkupLine(
                $"[dim][[{promptTokenCount} prompt tokens, {tokenCount} generated tokens, " +
                $"{ttftMs:F0} ms TTFT, {prefillTokSec:F1} prefill tok/s, {decodeTokSec:F1} decode tok/s]][/]");
            Console.WriteLine();
        }
    }

    private static int ResolveGpuLayers(Settings settings, ModelConfig config)
    {
        if (settings.GpuLayers.HasValue)
            return Math.Clamp(settings.GpuLayers.Value, 0, config.NumLayers);
        return settings.Device.StartsWith("gpu", StringComparison.OrdinalIgnoreCase)
            ? config.NumLayers : 0;
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
