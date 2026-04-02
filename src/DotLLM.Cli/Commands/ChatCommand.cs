using System.ComponentModel;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using DotLLM.Cli.Helpers;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.Constraints;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.ToolCallParsers;
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
        [Description("Constrain model output format: 'text' (default), 'json_object', 'json_schema', 'regex', or 'grammar'.")]
        [DefaultValue("text")]
        public string ResponseFormat { get; set; } = "text";

        /// <summary>JSON Schema for json_schema response format.</summary>
        [CommandOption("--schema")]
        [Description("JSON Schema string or file path (prefixed with @) for json_schema response format.")]
        public string? Schema { get; set; }

        /// <summary>Regex pattern for regex response format.</summary>
        [CommandOption("--pattern")]
        [Description("Regex pattern for regex response format. Entire output must match.")]
        public string? Pattern { get; set; }

        /// <summary>GBNF grammar for grammar response format.</summary>
        [CommandOption("--grammar")]
        [Description("GBNF grammar string or file path (prefixed with @) for grammar response format.")]
        public string? Grammar { get; set; }

        /// <summary>Tool definitions.</summary>
        [CommandOption("--tools")]
        [Description("Tool definitions: JSON array string or file path (prefixed with @). " +
                     "Each tool: {\"name\": \"...\", \"description\": \"...\", \"parameters\": {...}}.")]
        public string? Tools { get; set; }

        /// <summary>Tool choice strategy.</summary>
        [CommandOption("--tool-choice")]
        [Description("Tool choice: 'auto' (default), 'none', 'required', or a function name.")]
        [DefaultValue("auto")]
        public string ToolChoiceStr { get; set; } = "auto";

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

        // Parse tool definitions
        ToolDefinition[]? tools = ParseTools(settings.Tools);
        IToolCallParser? toolCallParser = tools is { Length: > 0 }
            ? GgufChatTemplateFactory.CreateToolCallParser(gguf.Metadata, config!.Architecture)
            : null;
        ToolChoice toolChoice = ParseToolChoice(settings.ToolChoiceStr, tools);

        // Common end-of-turn markers used by chat templates.
        // The EOS stop condition handles eos_token_id, but the end-of-turn marker
        // may be a different token (e.g., <|im_end|> in ChatML, <|eot_id|> in Llama 3).
        var stopSequences = new List<string>();
        foreach (var marker in new[] { "<|im_end|>", "<|eot_id|>", "<|end|>", "</s>" })
        {
            if (marker != eosTokenStr) // avoid duplicate with EOS stop condition
                stopSequences.Add(marker);
        }

        var responseFormat = settings.ResponseFormat.ToLowerInvariant() switch
        {
            "json_object" => (Core.Configuration.ResponseFormat)new Core.Configuration.ResponseFormat.JsonObject(),
            "json_schema" => BuildJsonSchemaFormat(settings.Schema),
            "regex" => BuildRegexFormat(settings.Pattern),
            "grammar" => BuildGrammarFormat(settings.Grammar),
            _ => null
        };

        // When tool_choice is required or specific function, constrain output to valid tool call JSON
        if (responseFormat is null && tools is { Length: > 0 })
        {
            string? argumentsKey = toolCallParser is LlamaToolCallParser ? "parameters" : "arguments";
            responseFormat = toolChoice switch
            {
                ToolChoice.Required => new Core.Configuration.ResponseFormat.JsonSchema
                {
                    Schema = ToolCallSchemaBuilder.BuildForRequired(tools, argumentsKey),
                    Name = "tool_call"
                },
                ToolChoice.Function fn => new Core.Configuration.ResponseFormat.JsonSchema
                {
                    Schema = ToolCallSchemaBuilder.BuildForFunction(
                        tools.First(t => t.Name == fn.Name), argumentsKey),
                    Name = "tool_call"
                },
                _ => null // auto/none — no constraint
            };
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
        var toolsLabel = tools is { Length: > 0 } ? $" | {tools.Length} tool(s)" : "";
        var segments = $"{config!.Architecture} {config.NumLayers}L/{config.HiddenSize}H | {quantLabel} | {deviceLabel} | {samplingLabel}{toolsLabel}";
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
            await RunRepl(generator, chatTemplate, inferenceOptions, history, settings, tools, toolCallParser);
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
        Settings settings,
        ToolDefinition[]? tools,
        IToolCallParser? toolCallParser)
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

            if (input.Equals("/tools", StringComparison.OrdinalIgnoreCase))
            {
                if (tools is not { Length: > 0 })
                    AnsiConsole.MarkupLine("[dim]No tools configured. Use --tools to provide tool definitions.[/]");
                else
                {
                    foreach (var tool in tools)
                        AnsiConsole.MarkupLine($"[blue]{Markup.Escape(tool.Name)}[/]: {Markup.Escape(tool.Description)}");
                }
                continue;
            }

            // Add user message
            history.Add(new ChatMessage { Role = "user", Content = input });

            // Generate (with tool call detection loop)
            await GenerateAndHandleToolCalls(generator, chatTemplate, options, history, tools, toolCallParser);
        }
    }

    private static async Task GenerateAndHandleToolCalls(
        TextGenerator generator,
        IChatTemplate chatTemplate,
        InferenceOptions options,
        List<ChatMessage> history,
        ToolDefinition[]? tools,
        IToolCallParser? toolCallParser)
    {
        var templateOptions = new ChatTemplateOptions
        {
            AddGenerationPrompt = true,
            Tools = tools
        };
        string prompt = chatTemplate.Apply(history, templateOptions);

        // Generate response via streaming
        var sw = Stopwatch.StartNew();
        int tokenCount = 0;
        long firstTokenTicks = 0;
        var sb = new StringBuilder();
        var accumulator = toolCallParser is not null ? new StreamingToolCallAccumulator(toolCallParser) : null;
        FinishReason finishReason = FinishReason.Length;
        InferenceTimings timings = default;
        int promptTokenCount = 0;

        await foreach (var token in generator.GenerateStreamingTokensAsync(prompt, options))
        {
            if (tokenCount == 0 && token.Text.Length > 0)
                firstTokenTicks = sw.ElapsedTicks;

            bool suppress = accumulator?.Append(token.Text) ?? false;
            if (!suppress)
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

        // Check for tool calls
        ToolCall[]? detectedCalls = toolCallParser?.TryParse(assistantText);

        if (detectedCalls is { Length: > 0 })
        {
            finishReason = FinishReason.ToolCalls;

            // Add assistant message with tool calls to history
            history.Add(new ChatMessage
            {
                Role = "assistant",
                Content = assistantText.TrimEnd(),
                ToolCalls = detectedCalls
            });

            // Display detected tool calls
            AnsiConsole.MarkupLine("[dim]Tool calls detected:[/]");
            foreach (var tc in detectedCalls)
            {
                AnsiConsole.MarkupLine(
                    $"  [blue][[{Markup.Escape(tc.Id)}]][/] [green]{Markup.Escape(tc.FunctionName)}[/]({Markup.Escape(tc.Arguments)})");
            }

            // Prompt user for tool results
            foreach (var tc in detectedCalls)
            {
                AnsiConsole.MarkupLine($"[dim]Result for {Markup.Escape(tc.FunctionName)} (Enter to skip):[/]");
                string result;
                try
                {
                    result = AnsiConsole.Prompt(new TextPrompt<string>("[tool]>>>").AllowEmpty());
                }
                catch (InvalidOperationException)
                {
                    result = "";
                }

                if (string.IsNullOrWhiteSpace(result))
                    result = "{}";

                history.Add(new ChatMessage
                {
                    Role = "tool",
                    Content = result,
                    ToolCallId = tc.Id
                });
            }

            // Print timing info for tool call generation
            PrintTimingInfo(firstTokenTicks, promptTokenCount, tokenCount, timings);

            // Re-generate with tool results in history
            AnsiConsole.MarkupLine("[dim]Generating response with tool results...[/]");
            await GenerateAndHandleToolCalls(generator, chatTemplate, options, history, tools, toolCallParser);
        }
        else
        {
            // Normal text response — add to history
            history.Add(new ChatMessage { Role = "assistant", Content = assistantText.TrimEnd() });
            PrintTimingInfo(firstTokenTicks, promptTokenCount, tokenCount, timings);
        }
    }

    private static void PrintTimingInfo(long firstTokenTicks, int promptTokenCount, int tokenCount, InferenceTimings timings)
    {
        double ttftMs = firstTokenTicks > 0 ? firstTokenTicks * 1000.0 / Stopwatch.Frequency : 0;
        double prefillTokSec = timings.PrefillTokensPerSec;
        double decodeTokSec = timings.DecodeTokensPerSec;
        AnsiConsole.MarkupLine(
            $"[dim][[{promptTokenCount} prompt tokens, {tokenCount} generated tokens, " +
            $"{ttftMs:F0} ms TTFT, {prefillTokSec:F1} prefill tok/s, {decodeTokSec:F1} decode tok/s]][/]");
        Console.WriteLine();
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

    private static Core.Configuration.ResponseFormat BuildJsonSchemaFormat(string? schema)
    {
        if (string.IsNullOrEmpty(schema))
            throw new InvalidOperationException("--schema is required when --response-format is json_schema");

        string schemaJson = schema.StartsWith('@')
            ? File.ReadAllText(schema[1..])
            : schema;

        return new Core.Configuration.ResponseFormat.JsonSchema { Schema = schemaJson };
    }

    private static Core.Configuration.ResponseFormat BuildRegexFormat(string? pattern)
    {
        if (string.IsNullOrEmpty(pattern))
            throw new InvalidOperationException("--pattern is required when --response-format is regex");

        return new Core.Configuration.ResponseFormat.Regex { Pattern = pattern };
    }

    private static Core.Configuration.ResponseFormat BuildGrammarFormat(string? grammar)
    {
        if (string.IsNullOrEmpty(grammar))
            throw new InvalidOperationException("--grammar is required when --response-format is grammar");

        string grammarText = grammar.StartsWith('@')
            ? File.ReadAllText(grammar[1..])
            : grammar;

        return new Core.Configuration.ResponseFormat.Grammar { GbnfGrammar = grammarText };
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

    private static ToolDefinition[]? ParseTools(string? toolsInput)
    {
        if (string.IsNullOrEmpty(toolsInput))
            return null;

        string json = toolsInput.StartsWith('@')
            ? File.ReadAllText(toolsInput[1..])
            : toolsInput;

        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Array)
                throw new InvalidOperationException("--tools must be a JSON array of tool definitions.");

            var tools = new List<ToolDefinition>();
            foreach (var element in root.EnumerateArray())
            {
                // Support both flat format and OpenAI-style {"type":"function","function":{...}}
                JsonElement funcElement = element;
                if (element.TryGetProperty("type", out var typeProp) &&
                    typeProp.GetString() == "function" &&
                    element.TryGetProperty("function", out var funcProp))
                {
                    funcElement = funcProp;
                }

                string name = funcElement.GetProperty("name").GetString()!;
                string description = funcElement.TryGetProperty("description", out var descProp)
                    ? descProp.GetString() ?? ""
                    : "";
                string parameters = funcElement.TryGetProperty("parameters", out var paramsProp)
                    ? paramsProp.GetRawText()
                    : "{}";

                tools.Add(new ToolDefinition(name, description, parameters));
            }
            return tools.ToArray();
        }
        catch (JsonException ex)
        {
            throw new InvalidOperationException($"Invalid --tools JSON: {ex.Message}");
        }
    }

    private static ToolChoice ParseToolChoice(string choice, ToolDefinition[]? tools)
    {
        return choice.ToLowerInvariant() switch
        {
            "auto" => new ToolChoice.Auto(),
            "none" => new ToolChoice.None(),
            "required" => new ToolChoice.Required(),
            _ => tools?.Any(t => t.Name == choice) == true
                ? new ToolChoice.Function(choice)
                : throw new InvalidOperationException(
                    $"Unknown --tool-choice '{choice}'. Use 'auto', 'none', 'required', or a function name.")
        };
    }

    // Default ChatML template used as fallback when GGUF has no chat_template
    private const string DefaultChatMlTemplate =
        "{% for message in messages %}" +
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}" +
        "{% endfor %}" +
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
}
