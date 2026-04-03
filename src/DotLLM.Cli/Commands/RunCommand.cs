using System.ComponentModel;
using System.Diagnostics;
using System.Text.Json;
using System.Text.RegularExpressions;
using DotLLM.Cli.Helpers;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.ToolCallParsers;
using Spectre.Console;
using Spectre.Console.Cli;
using Spectre.Console.Rendering;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Runs text generation on a GGUF model: load → encode prompt → stream tokens via TextGenerator.
/// Supports greedy (default) and sampled decoding via composable sampling pipeline.
/// </summary>
internal sealed class RunCommand : AsyncCommand<RunCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("Path to a GGUF file or HuggingFace repo ID (e.g., QuantFactory/SmolLM-135M-GGUF).")]
        public string Model { get; set; } = string.Empty;

        [CommandOption("--prompt|-p")]
        [Description("Input prompt for generation (required).")]
        public string? Prompt { get; set; }

        [CommandOption("--max-tokens|-n")]
        [Description("Maximum number of tokens to generate.")]
        [DefaultValue(128)]
        public int MaxTokens { get; set; } = 128;

        [CommandOption("--temp|-t")]
        [Description("Sampling temperature. 0 = greedy (default).")]
        [DefaultValue(0f)]
        public float Temperature { get; set; }

        [CommandOption("--top-k")]
        [Description("Top-K sampling. 0 = disabled.")]
        [DefaultValue(0)]
        public int TopK { get; set; }

        [CommandOption("--top-p")]
        [Description("Top-P (nucleus) sampling threshold.")]
        [DefaultValue(1.0f)]
        public float TopP { get; set; } = 1.0f;

        [CommandOption("--min-p")]
        [Description("Min-P sampling threshold. 0 = disabled.")]
        [DefaultValue(0f)]
        public float MinP { get; set; }

        [CommandOption("--repeat-penalty")]
        [Description("Repetition penalty factor. 1.0 = disabled.")]
        [DefaultValue(1.0f)]
        public float RepeatPenalty { get; set; } = 1.0f;

        [CommandOption("--repeat-last-n")]
        [Description("Number of recent tokens for repetition penalty lookback. 0 = full history.")]
        [DefaultValue(0)]
        public int RepeatLastN { get; set; }

        [CommandOption("--seed|-s")]
        [Description("Random seed for reproducible sampling. Omit for non-deterministic.")]
        public int? Seed { get; set; }

        [CommandOption("--threads")]
        [Description("Number of CPU threads for inference. 0 = auto/all cores (default), 1 = single-threaded.")]
        [DefaultValue(0)]
        public int Threads { get; set; }

        [CommandOption("--decode-threads")]
        [Description("Number of threads for decode. 0 = auto (caps at memory channel count).")]
        [DefaultValue(0)]
        public int DecodeThreads { get; set; }

        [CommandOption("--numa-pin")]
        [Description("Pin workers to NUMA-local cores on multi-socket systems.")]
        [DefaultValue(false)]
        public bool NumaPin { get; set; }

        [CommandOption("--pcore-only")]
        [Description("Pin workers to P-cores only (Intel hybrid architectures).")]
        [DefaultValue(false)]
        public bool PCoreOnly { get; set; }

        [CommandOption("--device|-d")]
        [Description("Compute device: 'cpu' (default), 'gpu', 'gpu:0', 'gpu:1'.")]
        [DefaultValue("cpu")]
        public string Device { get; set; } = "cpu";

        [CommandOption("--gpu-layers")]
        [Description("Number of transformer layers to offload to GPU. 0 = CPU only. " +
                     "Omit for default (0 with --device cpu, all with --device gpu).")]
        public int? GpuLayers { get; set; }

        [CommandOption("--quant|-q")]
        [Description("Quantization filter when multiple GGUF files exist (e.g., Q4_K_M, Q8_0).")]
        public string? Quant { get; set; }

        [CommandOption("--json")]
        [Description("Output result as a single JSON object (suppresses all formatted output).")]
        [DefaultValue(false)]
        public bool Json { get; set; }

        [CommandOption("--response-format")]
        [Description("Constrain model output format: 'text' (default), 'json_object', 'json_schema', 'regex', or 'grammar'.")]
        [DefaultValue("text")]
        public string ResponseFormat { get; set; } = "text";

        [CommandOption("--schema")]
        [Description("JSON Schema string or file path (prefixed with @) for json_schema response format.")]
        public string? Schema { get; set; }

        [CommandOption("--pattern")]
        [Description("Regex pattern for regex response format. Entire output must match.")]
        public string? Pattern { get; set; }

        [CommandOption("--grammar")]
        [Description("GBNF grammar string or file path (prefixed with @) for grammar response format.")]
        public string? Grammar { get; set; }

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

        /// <summary>Tool definitions.</summary>
        [CommandOption("--tools")]
        [Description("Tool definitions: JSON array string or file path (prefixed with @). " +
                     "When provided, the prompt is formatted via the model's chat template with tool definitions.")]
        public string? Tools { get; set; }
    }

    public override async Task<int> ExecuteAsync(CommandContext context, Settings settings)
    {
        if (string.IsNullOrEmpty(settings.Prompt))
        {
            if (settings.Json)
                Console.Error.WriteLine("Error: --prompt|-p is required.");
            else
                AnsiConsole.MarkupLine("[red]--prompt|-p is required.[/]");
            return 1;
        }

        var resolvedPath = GgufFileResolver.Resolve(settings.Model, settings.Quant);
        if (resolvedPath is null)
            return 1;

        GgufFile gguf = null!;
        ModelConfig config = null!;
        Tokenizers.Bpe.BpeTokenizer tokenizer = null!;
        IModel model = null!;

        void LoadModel()
        {
            gguf = GgufFile.Open(resolvedPath);
            config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

            int gpuLayers = ResolveGpuLayers(settings, config);
            if (gpuLayers <= 0)
            {
                model = TransformerModel.LoadFromGguf(gguf, config,
                    new ThreadingConfig(settings.Threads, settings.DecodeThreads, settings.NumaPin, settings.PCoreOnly));
            }
            else if (gpuLayers >= config.NumLayers)
            {
                int gpuId = ParseGpuId(settings.Device);
                model = DotLLM.Cuda.CudaTransformerModel.LoadFromGguf(gguf, config, gpuId);
            }
            else
            {
                int gpuId = ParseGpuId(settings.Device);
                model = DotLLM.Cuda.HybridTransformerModel.LoadFromGguf(gguf, config, gpuLayers, gpuId,
                    new ThreadingConfig(settings.Threads, settings.DecodeThreads, settings.NumaPin, settings.PCoreOnly));
            }
        }

        var loadSw = Stopwatch.StartNew();
        if (settings.Json)
        {
            LoadModel();
        }
        else
        {
            AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .Start("Loading model...", _ => LoadModel());
        }
        loadSw.Stop();

        // Display VRAM warning after spinner completes (so it stays visible).
        // In JSON mode, write to stderr so it doesn't corrupt the JSON output.
        string? vramWarning = (model as DotLLM.Cuda.CudaTransformerModel)?.VramWarning
                           ?? (model as DotLLM.Cuda.HybridTransformerModel)?.VramWarning;
        if (vramWarning is not null)
        {
            if (settings.Json)
                Console.Error.WriteLine($"WARNING: {vramWarning}");
            else
                AnsiConsole.MarkupLine($"[yellow]WARNING: {Markup.Escape(vramWarning)}[/]");
        }

        var threadingInfo = new ThreadingConfig(settings.Threads, settings.DecodeThreads, settings.NumaPin, settings.PCoreOnly);

        // Parse tool definitions and format prompt via chat template when tools are provided
        ToolDefinition[]? tools = ChatCommand.ParseToolDefinitions(settings.Tools);
        IToolCallParser? toolCallParser = null;
        string effectivePrompt = settings.Prompt;
        if (tools is { Length: > 0 })
        {
            string bosToken = tokenizer.DecodeToken(tokenizer.BosTokenId);
            string eosToken = tokenizer.DecodeToken(tokenizer.EosTokenId);
            var chatTemplate = GgufChatTemplateFactory.TryCreate(gguf.Metadata, tokenizer)
                ?? new JinjaChatTemplate(ChatCommand.DefaultChatMlTemplateText, bosToken, eosToken);

            var messages = new List<ChatMessage>
            {
                new() { Role = "user", Content = settings.Prompt }
            };
            effectivePrompt = chatTemplate.Apply(messages, new ChatTemplateOptions
            {
                AddGenerationPrompt = true,
                Tools = tools
            });
            toolCallParser = GgufChatTemplateFactory.CreateToolCallParser(gguf.Metadata, config.Architecture);
        }

        // Build inference options from CLI flags
        var responseFormat = settings.ResponseFormat.ToLowerInvariant() switch
        {
            "json_object" => (Core.Configuration.ResponseFormat)new Core.Configuration.ResponseFormat.JsonObject(),
            "json_schema" => BuildJsonSchemaFormat(settings.Schema),
            "regex" => BuildRegexFormat(settings.Pattern),
            "grammar" => BuildGrammarFormat(settings.Grammar),
            _ => null
        };
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
            ResponseFormat = responseFormat,
            Threading = threadingInfo
        };

        if (!settings.Json)
        {
            // Build compact pre-gen header rule
            var quantLabel = InferQuantLabel(resolvedPath, settings.Quant);
            var samplingLabel = BuildSamplingLabel(settings);
            var deviceLabel = model switch
            {
                DotLLM.Cuda.CudaTransformerModel => DotLLM.Cuda.CudaDevice.GetDevice(ParseGpuId(settings.Device)).ToString(),
                DotLLM.Cuda.HybridTransformerModel h => $"hybrid {h.NumGpuLayers}gpu/{config.NumLayers - h.NumGpuLayers}cpu",
                _ => $"{threadingInfo.EffectiveThreadCount} threads"
            };
            var segments = $"{config.Architecture} {config.NumLayers}L/{config.HiddenSize}H | {quantLabel} | {deviceLabel} | {samplingLabel}";
            AnsiConsole.Write(new Rule($"[grey]dotllm | {Markup.Escape(segments)}[/]").LeftJustified());
            AnsiConsole.WriteLine();
        }

        try
        {
            // Add stop sequences for tool calling end-of-turn tokens
            if (tools is { Length: > 0 })
            {
                string eosTokenStr = tokenizer.DecodeToken(tokenizer.EosTokenId);
                var toolStopSeqs = new List<string>();
                foreach (var marker in new[] { "<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|end|>", "</s>", "</tool_call>" })
                {
                    if (marker != eosTokenStr)
                        toolStopSeqs.Add(marker);
                }
                inferenceOptions = inferenceOptions with { StopSequences = toolStopSeqs };
            }

            if (!settings.Json)
                Console.Write(tools is { Length: > 0 } ? "" : settings.Prompt);

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

            var generator = new TextGenerator(model, tokenizer, kvFactory);
            var totalSw = Stopwatch.StartNew();
            int generated = 0;
            InferenceTimings timings = default;
            FinishReason finishReason = FinishReason.Length;
            var generatedText = new System.Text.StringBuilder();

            await foreach (var token in generator.GenerateStreamingTokensAsync(effectivePrompt, inferenceOptions))
            {
                if (settings.Json)
                    generatedText.Append(token.Text);
                else
                    Console.Write(token.Text);

                if (token.FinishReason is null || token.Text.Length > 0)
                    generated++;
                if (token.FinishReason.HasValue)
                {
                    finishReason = token.FinishReason.Value;
                    timings = token.Timings ?? default;
                }
            }

            totalSw.Stop();

            // Read timings from streaming result
            double loadMs = loadSw.Elapsed.TotalMilliseconds;
            double promptEvalMs = timings.PrefillTimeMs;
            double evalMs = timings.DecodeTimeMs;
            double samplerMs = timings.SamplingTimeMs;
            double totalMs = totalSw.Elapsed.TotalMilliseconds;
            int promptLen = timings.PrefillTokenCount;
            int evalSteps = timings.DecodeTokenCount;

            // Compute metrics
            int totalTokens = promptLen + generated;
            double decodeTokPerSec = evalSteps > 0 ? evalSteps / (evalMs / 1000.0) : 0;
            double prefillTokPerSec = promptLen > 0 ? promptLen / (promptEvalMs / 1000.0) : 0;
            double totalTokPerSec = totalTokens > 0 ? totalTokens / (totalMs / 1000.0) : 0;

            // Memory metrics
            long fileSize = new FileInfo(resolvedPath).Length;
            long modelWeightsBytes = fileSize - gguf.DataSectionOffset;
            long computeBytes = model.ComputeMemoryBytes;
            int cacheSize = Math.Min(promptLen + settings.MaxTokens, config.MaxSequenceLength);
            // Use actual KV-cache bytes from engine timings (reflects quantization compression).
            // Fall back to computed estimate for GPU caches (based on config).
            long kvCacheBytes;
            if (timings.KvCacheBytes > 0)
                kvCacheBytes = timings.KvCacheBytes;
            else if (kvConfig.IsQuantized)
                kvCacheBytes = ComputeQuantizedKvBytes(config, cacheSize, kvConfig);
            else
                kvCacheBytes = (long)config.NumLayers * 2 * cacheSize
                    * config.NumKvHeads * config.HeadDim
                    * (model is DotLLM.Cuda.CudaTransformerModel ? sizeof(ushort) : sizeof(float));
            long totalMemory = modelWeightsBytes + computeBytes + kvCacheBytes;

            // Detect tool calls in generated output
            string outputText = generatedText.ToString();
            ToolCall[]? detectedToolCalls = null;
            if (toolCallParser is not null && outputText.Length > 0)
            {
                // Strip stop sequence suffixes before parsing
                foreach (var seq in inferenceOptions.StopSequences)
                {
                    if (outputText.EndsWith(seq, StringComparison.Ordinal))
                    {
                        outputText = outputText[..^seq.Length];
                        break;
                    }
                }
                detectedToolCalls = toolCallParser.TryParse(outputText);
                if (detectedToolCalls is { Length: > 0 })
                    finishReason = FinishReason.ToolCalls;
            }

            if (settings.Json)
            {
                var result = new RunJsonResult
                {
                    Text = outputText,
                    Prompt = settings.Prompt,
                    Model = Path.GetFileName(resolvedPath),
                    Architecture = config.Architecture.ToString(),
                    FinishReason = finishReason.ToString().ToLowerInvariant(),
                    ToolCalls = detectedToolCalls?.Select(tc => new RunToolCallDto
                    {
                        Id = tc.Id,
                        FunctionName = tc.FunctionName,
                        Arguments = tc.Arguments,
                    }).ToArray(),
                    Usage = new RunUsageDto
                    {
                        PromptTokens = promptLen,
                        GeneratedTokens = generated,
                    },
                    Timings = new RunTimingsDto
                    {
                        LoadMs = Math.Round(loadMs, 1),
                        PrefillMs = Math.Round(promptEvalMs, 1),
                        DecodeMs = Math.Round(evalMs, 1),
                        SamplingMs = Math.Round(samplerMs, 1),
                        TotalMs = Math.Round(totalMs, 1),
                        PrefillTokS = Math.Round(prefillTokPerSec, 2),
                        DecodeTokS = Math.Round(decodeTokPerSec, 2),
                    },
                    Memory = new RunMemoryDto
                    {
                        WeightsBytes = modelWeightsBytes,
                        ComputeBytes = computeBytes,
                        KvCacheBytes = kvCacheBytes,
                        TotalBytes = totalMemory,
                    },
                };
                Console.WriteLine(JsonSerializer.Serialize(result, CliJsonContext.Default.RunJsonResult));
            }
            else
            {
                Console.WriteLine();
                AnsiConsole.WriteLine();

                // Header grid: title left, hero metric right
                var headerGrid = new Grid();
                headerGrid.AddColumn(new GridColumn().NoWrap());
                headerGrid.AddColumn(new GridColumn().NoWrap().RightAligned());
                headerGrid.AddRow(
                    new Markup("[bold]Generation Complete[/]"),
                    new Markup($"[bold green]{decodeTokPerSec:F2} tok/s[/]"));

                // Build body lines
                var bodyLines = new List<IRenderable>();
                bodyLines.Add(new Markup("  [bold]Performance[/]"));
                bodyLines.Add(new Markup(PerfLine("Prefill", promptEvalMs, promptLen, prefillTokPerSec)));
                bodyLines.Add(new Markup(PerfLine("Decode", evalMs, evalSteps, decodeTokPerSec)));
                bodyLines.Add(new Markup(PerfLine("Sampling", samplerMs, generated, null)));
                bodyLines.Add(new Markup("  [dim]──────────────────────────────────────────────────────[/]"));
                bodyLines.Add(new Markup(PerfLine("Total", totalMs, totalTokens, totalTokPerSec)));
                bodyLines.Add(new Markup(PerfLine("Load", loadMs, null, null)));
                bodyLines.Add(new Text(""));
                bodyLines.Add(new Markup("  [bold]Memory[/]"));
                bodyLines.Add(new Markup(MemLine("Weights", modelWeightsBytes, "(memory-mapped)")));
                bodyLines.Add(new Markup(MemLine("Compute", computeBytes, null)));
                string kvLabel = kvConfig.IsQuantized
                    ? $"({cacheSize} slots, K:{settings.CacheTypeK} V:{settings.CacheTypeV})"
                    : $"({cacheSize} slots)";
                bodyLines.Add(new Markup(MemLine("KV Cache", kvCacheBytes, kvLabel)));
                bodyLines.Add(new Markup("  [dim]──────────────────────────────────────────────────────[/]"));
                bodyLines.Add(new Markup(MemLine("Total", totalMemory, null)));
                bodyLines.Add(new Text(""));

                var finishReasonStr = finishReason.ToString().ToLowerInvariant();
                bodyLines.Add(new Markup($"  [dim]{Markup.Escape(finishReasonStr)} | {promptLen} prompt, {generated} generated[/]"));

                // Assemble panel
                var panelContent = new Rows(
                    new Text(""),
                    headerGrid,
                    new Text(""),
                    new Rows(bodyLines),
                    new Text(""));

                var panel = new Panel(panelContent)
                    .Border(BoxBorder.Rounded)
                    .Padding(2, 0);

                AnsiConsole.Write(panel);
            }
        }
        finally
        {
            model.Dispose();
            gguf.Dispose();
        }

        return 0;
    }

    private static string PerfLine(string label, double ms, int? tokens, double? tokPerSec)
    {
        var labelPart = $"[dim]{label,-14}[/]";
        var msPart = $"{ms,10:N1} ms";
        var tokensPart = tokens.HasValue ? $"{tokens.Value,6:N0} tokens" : "              ";
        var toksPart = tokPerSec.HasValue ? $"{tokPerSec.Value,10:F2} tok/s" : "";
        return $"  {labelPart} {msPart}   {tokensPart}   {toksPart}";
    }

    private static string MemLine(string label, long bytes, string? annotation)
    {
        var labelPart = $"[dim]{label,-14}[/]";
        var sizePart = $"{FormatHelpers.FormatMiB(bytes),12}";
        var annPart = annotation != null ? $"   [dim]{Markup.Escape(annotation)}[/]" : "";
        return $"  {labelPart} {sizePart}{annPart}";
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

    private static string InferQuantLabel(string resolvedPath, string? quantFlag)
    {
        if (!string.IsNullOrEmpty(quantFlag))
            return quantFlag;

        var match = Regex.Match(Path.GetFileName(resolvedPath), @"\.(Q[\w]+)\.gguf$", RegexOptions.IgnoreCase);
        return match.Success ? match.Groups[1].Value : "unknown";
    }

    private static int ResolveGpuLayers(Settings settings, ModelConfig config)
    {
        if (settings.GpuLayers.HasValue)
            return Math.Clamp(settings.GpuLayers.Value, 0, config.NumLayers);
        // Default: 0 for cpu device, all layers for gpu device
        return settings.Device.StartsWith("gpu", StringComparison.OrdinalIgnoreCase)
            ? config.NumLayers : 0;
    }

    private static int ParseGpuId(string device)
    {
        // "gpu" → 0, "gpu:0" → 0, "gpu:1" → 1
        int colonIdx = device.IndexOf(':');
        if (colonIdx < 0) return 0;
        return int.Parse(device.AsSpan(colonIdx + 1));
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

    private static long ComputeQuantizedKvBytes(ModelConfig config, int cacheSize, KvCacheConfig kvConfig)
    {
        int kvStride = config.NumKvHeads * config.HeadDim;
        int window = Math.Min(kvConfig.MixedPrecisionWindowSize, cacheSize);
        int quantSlots = Math.Max(0, cacheSize - window);
        int fpBytesPerRow = kvStride * sizeof(float); // FP32 on CPU, FP16 on GPU (close enough for estimate)

        int kQuantRowBytes = kvConfig.KeyDType switch
        {
            KvCacheDType.Q8_0 => kvStride / 32 * 34,
            KvCacheDType.Q4_0 => kvStride / 32 * 18,
            _ => fpBytesPerRow
        };
        int vQuantRowBytes = kvConfig.ValueDType switch
        {
            KvCacheDType.Q8_0 => kvStride / 32 * 34,
            KvCacheDType.Q4_0 => kvStride / 32 * 18,
            _ => fpBytesPerRow
        };

        // Quantized region + full-precision window
        long quantBytes = (long)config.NumLayers * quantSlots * (kQuantRowBytes + vQuantRowBytes);
        long windowBytes = (long)config.NumLayers * window * fpBytesPerRow * 2; // K + V
        return quantBytes + windowBytes;
    }

}
