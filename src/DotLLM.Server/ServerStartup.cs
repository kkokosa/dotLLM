using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;

namespace DotLLM.Server;

/// <summary>
/// Shared server startup logic used by both the standalone DotLLM.Server exe
/// and the CLI <c>dotllm serve</c> command.
/// </summary>
public static class ServerStartup
{
    /// <summary>
    /// Resolves a model argument (file path or HuggingFace repo ID) to a local GGUF path.
    /// </summary>
    public static string? ResolveModelPath(string modelArg, string? quant)
    {
        // Direct .gguf file path
        if (modelArg.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) && File.Exists(modelArg))
            return modelArg;

        // HuggingFace repo ID — check cached models directory
        var modelsDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models");

        var repoDir = Path.Combine(modelsDir, modelArg.Replace('/', Path.DirectorySeparatorChar));
        if (!Directory.Exists(repoDir))
            return null;

        var ggufFiles = Directory.GetFiles(repoDir, "*.gguf");
        if (quant is not null)
        {
            ggufFiles = ggufFiles.Where(f =>
                Path.GetFileName(f).Contains(quant, StringComparison.OrdinalIgnoreCase)).ToArray();
        }

        return ggufFiles.Length switch
        {
            1 => ggufFiles[0],
            > 1 => ggufFiles.OrderByDescending(f => new FileInfo(f).Length).First(),
            _ => null,
        };
    }

    /// <summary>
    /// Creates a bare <see cref="ServerState"/> with no model loaded.
    /// The server starts and serves UI, but inference requests return 503 until a model is loaded.
    /// </summary>
    public static ServerState CreateBareState(ServerOptions options) => new()
    {
        Options = options,
        IsReady = false,
    };

    /// <summary>
    /// Loads a model from the given GGUF path and returns a fully populated <see cref="ServerState"/>.
    /// </summary>
    public static ServerState LoadModel(string resolvedPath, ServerOptions options)
    {
        Console.WriteLine($"[dotllm] Loading model from {resolvedPath}...");
        var gguf = GgufFile.Open(resolvedPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        var threading = new ThreadingConfig(options.Threads, options.DecodeThreads);

        int gpuLayers = options.GpuLayers.HasValue
            ? Math.Clamp(options.GpuLayers.Value, 0, config.NumLayers)
            : options.Device.StartsWith("gpu", StringComparison.OrdinalIgnoreCase) ? config.NumLayers : 0;

        IModel model;
        if (gpuLayers <= 0)
        {
            Console.WriteLine($"[dotllm] CPU inference ({threading.EffectiveThreadCount} threads)");
            model = TransformerModel.LoadFromGguf(gguf, config, threading);
        }
        else if (gpuLayers >= config.NumLayers)
        {
            int gpuId = ParseGpuId(options.Device);
            Console.WriteLine($"[dotllm] GPU {gpuId} inference");
            model = DotLLM.Cuda.CudaTransformerModel.LoadFromGguf(gguf, config, gpuId);
        }
        else
        {
            int gpuId = ParseGpuId(options.Device);
            Console.WriteLine($"[dotllm] Hybrid inference ({gpuLayers} GPU + {config.NumLayers - gpuLayers} CPU layers)");
            model = DotLLM.Cuda.HybridTransformerModel.LoadFromGguf(gguf, config, gpuLayers, gpuId, threading);
        }

        // Create chat template
        string bosToken = tokenizer.DecodeToken(tokenizer.BosTokenId);
        string eosToken = tokenizer.DecodeToken(tokenizer.EosTokenId);
        IChatTemplate chatTemplate = GgufChatTemplateFactory.TryCreate(gguf.Metadata, tokenizer)
            ?? new JinjaChatTemplate(
                "{% for message in messages %}" +
                "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}" +
                "{% endfor %}" +
                "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
                bosToken, eosToken);

        // Tool call parser
        var toolCallParser = GgufChatTemplateFactory.CreateToolCallParser(gguf.Metadata, config.Architecture);

        // KV-cache configuration
        var kvConfig = new KvCacheConfig(
            KvCacheConfig.ParseDType(options.CacheTypeK),
            KvCacheConfig.ParseDType(options.CacheTypeV));

        Func<ModelConfig, int, IKvCache>? kvFactory = null;
        PagedKvCacheFactory? pagedFactory = null;
        if (model is DotLLM.Cuda.CudaTransformerModel cudaModel)
        {
            if (options.UsePaged)
                Console.WriteLine("[dotllm] Paged KV-cache not supported with CUDA, using GPU cache.");
            kvFactory = kvConfig.IsQuantized
                ? (cfg, size) => cudaModel.CreateKvCache(size, kvConfig)
                : (cfg, size) => cudaModel.CreateKvCache(size);
        }
        else if (model is DotLLM.Cuda.HybridTransformerModel hybridModel)
        {
            if (options.UsePaged)
                Console.WriteLine("[dotllm] Paged KV-cache not supported with hybrid GPU, using hybrid cache.");
            kvFactory = (cfg, size) => hybridModel.CreateKvCache(size);
        }
        else if (options.UsePaged && !kvConfig.IsQuantized)
        {
            pagedFactory = new PagedKvCacheFactory(
                config.NumLayers, config.NumKvHeads, config.HeadDim);
            kvFactory = (cfg, size) => pagedFactory.Create(size);
            Console.WriteLine("[dotllm] Using paged KV-cache (block-based allocation)");
        }
        else if (options.UsePaged && kvConfig.IsQuantized)
        {
            Console.WriteLine("[dotllm] Paged KV-cache does not support quantization yet, using quantized simple cache.");
            kvFactory = (cfg, size) => new QuantizedKvCache(
                cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, size,
                kvConfig.KeyDType, kvConfig.ValueDType, kvConfig.MixedPrecisionWindowSize);
        }
        else if (kvConfig.IsQuantized)
        {
            kvFactory = (cfg, size) => new QuantizedKvCache(
                cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, size,
                kvConfig.KeyDType, kvConfig.ValueDType, kvConfig.MixedPrecisionWindowSize);
        }

        PrefixCache? prefixCache = options.PromptCacheEnabled
            ? new PrefixCache(options.PromptCacheSize)
            : null;

        // Load speculative draft model if configured
        IModel? draftModel = null;
        GgufFile? draftGguf = null;
        string draftModelPath = "";
        if (!string.IsNullOrEmpty(options.SpeculativeModel))
        {
            var draftPath = ResolveModelPath(options.SpeculativeModel, null);
            if (draftPath is null)
                throw new InvalidOperationException($"Speculative draft model not found: {options.SpeculativeModel}");

            draftGguf = GgufFile.Open(draftPath);
            var draftConfig = GgufModelConfigExtractor.Extract(draftGguf.Metadata);
            if (!SpeculativeConstants.AreVocabsCompatible(config.VocabSize, draftConfig.VocabSize))
            {
                draftGguf.Dispose();
                throw new InvalidOperationException(
                    $"Draft model vocab size ({draftConfig.VocabSize}) differs from target ({config.VocabSize}) " +
                    $"by more than {SpeculativeConstants.MaxVocabSizeDifference} tokens. " +
                    "Models must share the same base tokenizer.");
            }
            if (draftConfig.VocabSize != config.VocabSize)
                Console.WriteLine($"[dotllm] Note: vocab sizes differ slightly ({draftConfig.VocabSize} vs {config.VocabSize}) — using shared range for speculative comparison.");

            var draftThreading = new ThreadingConfig(options.Threads, options.DecodeThreads);
            draftModel = TransformerModel.LoadFromGguf(draftGguf, draftConfig, draftThreading);
            draftModelPath = draftPath;
            Console.WriteLine($"[dotllm] Speculative decoding: draft={Path.GetFileName(draftPath)}, K={options.SpeculativeCandidates}");
        }

        var generator = new TextGenerator(model, tokenizer, kvFactory, prefixCache,
            draftModel: draftModel, speculativeCandidates: options.SpeculativeCandidates);

        // Warm-up: JIT pre-compilation + CUDA kernel loading
        WarmupRunner.Run(generator, tokenizer, options.Warmup);
        prefixCache?.Clear(); // Discard warm-up KV-cache entries

        return new ServerState
        {
            Options = options,
            Config = config,
            ToolCallParser = toolCallParser,
            KvCacheConfig = kvConfig,
            KvCacheFactory = kvFactory,
            PagedFactory = pagedFactory,
            PrefixCache = prefixCache,
            IsReady = true,
            Model = model,
            Tokenizer = tokenizer,
            ChatTemplate = chatTemplate,
            Generator = generator,
            LoadedModelPath = resolvedPath,
            CurrentGguf = gguf,
            DraftModel = draftModel,
            DraftModelPath = draftModelPath,
            DraftGguf = draftGguf,
        };
    }

    /// <summary>
    /// Builds and configures a <see cref="WebApplication"/> with all dotLLM endpoints.
    /// </summary>
    /// <param name="state">Populated server state with loaded model.</param>
    /// <param name="args">Raw command-line arguments for ASP.NET configuration.</param>
    /// <param name="serveUi">When true, also serves the embedded web chat UI.</param>
    public static WebApplication BuildApp(ServerState state, string[] args, bool serveUi = false)
    {
        var builder = WebApplication.CreateSlimBuilder(args);
        builder.Services.AddSingleton(state);

        // Wire source-generated JSON context for AOT-compatible serialization
        builder.Services.ConfigureHttpJsonOptions(options =>
            options.SerializerOptions.TypeInfoResolverChain.Insert(0, ServerJsonContext.Default));

        // CORS — permissive for development and Chat UI
        builder.Services.AddCors(o => o.AddDefaultPolicy(p =>
            p.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader()));

        // Keep only warning+ logging to avoid noisy request logs
        builder.Logging.SetMinimumLevel(LogLevel.Warning);

        var app = builder.Build();
        app.UseDeveloperExceptionPage();
        app.UseCors();
        app.MapDotLLMEndpoints(serveUi);

        return app;
    }

    private static int ParseGpuId(string device) =>
        device.IndexOf(':') is int ci and > 0
            ? int.Parse(device.AsSpan(ci + 1))
            : 0;
}
