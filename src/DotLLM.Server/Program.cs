using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Server;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;

// Parse CLI arguments
var serverOptions = ServerOptions.Parse(args);

// Resolve model path (local file or HuggingFace cached model)
Console.WriteLine($"[dotllm] Resolving model: {serverOptions.Model}");
var resolvedPath = ResolveModelPath(serverOptions.Model, serverOptions.Quant);
if (resolvedPath is null)
{
    Console.Error.WriteLine("Failed to resolve model path. Provide a .gguf file path or a pre-downloaded HuggingFace repo.");
    Console.Error.WriteLine("Download models via: dotllm model pull <repo>");
    return 1;
}

// Load model
Console.WriteLine($"[dotllm] Loading model from {resolvedPath}...");
var gguf = GgufFile.Open(resolvedPath);
var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

var threading = new ThreadingConfig(serverOptions.Threads, serverOptions.DecodeThreads);

int gpuLayers = serverOptions.GpuLayers.HasValue
    ? Math.Clamp(serverOptions.GpuLayers.Value, 0, config.NumLayers)
    : serverOptions.Device.StartsWith("gpu", StringComparison.OrdinalIgnoreCase) ? config.NumLayers : 0;

IModel model;
if (gpuLayers <= 0)
{
    Console.WriteLine($"[dotllm] CPU inference ({threading.EffectiveThreadCount} threads)");
    model = TransformerModel.LoadFromGguf(gguf, config, threading);
}
else if (gpuLayers >= config.NumLayers)
{
    int gpuId = serverOptions.Device.IndexOf(':') is int ci and > 0
        ? int.Parse(serverOptions.Device.AsSpan(ci + 1)) : 0;
    Console.WriteLine($"[dotllm] GPU {gpuId} inference");
    model = DotLLM.Cuda.CudaTransformerModel.LoadFromGguf(gguf, config, gpuId);
}
else
{
    int gpuId = serverOptions.Device.IndexOf(':') is int ci2 and > 0
        ? int.Parse(serverOptions.Device.AsSpan(ci2 + 1)) : 0;
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

// Create tool call parser
var toolCallParser = GgufChatTemplateFactory.CreateToolCallParser(gguf.Metadata, config.Architecture);

// KV-cache configuration
var kvConfig = new KvCacheConfig(
    KvCacheConfig.ParseDType(serverOptions.CacheTypeK),
    KvCacheConfig.ParseDType(serverOptions.CacheTypeV));

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
    kvFactory = (cfg, size) => new QuantizedKvCache(
        cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, size,
        kvConfig.KeyDType, kvConfig.ValueDType, kvConfig.MixedPrecisionWindowSize);
}

// Build server
var builder = WebApplication.CreateBuilder(args);
builder.Services.AddSingleton<IModel>(model);
builder.Services.AddSingleton<ITokenizer>(tokenizer);
builder.Services.AddSingleton(chatTemplate);
builder.Services.AddSingleton(new TextGenerator(model, tokenizer, kvFactory));

var serverState = new ServerState
{
    Options = serverOptions,
    Config = config,
    ToolCallParser = toolCallParser,
    KvCacheConfig = kvConfig,
    KvCacheFactory = kvFactory,
    IsReady = true,
};
builder.Services.AddSingleton(serverState);

// CORS — permissive for development and Chat UI (Step 53)
builder.Services.AddCors(o => o.AddDefaultPolicy(p =>
    p.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader()));

var app = builder.Build();
app.UseCors();
app.MapDotLLMEndpoints();

var url = $"http://{serverOptions.Host}:{serverOptions.Port}";
Console.WriteLine($"[dotllm] {config.Architecture} {config.NumLayers}L/{config.HiddenSize}H | {Path.GetFileName(resolvedPath)}");
Console.WriteLine($"[dotllm] Listening on {url}");
Console.WriteLine($"[dotllm] Endpoints: /v1/chat/completions, /v1/completions, /v1/models, /v1/tokenize, /v1/detokenize");

app.Run(url);

// Cleanup
model.Dispose();
gguf.Dispose();
return 0;

// ── Helpers ──

static string? ResolveModelPath(string modelArg, string? quant)
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
        > 1 => ggufFiles.OrderByDescending(f => new FileInfo(f).Length).First(), // pick largest
        _ => null,
    };
}
