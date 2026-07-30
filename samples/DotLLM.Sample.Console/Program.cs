using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Engine;
using DotLLM.Engine.Samplers;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Metal;
using DotLLM.Metal.Weights.Strategies;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

if (args.Length < 1)
{
    Console.Error.WriteLine("Usage: dotLLM.Sample.Console <model.gguf> [prompt]");
    Console.Error.WriteLine("  model.gguf  Path to a GGUF model file");
    Console.Error.WriteLine("  prompt      Text prompt (default: \"The capital of France is\")");
    return 1;
}

string modelPath = args[0];
string prompt = args.Length > 1 ? string.Join(' ', args.Skip(1)) : "The capital of France is";

Console.WriteLine($"Loading model: {modelPath}");

var runCtx = LoadModel();
IModel model = runCtx.Model;
GgufFile gguf = runCtx.Gguf;
ModelConfig config = runCtx.Config;

var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

Console.WriteLine($"Model: {config.Architecture}, {config.NumLayers} layers, {config.VocabSize} vocab");
Console.WriteLine($"Prompt: \"{prompt}\"");
Console.WriteLine();

var generator = new TextGenerator(model, tokenizer, kvCacheFactory: runCtx.KvCacheFactory);

// --- Composable sampling pipeline ---
var options = new InferenceOptions
{
    SamplerSteps =
    [
        new TemperatureSampler(0.8f),
        new TopKSampler(40),
        new TopPSampler(0.95f),
        new MinPSampler(0.05f)
    ],
    StopConditions =
    [
        new EosStopCondition(tokenizer.EosTokenId),
        new MaxTokensStopCondition(128)
    ],
    Seed = 42,
    MaxTokens = 128,
};

// --- Streaming generation via IAsyncEnumerable ---
Console.Write(prompt);

InferenceTimings timings = default;
int tokenCount = 0;

await foreach (var token in generator.GenerateStreamingTokensAsync(prompt, options))
{
    Console.Write(token.Text);
    tokenCount++;
    if (token.Timings.HasValue)
        timings = token.Timings.Value;
}

Console.WriteLine();
Console.WriteLine();
Console.WriteLine($"[Prompt tokens: {timings.PrefillTokenCount}, Generated: {tokenCount}, " +
    $"Decode steps: {timings.DecodeTokenCount}]");
Console.WriteLine($"[Prefill: {timings.PrefillTimeMs:F1} ms ({timings.PrefillTokensPerSec:F1} tok/s), " +
    $"Decode: {timings.DecodeTimeMs:F1} ms ({timings.DecodeTokensPerSec:F1} tok/s), " +
    $"Sampling: {timings.SamplingTimeMs:F1} ms]");

return 0;

ModelRunContext LoadModel()
{
    ModelRunContext LoadModelFromCpu()
    {
        var ggufFile = GgufFile.Open(modelPath);
        var modelConfig = GgufModelConfigExtractor.Extract(ggufFile.Metadata);
        var transformerModel = TransformerModel.LoadFromGguf(ggufFile, modelConfig);

        return new ModelRunContext(transformerModel, ggufFile, modelConfig);
    }

    var forceCpuVar = Environment.GetEnvironmentVariable("FORCE_CPU_LLM");
    if (!string.IsNullOrEmpty(forceCpuVar) && bool.TryParse(forceCpuVar, out var forceCpu) && forceCpu)
    {
        return LoadModelFromCpu();
    }

    if (CudaDevice.IsAvailable())
    {
        var device = CudaDevice.GetDevice(0);
        Console.WriteLine($"CUDA available: using {device}.");
        return CudaModelLoader.LoadFromGguf(modelPath, deviceId: 0);
    }

    if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
    {
        return MetalModelLoader.LoadFromGguf(modelPath, new DequantToFp16Strategy());
    }

    return LoadModelFromCpu();
}
