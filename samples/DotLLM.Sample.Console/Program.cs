using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.Samplers;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;

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
using var gguf = GgufFile.Open(modelPath);
var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
using var model = LlamaModel.LoadFromGguf(gguf, config);
var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

Console.WriteLine($"Model: {config.Architecture}, {config.NumLayers} layers, {config.VocabSize} vocab");
Console.WriteLine($"Prompt: \"{prompt}\"");
Console.WriteLine();

var generator = new TextGenerator(model, tokenizer);

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
    MaxTokens = 128
};

var response = generator.Generate(prompt, options);

Console.Write(prompt);
Console.WriteLine(response.Text);
Console.WriteLine();
Console.WriteLine($"[Prompt tokens: {response.PromptTokenCount}, Generated: {response.GeneratedTokenCount}, Finish: {response.FinishReason}]");

var t = response.Timings;
Console.WriteLine($"[Prefill: {t.PrefillTimeMs:F1} ms ({t.PrefillTokensPerSec:F1} tok/s), " +
    $"Decode: {t.DecodeTimeMs:F1} ms ({t.DecodeTokensPerSec:F1} tok/s), " +
    $"Sampling: {t.SamplingTimeMs:F1} ms]");

return 0;
