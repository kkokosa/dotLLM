using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;

if (args.Length < 1)
{
    Console.Error.WriteLine("Usage: DotLLM.Sample.Logprobs <model.gguf> [prompt]");
    Console.Error.WriteLine("  model.gguf  Path to a GGUF model file");
    Console.Error.WriteLine("  prompt      Text prompt (default: \"The capital of France is\")");
    return 1;
}

string modelPath = args[0];
string prompt = args.Length > 1 ? string.Join(' ', args.Skip(1)) : "The capital of France is";

Console.WriteLine($"Loading model: {modelPath}");
using var gguf = GgufFile.Open(modelPath);
var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
using var model = TransformerModel.LoadFromGguf(gguf, config);
var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

Console.WriteLine($"Model: {config.Architecture}, {config.NumLayers} layers, {config.VocabSize} vocab");
Console.WriteLine($"Prompt: \"{prompt}\"");
Console.WriteLine();

var generator = new TextGenerator(model, tokenizer);

var options = new InferenceOptions
{
    Temperature = 0.7f,
    MaxTokens = 64,
    Logprobs = true,
    TopLogprobs = 5,
    Seed = 42,
};

// ── Streaming generation with logprobs ──
Console.Write(prompt);

InferenceTimings timings = default;
var allLogprobs = new List<TokenLogprobInfo>();

await foreach (var token in generator.GenerateStreamingTokensAsync(prompt, options))
{
    // Color-code the token by confidence
    if (token.Logprobs.HasValue)
    {
        var lp = token.Logprobs.Value;
        allLogprobs.Add(lp);
        float p = MathF.Exp(lp.Logprob);
        string color = p switch
        {
            > 0.9f => "\x1b[92m", // bright green
            > 0.7f => "\x1b[32m", // green
            > 0.5f => "\x1b[93m", // yellow
            > 0.3f => "\x1b[33m", // dark yellow/orange
            _ => "\x1b[91m",      // red
        };
        Console.Write($"{color}{token.Text}\x1b[0m");
    }
    else
    {
        Console.Write(token.Text);
    }

    if (token.Timings.HasValue)
        timings = token.Timings.Value;
}

Console.WriteLine();
Console.WriteLine();

// ── Token-by-token analysis ──
Console.WriteLine("=== Token Logprobs ===");
Console.WriteLine($"{"#",-4} {"Token",-16} {"Logprob",10} {"Prob",8}   Top alternatives");
Console.WriteLine(new string('-', 80));

double totalLogprob = 0;
float minProb = 1f;
int surpriseCount = 0;

for (int i = 0; i < allLogprobs.Count; i++)
{
    var lp = allLogprobs[i];
    float p = MathF.Exp(lp.Logprob);
    totalLogprob += lp.Logprob;
    if (p < minProb) minProb = p;
    if (p < 0.1f) surpriseCount++;

    string tokenDisplay = lp.Token.Replace("\n", "\\n").Replace("\r", "\\r");
    if (tokenDisplay.Length > 14) tokenDisplay = tokenDisplay[..14] + "..";

    string alts = "";
    if (lp.TopLogprobs is { Length: > 0 })
    {
        var altStrs = lp.TopLogprobs
            .Take(3)
            .Select(t =>
            {
                string mark = t.Token == lp.Token ? "*" : " ";
                return $"{mark}\"{t.Token.Replace("\n", "\\n")}\" {MathF.Exp(t.Logprob)*100:F1}%";
            });
        alts = string.Join("  ", altStrs);
    }

    string padded = $"\"{tokenDisplay}\"".PadRight(16);
    Console.WriteLine($"{i+1,-4} {padded} {lp.Logprob,10:F4} {p*100,7:F1}%   {alts}");
}

Console.WriteLine();
Console.WriteLine("=== Summary ===");
Console.WriteLine($"Tokens: {allLogprobs.Count}");
Console.WriteLine($"Mean logprob: {totalLogprob / allLogprobs.Count:F4}");
Console.WriteLine($"Perplexity: {MathF.Exp((float)(-totalLogprob / allLogprobs.Count)):F2}");
Console.WriteLine($"Min probability: {minProb*100:F1}%");
Console.WriteLine($"Surprise tokens (p<10%): {surpriseCount}");
Console.WriteLine();
Console.WriteLine($"[Prefill: {timings.PrefillTimeMs:F1} ms ({timings.PrefillTokensPerSec:F1} tok/s), " +
    $"Decode: {timings.DecodeTimeMs:F1} ms ({timings.DecodeTokensPerSec:F1} tok/s)]");

return 0;
