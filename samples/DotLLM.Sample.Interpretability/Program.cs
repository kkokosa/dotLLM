using DotLLM.Core.Diagnostics;
using DotLLM.Diagnostics;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Bpe;

if (args.Length < 1)
{
    Console.Error.WriteLine("Usage: dotLLM.Sample.Interpretability <model.gguf> [prompt]");
    Console.Error.WriteLine("  model.gguf  Path to a GGUF model file");
    Console.Error.WriteLine("  prompt      Text prompt (default: \"The capital of France is\")");
    Console.Error.WriteLine();
    Console.Error.WriteLine("Tip: a small model such as QuantFactory/SmolLM-135M-GGUF Q8_0 works well.");
    return 1;
}

string modelPath = args[0];
string prompt = args.Length > 1 ? string.Join(' ', args.Skip(1)) : "The capital of France is";

Console.WriteLine($"Loading model: {modelPath}");
using var gguf = GgufFile.Open(modelPath);
var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
using var model = TransformerModel.LoadFromGguf(gguf, config);
var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

Console.WriteLine($"Model: {config.Architecture}, {config.NumLayers} layers, {config.HiddenSize} hidden, {config.VocabSize} vocab");
Console.WriteLine($"Prompt: \"{prompt}\"");

int[] promptTokens = tokenizer.Encode(prompt);
int[] positions = new int[promptTokens.Length];
for (int i = 0; i < positions.Length; i++) positions[i] = i;

Console.WriteLine($"Tokens ({promptTokens.Length}): [{string.Join(", ", promptTokens)}]");
Console.WriteLine();

// Register the logit lens hook over every layer at the final prompt position.
// Capturing only one position keeps memory bounded and is the standard logit-lens setup.
int finalPosition = positions[^1];
var hookConfig = new LogitLensConfig
{
    Layers = LogitLensLayerSelector.AllLayers,
    TopK = 5,
    StoreFullProbabilities = true,
    TokenPositions = [finalPosition],
};

var lens = new LogitLensHook(model, hookConfig);
model.Hooks ??= new HookRegistry();
model.Hooks.Register(lens);

// Single forward pass over the prompt — no KV cache needed for one-shot logit lens.
using var logitsTensor = model.Forward(promptTokens, positions, deviceId: -1);

// Identify the actual top-1 token the full model predicts at the final position.
int vocab = config.VocabSize;
unsafe
{
    var finalLogits = new ReadOnlySpan<float>(
        (float*)logitsTensor.DataPointer + (promptTokens.Length - 1) * vocab, vocab);

    int finalTopToken = 0;
    float finalTopLogit = float.NegativeInfinity;
    for (int i = 0; i < finalLogits.Length; i++)
    {
        if (finalLogits[i] > finalTopLogit)
        {
            finalTopLogit = finalLogits[i];
            finalTopToken = i;
        }
    }

    var results = lens.GetResults();
    string finalTokenText = TryDecode(tokenizer, finalTopToken);
    Console.WriteLine($"Final-layer top-1 token: id={finalTopToken} \"{Escape(finalTokenText)}\"");
    Console.WriteLine();

    // Per-layer table: layer | entropy | top-5 (token-id "text" prob)
    Console.WriteLine($"{"Layer",-5} {"Entropy",-8}  Top-{hookConfig.TopK}");
    Console.WriteLine(new string('-', 78));
    foreach (var r in results)
    {
        var topStrings = new string[r.TopKTokens.Length];
        for (int i = 0; i < r.TopKTokens.Length; i++)
        {
            string text = TryDecode(tokenizer, r.TopKTokens[i]);
            topStrings[i] = $"{r.TopKTokens[i]}:\"{Escape(text)}\"={r.TopKProbabilities[i]:F3}";
        }
        Console.WriteLine($"{r.LayerIndex,-5} {r.Entropy,-8:F3}  {string.Join("  ", topStrings)}");
    }

    Console.WriteLine();
    int? convergenceLayer = LogitLensAnalysis.ConvergenceLayer(results, finalTopToken, finalPosition);
    Console.WriteLine(convergenceLayer is null
        ? $"Final token (id={finalTopToken}) was never the top-1 at any analysed layer."
        : $"Final token (id={finalTopToken}) first becomes top-1 at layer {convergenceLayer.Value}.");

    var trajectory = LogitLensAnalysis.ConfidenceAcrossLayers(results, finalTopToken, finalPosition);
    Console.WriteLine();
    Console.WriteLine($"Confidence trajectory for token {finalTopToken}:");
    foreach (var (layer, p) in trajectory)
        Console.WriteLine($"  layer {layer,2}: p = {p:F4}");
}

return 0;

static string TryDecode(ITokenizer tokenizer, int tokenId)
{
    try
    {
        return tokenizer.Decode([tokenId]);
    }
    catch
    {
        return $"<{tokenId}>";
    }
}

static string Escape(string s)
{
    return s.Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");
}
