using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using DotLLM.Ascend;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;

internal static class TextSmoke
{
    internal record Case(string Name, int[] Tokens);
    internal static void Emit(string modelPath, string output)
    {
        using var gguf = GgufFile.Open(modelPath);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        string[] texts = [
            "The purpose of a key value cache is to reuse the attention states of earlier tokens. ",
            "推理框架需要正确管理缓存、位置编码和设备内存，每个请求应保持独立状态。",
            "def fibonacci(n):\n    a, b = 0, 1\n    for i in range(n):\n        a, b = b, a + b\n    return a\n"];
        var cases = new List<Case>();
        string[] names = ["english", "chinese", "code"];
        for (int language = 0; language < texts.Length; language++)
        {
            int[] tokens = tokenizer.Encode(string.Concat(Enumerable.Repeat(texts[language], 100)));
            foreach (int length in new[] { 1, 32, 128, 512 })
                cases.Add(new($"{names[language]}_{length}", tokens[..length]));
        }
        cases.Add(new("eos", [tokenizer.EosTokenId]));
        File.WriteAllText(output, JsonSerializer.Serialize(cases));
    }
    internal static void Run(string modelPath, string casesPath, string goldenPath)
    {
        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = AscendTransformerModel.LoadFromGguf(gguf, config);
        var cases = JsonSerializer.Deserialize<Case[]>(File.ReadAllText(casesPath))!;
        float[] golden = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(goldenPath)).ToArray();
        if (golden.Length != cases.Length * config.VocabSize) throw new Exception("Invalid text reference size.");
        for (int i = 0; i < cases.Length; i++)
        {
            var test = cases[i];
            int[] positions = Enumerable.Range(0, test.Tokens.Length).ToArray();
            var watch = Stopwatch.StartNew();
            float[] full = ModelSmoke.Copy(model.Forward(test.Tokens, positions, -1));
            double coldMs = watch.Elapsed.TotalMilliseconds;
            ModelSmoke.Compare(full, golden.AsSpan(i * config.VocabSize, config.VocabSize),
                test.Name, 0.015, 0.25, rmsAtol: 0.03);
            using var cache = model.CreateKvCache(test.Tokens.Length);
            int split = test.Tokens.Length / 2;
            if (split > 0) model.Forward(test.Tokens.AsSpan(0, split), positions.AsSpan(0, split), -1, cache);
            float[] appended = ModelSmoke.Copy(model.Forward(test.Tokens.AsSpan(split), positions.AsSpan(split), -1, cache));
            ModelSmoke.Compare(appended, full, test.Name + "_split", 0.015, 0.05);
            if (cache.CurrentLength != test.Tokens.Length) throw new Exception("Cache boundary commit failed.");
            try { model.Forward([1], [test.Tokens.Length], -1, cache); throw new Exception("Capacity rejection failed."); }
            catch (ArgumentOutOfRangeException) { }
            cache.Rollback(split);
            ModelSmoke.Compare(ModelSmoke.Copy(model.Forward(test.Tokens.AsSpan(split),
                positions.AsSpan(split), -1, cache)), appended, test.Name + "_rollback", 0, 0);
            var measurements = new List<double>();
            for (int repeat = 0; repeat < 3; repeat++)
            {
                watch.Restart();
                using var ignored = model.Forward(test.Tokens, positions, -1);
                measurements.Add(watch.Elapsed.TotalMilliseconds);
            }
            Console.WriteLine(JsonSerializer.Serialize(new { test = "text_timing", test.Name,
                tokens = test.Tokens.Length, coldMs, synchronousForwardMs = measurements,
                model.DeviceWeightBytes, model.ComputeMemoryBytes, cache.AllocatedBytes }));
        }
        Console.WriteLine(JsonSerializer.Serialize(new { test = "text_suite", status = "PASS",
            model = Path.GetFileName(modelPath), cases = cases.Length }));
    }
}
