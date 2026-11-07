using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using DotLLM.Ascend;
using DotLLM.Cuda;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

try
{
    if (args.Length != 6) throw new ArgumentException("device model cases.json teacher.bin output-directory mode");
    string device = args[0], modelPath = args[1], mode = args[5];
    string output = args[4];
    Directory.CreateDirectory(output);
    using var gguf = GgufFile.Open(modelPath);
    var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
    var clock = Stopwatch.StartNew();
    using IModel model = device == "ascend"
        ? AscendTransformerModel.LoadFromGguf(gguf, config)
        : device == "cuda" ? CudaTransformerModel.LoadFromGguf(gguf, config)
        : throw new ArgumentException("device must be ascend or cuda");
    if (model is CudaTransformerModel cuda && Environment.GetEnvironmentVariable("COMPARE_ROPE_OVERRIDE") == "1")
    {
        typeof(CudaTransformerModel).GetProperty("DebugRopeTypeOverride",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.SetValue(cuda, 1);
        Emit(new { test = "diagnostic_override", rawKernelRopeType = 1,
            scope = "diagnostic experiment only; baseline model source is unchanged" });
    }
    Emit(new { test = "load", device, model = Path.GetFileName(modelPath), elapsedMs = clock.Elapsed.TotalMilliseconds });
    IKvCache Cache(int capacity) => model is AscendTransformerModel ascend
        ? ascend.CreateKvCache(capacity)
        : new CudaKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, capacity);
    float[] Forward(int[] tokens, int[] positions, IKvCache? cache = null)
    {
        using var value = model.Forward(tokens, positions, -1, cache);
        return Copy(value, config.VocabSize);
    }
    var cases = JsonSerializer.Deserialize<Case[]>(File.ReadAllText(args[2]))!;
    var english = cases.Where(x => x.Name.StartsWith("english_")).ToArray();
    if (mode is "all" or "warm")
    {
        foreach (var item in english)
        {
            int[] positions = Enumerable.Range(0, item.Tokens.Length).ToArray();
            clock.Restart();
            float[] full = Forward(item.Tokens, positions);
            double firstMs = clock.Elapsed.TotalMilliseconds;
            Save(output, item.Name, full);
            var times = new List<double>();
            for (int repeat = 0; repeat < 3; repeat++)
            {
                clock.Restart();
                Forward(item.Tokens, positions);
                times.Add(clock.Elapsed.TotalMilliseconds);
            }
            using var cache = Cache(item.Tokens.Length + 8);
            int split = item.Tokens.Length / 2;
            if (split > 0) Forward(item.Tokens[..split], positions[..split], cache);
            float[] appended = Forward(item.Tokens[split..], positions[split..], cache);
            Compare(appended, full, item.Name + "_split", device, 0, 0.015, 0.05);
            cache.Rollback(split);
            float[] replay = Forward(item.Tokens[split..], positions[split..], cache);
            Compare(replay, appended, item.Name + "_rollback", device, 0, 0, 0);
            Emit(new { test = "warm_prefill", device, item.Name, tokens = item.Tokens.Length,
                coldMs = firstMs, synchronousForwardMs = times, model.ComputeMemoryBytes, cache.AllocatedBytes });
        }
    }
    if (mode is "all" or "teacher")
    {
        float[] golden = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(args[3])).ToArray();
        int steps = golden.Length / config.VocabSize - 1;
        int[] tokens = Enumerable.Range(1, 8).Concat(
            Enumerable.Range(8, steps).Select(i => (i * 7919 + 17) % config.VocabSize)).ToArray();
        using var cache = Cache(tokens.Length);
        var logits = new List<float[]>();
        float[] prefill = Forward(tokens[..8], Enumerable.Range(0, 8).ToArray(), cache);
        Compare(prefill, golden.AsSpan(0, config.VocabSize), "teacher_prefill", device, 0.03, 0.015, 0.25);
        logits.Add(prefill);
        var times = new List<double>();
        for (int step = 0; step < steps; step++)
        {
            int position = step + 8;
            clock.Restart();
            float[] value = Forward([tokens[position]], [position], cache);
            times.Add(clock.Elapsed.TotalMilliseconds);
            logits.Add(value);
            Compare(value, golden.AsSpan((step + 1) * config.VocabSize, config.VocabSize),
                "teacher_" + position, device, 0.03, 0.015, 0.25);
        }
        using var file = File.Create(Path.Combine(output, "teacher.bin"));
        foreach (float[] values in logits) file.Write(MemoryMarshal.AsBytes(values.AsSpan()));
        Emit(new { test = "teacher_timing", device, steps, synchronousDecodeMs = times });
    }
    if (mode == "profile")
    {
        var item = english.Single(x => x.Tokens.Length == 32);
        int[] positions = Enumerable.Range(0, 32).ToArray();
        Forward(item.Tokens, positions); // warm device kernels and managed JIT
        using var cache = Cache(40);
        Emit(new { test = "profile_phase", phase = "prefill_begin", utc = DateTimeOffset.UtcNow });
        clock.Restart();
        Forward(item.Tokens, positions, cache);
        Emit(new { test = "profile_phase", phase = "prefill_end", elapsedMs = clock.Elapsed.TotalMilliseconds, utc = DateTimeOffset.UtcNow });
        for (int i = 0; i < 8; i++)
        {
            clock.Restart();
            Forward([(i * 7919 + 17) % config.VocabSize], [32 + i], cache);
            Emit(new { test = "profile_phase", phase = "decode", step = i, elapsedMs = clock.Elapsed.TotalMilliseconds, utc = DateTimeOffset.UtcNow });
        }
    }
    Emit(new { test = "execution_complete", device, mode, status = "PASS", scope = "execution only; numerical checks report PASS/FAIL independently" });
    return 0;
}
catch (Exception error)
{
    Console.Error.WriteLine(JsonSerializer.Serialize(new { test = "execution_failure", error = error.ToString() }));
    return 1;
}

static unsafe float[] Copy(DotLLM.Core.Tensors.ITensor value, int vocab)
    => new ReadOnlySpan<float>((float*)value.DataPointer + (value.Shape[0] - 1) * vocab, vocab).ToArray();
static void Save(string output, string name, float[] values)
    => File.WriteAllBytes(Path.Combine(output, name + ".bin"), MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
static void Emit(object item) => Console.WriteLine(JsonSerializer.Serialize(item));
static void Compare(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected, string test,
    string device, double atol, double rtol, double maxLimit)
{
    double error = 0, reference = 0, norm = 0, dot = 0, max = 0;
    bool finite = true;
    int topActual = 0, topExpected = 0;
    for (int i = 0; i < actual.Length; i++)
    {
        double a = actual[i], e = expected[i], d = a - e;
        finite &= double.IsFinite(a) && double.IsFinite(e);
        error += d * d; reference += e * e; norm += a * a; dot += a * e;
        max = Math.Max(max, Math.Abs(d));
        if (actual[i] > actual[topActual]) topActual = i;
        if (expected[i] > expected[topExpected]) topExpected = i;
    }
    double rmsError = Math.Sqrt(error / actual.Length), referenceRms = Math.Sqrt(reference / actual.Length);
    double cosine = dot / Math.Sqrt(reference * norm);
    bool pass = finite && rmsError <= atol + rtol * referenceRms && max <= maxLimit && cosine >= 0.999;
    Emit(new { test, device, status = pass ? "PASS" : "FAIL", finite, rmsError, referenceRms,
        rmsAtol = atol, rmsRtol = rtol, maxAbsoluteError = max, maxErrorLimit = maxLimit,
        cosine, actualTop1 = topActual, expectedTop1 = topExpected });
}
internal record Case(string Name, int[] Tokens);
