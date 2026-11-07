using System.Diagnostics;
using System.Text.Json;
using DotLLM.Ascend;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

internal static class ModelSmoke
{
    internal static unsafe void Run(string path, string? referencePath, string? mathReferencePath = null)
    {
        using var file = GgufFile.Open(path);
        var c = GgufModelConfigExtractor.Extract(file.Metadata);
        var watch = Stopwatch.StartNew();
        using var model = AscendTransformerModel.LoadFromGguf(file, c);
        double loadMs = watch.Elapsed.TotalMilliseconds;
        using var referenceFile = GgufFile.Open(referencePath ?? path);
        using var cpu = TransformerModel.LoadFromGguf(referenceFile, c);
        int[] tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        int[] positions = [0, 1, 2, 3, 4, 5, 6, 7];
        watch.Restart();
        float[] full = Copy(model.Forward(tokens, positions, -1));
        double prefillMs = watch.Elapsed.TotalMilliseconds;
        using var expected = cpu.Forward(tokens, positions, -1);
        var expectedLast = new ReadOnlySpan<float>((float*)expected.DataPointer + (tokens.Length - 1) * c.VocabSize, c.VocabSize);
        float[]? mathReference = mathReferencePath is null ? null :
            System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(mathReferencePath)).ToArray();
        if (mathReference is not null)
            Compare(expectedLast, mathReference.AsSpan(0, c.VocabSize), "cpu_vs_numpy_prefill", 0.0001, 0.005, enforce: false);
        // Full-model tolerance is distinct from the exact kernel tolerance. The absolute
        // RMS term avoids a near-zero reference denominator; raw errors remain reported.
        Compare(full, mathReference is null ? expectedLast : mathReference.AsSpan(0, c.VocabSize),
            mathReference is null ? "cpu_reference" : "numpy_reference", 0.015, 0.25, rmsAtol: 0.03);

        using var cache = model.CreateKvCache(16);
        model.Forward(tokens.AsSpan(0, 4), positions.AsSpan(0, 4), -1, cache);
        float[] split = Copy(model.Forward(tokens.AsSpan(4), positions.AsSpan(4), -1, cache));
        Compare(split, full, "split_prefill", 0.01, 0.05);
        if (cache.CurrentLength != 8) throw new Exception("KV commit length incorrect.");
        cache.Rollback(4);
        watch.Restart();
        float[] decode = [];
        for (int i = 4; i < 8; i++)
            decode = Copy(model.Forward(tokens.AsSpan(i, 1), positions.AsSpan(i, 1), -1, cache));
        double decodeMs = watch.Elapsed.TotalMilliseconds;
        Compare(decode, full, "decode_after_rollback", 0.015, 0.05);
        cache.Rollback(0);
        Compare(Copy(model.Forward(tokens, positions, -1, cache)), full, "rollback_to_zero", 0.01, 0.05);

        using var other = model.CreateKvCache(16);
        model.Forward(tokens.AsSpan(0, 2), positions.AsSpan(0, 2), -1, other);
        if (cache.CurrentLength != 8 || other.CurrentLength != 2) throw new Exception("Independent cache lengths leaked.");
        Expect<ArgumentException>(() => model.Forward([1], [0], -1, cache));
        Expect<ArgumentException>(() => model.Forward([c.VocabSize], [8], -1, cache));
        Expect<ArgumentException>(() => model.Forward([1], [8], 99, cache));
        if (cache.CurrentLength != 8) throw new Exception("Rejected forward modified visible cache.");
        Expect<ArgumentOutOfRangeException>(() => cache.Rollback(9));
        using var small = model.CreateKvCache(1);
        Expect<ArgumentOutOfRangeException>(() => model.Forward([1, 2], [0, 1], -1, small));
        using var otherModel = AscendTransformerModel.LoadFromGguf(file, c);
        Expect<ArgumentException>(() => otherModel.Forward([1], [8], -1, cache));
        other.Dispose();
        Expect<ObjectDisposedException>(() => model.Forward([1], [2], -1, other));
        int teacherSteps = Math.Min(32, c.MaxSequenceLength - 8);
        int[] teacherTokens = new int[8 + teacherSteps];
        int[] teacherPositions = Enumerable.Range(0, teacherTokens.Length).ToArray();
        tokens.CopyTo(teacherTokens, 0);
        for (int i = 8; i < teacherTokens.Length; i++)
            teacherTokens[i] = (i * 7919 + 17) % c.VocabSize;
        using var teacherCache = model.CreateKvCache(teacherTokens.Length);
        model.Forward(tokens, positions, -1, teacherCache);
        for (int i = 8; i < teacherTokens.Length; i++)
        {
            float[] actual = Copy(model.Forward(teacherTokens.AsSpan(i, 1), teacherPositions.AsSpan(i, 1), -1, teacherCache));
            float[] last = [];
            if (mathReference is null)
            {
                using var teacherExpected = cpu.Forward(teacherTokens.AsSpan(0, i + 1), teacherPositions.AsSpan(0, i + 1), -1);
                last = new ReadOnlySpan<float>((float*)teacherExpected.DataPointer + i * c.VocabSize, c.VocabSize).ToArray();
            }
            Compare(actual, mathReference is null ? last : mathReference.AsSpan((i - 7) * c.VocabSize, c.VocabSize),
                $"teacher_forced_{i}", 0.015, 0.25, rmsAtol: 0.03);
        }
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            test = "model_and_kv", status = "PASS", model = Path.GetFileName(path),
            reference = mathReference is not null ? "independent NumPy FP32 math" :
                referencePath is null ? "existing CPU backend (may quantize activations)" : "CPU FP32 weights/activations (approximate attention)",
            loadMs, prefillMs, decodeMs, decodeTokens = 4, teacherSteps,
            prefillTokensPerSecond = 8000 / prefillMs, decodeTokensPerSecond = 4000 / decodeMs,
            model.DeviceWeightBytes, model.ComputeMemoryBytes, cache.AllocatedBytes
        }));
    }

    internal static unsafe float[] Copy(ITensor tensor)
    {
        using (tensor)
            return new ReadOnlySpan<float>((void*)tensor.DataPointer, checked((int)tensor.ElementCount)).ToArray();
    }

    internal static void Compare(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected,
        string name, double nrmseLimit, double maxErrorLimit, bool enforce = true, double rmsAtol = 0)
    {
        if (actual.Length != expected.Length || actual.IsEmpty)
            throw new Exception("Reference shape mismatch.");
        double squared = 0, reference = 0, max = 0, dot = 0, actualSquared = 0, absoluteSum = 0;
        double actualMean = 0, expectedMean = 0;
        for (int i = 0; i < actual.Length; i++) { actualMean += actual[i]; expectedMean += expected[i]; }
        actualMean /= actual.Length; expectedMean /= expected.Length;
        double centeredSquared = 0, centeredReference = 0, centeredMax = 0;
        int aTop = 0, eTop = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            if (!float.IsFinite(actual[i]) || !float.IsFinite(expected[i])) throw new Exception("Non-finite logits or reference.");
            double error = actual[i] - expected[i];
            double centeredError = error - (actualMean - expectedMean);
            centeredSquared += centeredError * centeredError;
            centeredReference += (expected[i] - expectedMean) * (expected[i] - expectedMean);
            centeredMax = Math.Max(centeredMax, Math.Abs(centeredError));
            absoluteSum += Math.Abs(error);
            squared += error * error; reference += expected[i] * (double)expected[i];
            actualSquared += actual[i] * (double)actual[i]; dot += actual[i] * (double)expected[i];
            max = Math.Max(max, Math.Abs(error));
            if (actual[i] > actual[aTop]) aTop = i;
            if (expected[i] > expected[eTop]) eTop = i;
        }
        double nrmse = Math.Sqrt(squared / Math.Max(reference, 1e-30));
        double rmsError = Math.Sqrt(squared / actual.Length);
        double referenceRms = Math.Sqrt(reference / actual.Length);
        double rmsLimit = rmsAtol + nrmseLimit * referenceRms;
        double cosine = dot / Math.Sqrt(Math.Max(reference * actualSquared, 1e-30));
        int[] actualTop5 = actual.ToArray().Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(5).Select(x => x.i).ToArray();
        int[] expectedTop5 = expected.ToArray().Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(5).Select(x => x.i).ToArray();
        double top5Overlap = actualTop5.Intersect(expectedTop5).Count() / 5.0;
        Console.WriteLine(JsonSerializer.Serialize(new { test = name, nrmse, maxAbsoluteError = max, cosine,
            meanAbsoluteError = absoluteSum / actual.Length, top5Overlap,
            meanShift = actualMean - expectedMean, centeredMaxAbsoluteError = centeredMax,
            centeredNrmse = Math.Sqrt(centeredSquared / Math.Max(centeredReference, 1e-30)),
            actualTop1 = aTop, expectedTop1 = eTop, nrmseLimit, maxErrorLimit, rmsError, referenceRms, rmsAtol, rmsLimit, enforced = enforce }));
        if (enforce && (rmsError > rmsLimit || max > maxErrorLimit || cosine < 0.999))
            throw new Exception($"{name} logits differ: NRMSE={nrmse}, max={max}, cosine={cosine}");
    }

    private static void Expect<T>(Action action) where T : Exception
    {
        try { action(); } catch (T) { return; }
        throw new Exception($"Expected {typeof(T).Name}.");
    }
}
