using System.Text.Json;
using DotLLM.Ascend;
using DotLLM.Core.Tensors;

internal static unsafe class KernelSmoke
{
    internal static void Run(AscendSession session)
    {
        Float32Precision(session);
        NormAndAdd(session);
        SwiGlu(session);
        foreach (bool neox in new[] { false, true })
            foreach (int rotaryDim in new[] { 8, 16 })
                Rope(session, neox, rotaryDim);
        Attention(session, 4, 1, 5, 0);
        Attention(session, 4, 2, 7, 3);
        Attention(session, 4, 4, 7, 6);
    }

    private static void Float32Precision(AscendSession session)
    {
        using var x = session.AllocateFloat32(new(1, 2));
        using var w = session.AllocateFloat32(new(2, 2));
        using var y = session.AllocateFloat32(new(1, 2));
        using var delta = session.AllocateFloat32(new(1, 2));
        session.Upload(x, new float[] { 65536, -65536 });
        session.Upload(delta, new float[] { 1, -1 });
        session.AddFloat32(x, delta, y);
        Span<float> result = stackalloc float[2];
        session.Download(y, result);
        if (result[0] != 65537 || result[1] != -65537)
            throw new Exception("Residual lost FP32 precision or silently converted to FP16.");
        session.Upload(x, new float[] { 1, 2 });
        session.Upload(w, new float[] { 1, -1, 2, 1 });
        session.LinearFloat32(x, w, y);
        session.Download(y, result);
        if (result[0] != -1 || result[1] != 4) throw new Exception("FP32 rectangular layout is wrong.");
        using var gamma = session.AllocateFloat32(new(2));
        session.Upload(gamma, new float[] { 1, 0.5f });
        session.RmsNormFloat32(x, gamma, y, 1e-5);
        session.Download(y, result);
        float expected = (float)(1 / Math.Sqrt(2.5 + 1e-5));
        if (Math.Abs(result[0] - expected) > 1e-5 || Math.Abs(result[1] - expected) > 1e-5)
            throw new Exception("FP32 RMSNorm differs from analytic reference.");
        using var halves = session.AllocateFloat16(new(1, 2));
        session.Cast(x, halves); session.Cast(halves, y);
        session.Download(y, result);
        if (result[0] != 1 || result[1] != 2) throw new Exception("FP16/FP32 cast round trip failed.");
        try { session.LinearFloat16(x, w, y); throw new Exception("Wrong dtype accepted."); }
        catch (ArgumentException) { }
        Console.WriteLine("""{"test":"fp32_precision_and_dtype","status":"PASS","scope":"residual precision, analytic linear/norm, cast, dtype rejection"}""");
    }

    private static void NormAndAdd(AscendSession session)
    {
        const int rows = 3, width = 37;
        using var input = UnmanagedTensor.Allocate(new(rows, width), DType.Float16);
        using var weights = UnmanagedTensor.Allocate(new(width), DType.Float16);
        using var output = UnmanagedTensor.Allocate(new(rows, width), DType.Float16);
        using var expected = UnmanagedTensor.Allocate(new(rows, width), DType.Float32);
        var x = HalfSpan(input);
        var w = HalfSpan(weights);
        var reference = FloatSpan(expected);
        Fill(x, 11);
        for (int i = 0; i < width; i++) w[i] = (Half)(0.5 + i / 37.0);
        // Include zero and large-magnitude rows to check epsilon/statistics behavior.
        x[..width].Clear();
        for (int i = width * 2; i < x.Length; i++) x[i] = (Half)((float)x[i] * 100);
        for (int row = 0; row < rows; row++)
        {
            double sum = 0;
            for (int i = 0; i < width; i++) sum += Math.Pow((double)x[row * width + i], 2);
            double scale = 1 / Math.Sqrt(sum / width + 1e-5);
            for (int i = 0; i < width; i++) reference[row * width + i] = (float)((double)x[row * width + i] * scale * (double)w[i]);
        }
        using var dx = session.AllocateFloat16(new(rows, width));
        using var dw = session.AllocateFloat16(new(width));
        using var dy = session.AllocateFloat16(new(rows, width));
        session.Upload(dx, x);
        session.Upload(dw, w);
        session.RmsNormFloat16(dx, dw, dy, 1e-5);
        session.Download(dy, HalfSpan(output));
        Compare("rms_norm_f16", HalfSpan(output), reference);
        session.AddFloat16(dx, dw, dy);
        session.Download(dy, HalfSpan(output));
        for (int i = 0; i < x.Length; i++) reference[i] = (float)x[i] + (float)w[i % width];
        Compare("bias_add_f16", HalfSpan(output), reference);
        session.AddFloat16(dx, dx, dy);
        session.Download(dy, HalfSpan(output));
        for (int i = 0; i < x.Length; i++) reference[i] = (float)x[i] * 2;
        Compare("residual_add_f16", HalfSpan(output), reference);
    }

    private static void SwiGlu(AscendSession session)
    {
        const int count = 129;
        using var gate = UnmanagedTensor.Allocate(new(1, count), DType.Float16);
        using var up = UnmanagedTensor.Allocate(new(1, count), DType.Float16);
        using var output = UnmanagedTensor.Allocate(new(1, count), DType.Float16);
        using var expected = UnmanagedTensor.Allocate(new(1, count), DType.Float32);
        var g = HalfSpan(gate);
        var u = HalfSpan(up);
        var reference = FloatSpan(expected);
        Fill(g, 29);
        Fill(u, 7);
        for (int i = 0; i < count; i++)
        {
            g[i] = (Half)((float)g[i] * 40);
            reference[i] = (float)((double)g[i] / (1 + Math.Exp(-(double)g[i])) * (double)u[i]);
        }
        using var dg = session.AllocateFloat16(new(1, count));
        using var du = session.AllocateFloat16(new(1, count));
        using var dy = session.AllocateFloat16(new(1, count));
        session.Upload(dg, g);
        session.Upload(du, u);
        session.SwiGluFloat16(dg, du, dy);
        session.Download(dy, HalfSpan(output));
        Compare("swiglu_f16", HalfSpan(output), reference);
    }

    private static void Rope(AscendSession session, bool neox, int rotaryDim)
    {
        const int rows = 3, heads = 2, headDim = 16;
        using var input = UnmanagedTensor.Allocate(new(rows, heads * headDim), DType.Float16);
        using var cos = UnmanagedTensor.Allocate(new(rows, rotaryDim), DType.Float16);
        using var sin = UnmanagedTensor.Allocate(new(rows, rotaryDim), DType.Float16);
        using var output = UnmanagedTensor.Allocate(new(rows, heads * headDim), DType.Float16);
        using var expected = UnmanagedTensor.Allocate(new(rows, heads * headDim), DType.Float32);
        var x = HalfSpan(input);
        var c = HalfSpan(cos);
        var s = HalfSpan(sin);
        var reference = FloatSpan(expected);
        Fill(x, 43);
        ReadOnlySpan<int> positions = [0, 7, 511];
        for (int row = 0; row < rows; row++)
            for (int pair = 0; pair < rotaryDim / 2; pair++)
            {
                double angle = positions[row] * Math.Pow(10000, -2.0 * pair / rotaryDim);
                int a = neox ? pair : 2 * pair;
                int b = neox ? pair + rotaryDim / 2 : a + 1;
                c[row * rotaryDim + a] = c[row * rotaryDim + b] = (Half)Math.Cos(angle);
                s[row * rotaryDim + a] = s[row * rotaryDim + b] = (Half)Math.Sin(angle);
            }
        for (int i = 0; i < x.Length; i++) reference[i] = (float)x[i];
        for (int row = 0; row < rows; row++)
            for (int head = 0; head < heads; head++)
                for (int pair = 0; pair < rotaryDim / 2; pair++)
                {
                    int a = neox ? pair : pair * 2;
                    int b = neox ? pair + rotaryDim / 2 : a + 1;
                    int offset = (row * heads + head) * headDim;
                    double ca = (double)c[row * rotaryDim + a], sa = (double)s[row * rotaryDim + a];
                    reference[offset + a] = (float)((double)x[offset + a] * ca - (double)x[offset + b] * sa);
                    reference[offset + b] = (float)((double)x[offset + b] * ca + (double)x[offset + a] * sa);
                }
        using var dx = session.AllocateFloat16(input.Shape);
        using var dc = session.AllocateFloat16(cos.Shape);
        using var ds = session.AllocateFloat16(sin.Shape);
        using var dy = session.AllocateFloat16(output.Shape);
        session.Upload(dx, x);
        session.Upload(dc, c);
        session.Upload(ds, s);
        session.RopeFloat16(dx, dc, ds, dy, heads, headDim, rotaryDim, neox);
        session.Download(dy, HalfSpan(output));
        Compare($"rope_f16_neox={neox}_rotary={rotaryDim}", HalfSpan(output), reference);
    }

    private static void Attention(AscendSession session, int heads, int kvHeads, int validRows, int offset)
    {
        const int headDim = 16;
        int rows = validRows - offset, capacity = validRows + 3;
        using var query = UnmanagedTensor.Allocate(new(rows, heads * headDim), DType.Float16);
        using var keys = UnmanagedTensor.Allocate(new(capacity, kvHeads * headDim), DType.Float16);
        using var values = UnmanagedTensor.Allocate(new(capacity, kvHeads * headDim), DType.Float16);
        using var output = UnmanagedTensor.Allocate(query.Shape, DType.Float16);
        using var expected = UnmanagedTensor.Allocate(query.Shape, DType.Float32);
        var q = HalfSpan(query);
        var k = HalfSpan(keys);
        var v = HalfSpan(values);
        var reference = FloatSpan(expected);
        Fill(q, 19);
        Fill(k, 31);
        Fill(v, 71);
        // Unused cache tail must not influence scores, including NaN values.
        k[(validRows * kvHeads * headDim)..].Fill(Half.NaN);
        v[(validRows * kvHeads * headDim)..].Fill(Half.NaN);
        Span<double> scoreBuffer = stackalloc double[validRows];
        for (int row = 0; row < rows; row++)
            for (int head = 0; head < heads; head++)
            {
                int kh = head / (heads / kvHeads), visible = offset + row + 1;
                Span<double> scores = scoreBuffer[..visible];
                double max = double.NegativeInfinity, total = 0;
                for (int t = 0; t < visible; t++)
                {
                    double dot = 0;
                    for (int d = 0; d < headDim; d++)
                        dot += (double)q[(row * heads + head) * headDim + d] *
                            (double)k[(t * kvHeads + kh) * headDim + d];
                    max = Math.Max(max, scores[t] = dot / Math.Sqrt(headDim));
                }
                for (int t = 0; t < visible; t++) total += scores[t] = Math.Exp(scores[t] - max);
                for (int d = 0; d < headDim; d++)
                {
                    double sum = 0;
                    for (int t = 0; t < visible; t++) sum += scores[t] / total * (double)v[(t * kvHeads + kh) * headDim + d];
                    reference[(row * heads + head) * headDim + d] = (float)sum;
                }
            }
        using var dq = session.AllocateFloat16(query.Shape);
        using var dk = session.AllocateFloat16(keys.Shape);
        using var dv = session.AllocateFloat16(values.Shape);
        using var dy = session.AllocateFloat16(query.Shape);
        session.Upload(dq, q);
        session.Upload(dk, k);
        session.Upload(dv, v);
        session.AttentionFloat16(dq, dk, dv, dy, heads, kvHeads, headDim, validRows, offset);
        session.Download(dy, HalfSpan(output));
        Compare($"attention_f16_kvheads={kvHeads}_offset={offset}", HalfSpan(output), reference);
    }

    private static Span<Half> HalfSpan(UnmanagedTensor tensor) =>
        new((void*)tensor.DataPointer, checked((int)tensor.ElementCount));
    private static Span<float> FloatSpan(UnmanagedTensor tensor) =>
        new((void*)tensor.DataPointer, checked((int)tensor.ElementCount));
    private static void Fill(Span<Half> values, uint seed)
    {
        for (int i = 0; i < values.Length; i++)
        {
            seed = unchecked(seed * 1664525 + 1013904223);
            values[i] = (Half)(((int)(seed >> 16) - 32768) / 65536.0f);
        }
    }

    private static void Compare(string test, ReadOnlySpan<Half> output, ReadOnlySpan<float> expected)
    {
        double maxError = 0;
        for (int i = 0; i < output.Length; i++)
        {
            double actual = (double)output[i], error = Math.Abs(actual - expected[i]);
            if (!double.IsFinite(actual) || error > 0.002 + Math.Abs(expected[i]) * 0.002)
                throw new InvalidOperationException($"{test}: index={i}, actual={actual}, expected={expected[i]}, error={error}");
            maxError = Math.Max(maxError, error);
        }
        Console.WriteLine(JsonSerializer.Serialize(new { test, status = "PASS", maxAbsoluteError = maxError }));
    }
}
