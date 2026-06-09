using DotLLM.Core.Lora;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Correctness tests for <see cref="MlaAttention"/>. Each test sets up a
/// synthetic MLA layer (deterministic weights) and compares the kernel's
/// forward pass against a manually-coded reference that reproduces the
/// DeepSeek-V2 attention math step by step.
/// </summary>
public sealed class MlaAttentionTests
{
    private const float Tolerance = 5e-4f;

    [Fact]
    public void Execute_SingleToken_SingleHead_MatchesReference()
    {
        const int seqLen = 1;
        const int hiddenSize = 8;
        const int numHeads = 1;
        const int qkNope = 4;
        const int qkRope = 2;
        const int vHead = 4;
        const int qLora = 6;
        const int kvLora = 5;
        const float eps = 1e-6f;
        const int maxSeq = 4;

        var fixture = BuildFixture(seqLen, hiddenSize, numHeads, qkNope, qkRope, vHead,
                                   qLora, kvLora, eps, maxSeq, seed: 42);

        float[] actual = new float[seqLen * hiddenSize];
        RunKernel(fixture, actual);

        float[] expected = new float[seqLen * hiddenSize];
        RunReference(fixture, expected);

        AssertSpansClose(expected, actual, Tolerance);
    }

    [Fact]
    public void Execute_Prefill_MultipleHeads_MatchesReference()
    {
        const int seqLen = 4;
        const int hiddenSize = 12;
        const int numHeads = 3;
        const int qkNope = 4;
        const int qkRope = 2;
        const int vHead = 4;
        const int qLora = 8;
        const int kvLora = 6;
        const float eps = 1e-6f;
        const int maxSeq = 8;

        var fixture = BuildFixture(seqLen, hiddenSize, numHeads, qkNope, qkRope, vHead,
                                   qLora, kvLora, eps, maxSeq, seed: 7);

        float[] actual = new float[seqLen * hiddenSize];
        RunKernel(fixture, actual);

        float[] expected = new float[seqLen * hiddenSize];
        RunReference(fixture, expected);

        AssertSpansClose(expected, actual, Tolerance);
    }

    [Fact]
    public void Execute_NoQFactorisation_MonolithicQProj_MatchesReference()
    {
        const int seqLen = 3;
        const int hiddenSize = 8;
        const int numHeads = 2;
        const int qkNope = 4;
        const int qkRope = 2;
        const int vHead = 4;
        const int qLora = 0;     // <-- monolithic Q path
        const int kvLora = 5;
        const float eps = 1e-6f;
        const int maxSeq = 8;

        var fixture = BuildFixture(seqLen, hiddenSize, numHeads, qkNope, qkRope, vHead,
                                   qLora, kvLora, eps, maxSeq, seed: 123);

        float[] actual = new float[seqLen * hiddenSize];
        RunKernel(fixture, actual);

        float[] expected = new float[seqLen * hiddenSize];
        RunReference(fixture, expected);

        AssertSpansClose(expected, actual, Tolerance);
    }

    [Fact]
    public void Execute_CausalMask_NoFutureLeakage()
    {
        // Use distinguishable V per position and verify the first token's
        // output cannot include contributions from later positions.
        const int seqLen = 3;
        const int hiddenSize = 6;
        const int numHeads = 1;
        const int qkNope = 2;
        const int qkRope = 2;
        const int vHead = 4;
        const int qLora = 0;
        const int kvLora = 4;
        const float eps = 1e-6f;
        const int maxSeq = 8;

        var fixture = BuildFixture(seqLen, hiddenSize, numHeads, qkNope, qkRope, vHead,
                                   qLora, kvLora, eps, maxSeq, seed: 1);

        float[] actual = new float[seqLen * hiddenSize];
        RunKernel(fixture, actual);

        // Recompute the reference but force seqLen=1 (just the first token)
        // — the two should agree on the first row.
        var firstOnly = new Fixture(fixture)
        {
            SeqLen = 1,
            Hidden = fixture.Hidden.AsSpan(0, hiddenSize).ToArray()
        };
        float[] refFirst = new float[hiddenSize];
        RunReference(firstOnly, refFirst);

        for (int d = 0; d < hiddenSize; d++)
        {
            Assert.True(MathF.Abs(actual[d] - refFirst[d]) < Tolerance,
                $"Token 0 output[{d}] = {actual[d]} differs from single-token reference {refFirst[d]}");
        }
    }

    [Fact]
    public void Execute_AttnScaleMultiplier_ChangesOutput_RemainsFinite()
    {
        // YaRN softmax correction: pass multiplier != 1.0f → attention
        // weights must redistribute → output must differ from the default
        // multiplier=1.0f case, while remaining finite. Default 1.0f is
        // bit-identical with the pre-YaRN behaviour (covered by the
        // reference-matching tests above which call Execute without the
        // parameter).
        const int seqLen = 4;
        const int hiddenSize = 12;
        const int numHeads = 3;
        const int qkNope = 4;
        const int qkRope = 2;
        const int vHead = 4;
        const int qLora = 8;
        const int kvLora = 6;
        const float eps = 1e-6f;
        const int maxSeq = 8;

        var fixture = BuildFixture(seqLen, hiddenSize, numHeads, qkNope, qkRope, vHead,
                                   qLora, kvLora, eps, maxSeq, seed: 99);

        float[] unit = new float[seqLen * hiddenSize];
        RunKernelWithScale(fixture, unit, attnScaleMultiplier: 1.0f);

        float[] yarn = new float[seqLen * hiddenSize];
        // Approximately DeepSeek-V2-Lite's mscale² ≈ 1.59.
        RunKernelWithScale(fixture, yarn, attnScaleMultiplier: 1.59f);

        foreach (float x in yarn) Assert.True(float.IsFinite(x), $"non-finite YaRN output: {x}");

        // Outputs must differ — softmax renormalises nonlinearly when scale
        // changes, so even with identical inputs the attended vectors shift.
        bool anyDifferent = false;
        for (int i = 0; i < unit.Length; i++)
        {
            if (MathF.Abs(unit[i] - yarn[i]) > 1e-5f) { anyDifferent = true; break; }
        }
        Assert.True(anyDifferent, "attnScaleMultiplier had no effect on output");
    }

    [Fact]
    public unsafe void Execute_LoraQAProj_MatchesEquivalentMergedWeight()
    {
        const int seqLen = 4;
        const int hiddenSize = 12;
        const int numHeads = 3;
        const int qkNope = 4;
        const int qkRope = 2;
        const int vHead = 4;
        const int qLora = 8;
        const int kvLora = 6;
        const float eps = 1e-6f;
        const int maxSeq = 8;

        var fixture = BuildFixture(seqLen, hiddenSize, numHeads, qkNope, qkRope, vHead,
                                   qLora, kvLora, eps, maxSeq, seed: 314);
        float[] b = Enumerable.Range(0, hiddenSize).Select(i => (i + 1) * 0.003f).ToArray();
        float[] a = Enumerable.Range(0, qLora).Select(i => (i - 3) * 0.002f).ToArray();

        using var adapter = BuildRankOneAdapter("q_a_proj", inputDim: hiddenSize, outputDim: qLora, b, a);

        float[] actual = new float[seqLen * hiddenSize];
        RunKernelWithAdapter(fixture, actual, adapter);

        var merged = new Fixture(fixture) { QAProj = (float[])fixture.QAProj.Clone() };
        AddOuterProduct(merged.QAProj, rows: qLora, cols: hiddenSize, a, b);
        float[] expected = new float[seqLen * hiddenSize];
        RunKernel(merged, expected);

        AssertSpansClose(expected, actual, Tolerance);
    }

    // ───────────────────────── helpers ─────────────────────────

    private sealed class Fixture
    {
        public int SeqLen;
        public int HiddenSize;
        public int NumHeads;
        public int QkNope;
        public int QkRope;
        public int VHead;
        public int QLora;
        public int KvLora;
        public float Eps;
        public int MaxSeq;
        public float[] Hidden = [];
        public float[] QAProj = [];
        public float[] QANorm = [];
        public float[] QBProj = [];
        public float[] QProj = [];
        public float[] KvAProj = [];
        public float[] KvANorm = [];
        public float[] KvBProj = [];
        public float[] OProj = [];
        public float[] CosTable = [];
        public float[] SinTable = [];

        public Fixture() { }

        public Fixture(Fixture other)
        {
            SeqLen = other.SeqLen;
            HiddenSize = other.HiddenSize;
            NumHeads = other.NumHeads;
            QkNope = other.QkNope;
            QkRope = other.QkRope;
            VHead = other.VHead;
            QLora = other.QLora;
            KvLora = other.KvLora;
            Eps = other.Eps;
            MaxSeq = other.MaxSeq;
            Hidden = other.Hidden;
            QAProj = other.QAProj;
            QANorm = other.QANorm;
            QBProj = other.QBProj;
            QProj = other.QProj;
            KvAProj = other.KvAProj;
            KvANorm = other.KvANorm;
            KvBProj = other.KvBProj;
            OProj = other.OProj;
            CosTable = other.CosTable;
            SinTable = other.SinTable;
        }
    }

    private static Fixture BuildFixture(
        int seqLen, int hiddenSize, int numHeads,
        int qkNope, int qkRope, int vHead,
        int qLora, int kvLora,
        float eps, int maxSeq, int seed)
    {
        var rng = new System.Random(seed);
        int qkHead = qkNope + qkRope;
        int qTotal = numHeads * qkHead;
        int kvBOut = numHeads * (qkNope + vHead);
        int oInput = numHeads * vHead;

        float[] hidden = RandomArr(rng, seqLen * hiddenSize, 0.3f);

        float[] qAProj = qLora > 0 ? RandomArr(rng, qLora * hiddenSize, 0.1f) : Array.Empty<float>();
        float[] qANorm = qLora > 0 ? FillArr(rng, qLora, 1.0f, 0.05f) : Array.Empty<float>();
        float[] qBProj = qLora > 0 ? RandomArr(rng, qTotal * qLora, 0.1f) : Array.Empty<float>();
        float[] qProj = qLora == 0 ? RandomArr(rng, qTotal * hiddenSize, 0.1f) : Array.Empty<float>();

        float[] kvAProj = RandomArr(rng, (kvLora + qkRope) * hiddenSize, 0.1f);
        float[] kvANorm = FillArr(rng, kvLora, 1.0f, 0.05f);
        float[] kvBProj = RandomArr(rng, kvBOut * kvLora, 0.1f);
        float[] oProj = RandomArr(rng, hiddenSize * oInput, 0.1f);

        (float[] cosTable, float[] sinTable) = PrecomputeRopeTables(maxSeq, qkRope, theta: 10000.0f);

        return new Fixture
        {
            SeqLen = seqLen, HiddenSize = hiddenSize, NumHeads = numHeads,
            QkNope = qkNope, QkRope = qkRope, VHead = vHead,
            QLora = qLora, KvLora = kvLora, Eps = eps, MaxSeq = maxSeq,
            Hidden = hidden,
            QAProj = qAProj, QANorm = qANorm, QBProj = qBProj, QProj = qProj,
            KvAProj = kvAProj, KvANorm = kvANorm, KvBProj = kvBProj, OProj = oProj,
            CosTable = cosTable, SinTable = sinTable,
        };
    }

    private static void RunKernel(Fixture f, Span<float> output) =>
        RunKernelWithScale(f, output, attnScaleMultiplier: 1.0f);

    private static void RunKernelWithScale(Fixture f, Span<float> output, float attnScaleMultiplier)
    {
        MlaAttention.Execute(
            hidden: f.Hidden,
            output: output,
            seqLen: f.SeqLen,
            positionOffset: 0,
            hiddenSize: f.HiddenSize,
            numHeads: f.NumHeads,
            qkNopeHeadDim: f.QkNope,
            qkRopeHeadDim: f.QkRope,
            vHeadDim: f.VHead,
            qLoraRank: f.QLora,
            kvLoraRank: f.KvLora,
            rmsNormEps: f.Eps,
            ropeCosTable: f.CosTable,
            ropeSinTable: f.SinTable,
            qAProj: f.QAProj,
            qALayernormWeight: f.QANorm,
            qBProj: f.QBProj,
            qProj: f.QProj,
            kvAProjWithMqa: f.KvAProj,
            kvALayernormWeight: f.KvANorm,
            kvBProj: f.KvBProj,
            oProj: f.OProj,
            attnScaleMultiplier: attnScaleMultiplier);
    }

    private static void RunKernelWithAdapter(Fixture f, Span<float> output, ILoraAdapter adapter)
    {
        MlaAttention.Execute(
            hidden: f.Hidden,
            output: output,
            seqLen: f.SeqLen,
            positionOffset: 0,
            hiddenSize: f.HiddenSize,
            numHeads: f.NumHeads,
            qkNopeHeadDim: f.QkNope,
            qkRopeHeadDim: f.QkRope,
            vHeadDim: f.VHead,
            qLoraRank: f.QLora,
            kvLoraRank: f.KvLora,
            rmsNormEps: f.Eps,
            ropeCosTable: f.CosTable,
            ropeSinTable: f.SinTable,
            qAProj: f.QAProj,
            qALayernormWeight: f.QANorm,
            qBProj: f.QBProj,
            qProj: f.QProj,
            kvAProjWithMqa: f.KvAProj,
            kvALayernormWeight: f.KvANorm,
            kvBProj: f.KvBProj,
            oProj: f.OProj,
            loraAdapter: adapter,
            loraLayer: 0);
    }

    private static unsafe LoraAdapter BuildRankOneAdapter(
        string projection,
        int inputDim,
        int outputDim,
        float[] b,
        float[] a)
    {
        var adapter = new LoraAdapter("mla-test", rank: 1, alpha: 1f, targetModules: [projection]);
        nint bHandle = LoraAdapter.AllocAligned(inputDim);
        nint aHandle = LoraAdapter.AllocAligned(outputDim);
        b.CopyTo(new Span<float>((void*)bHandle, inputDim));
        a.CopyTo(new Span<float>((void*)aHandle, outputDim));
        adapter.AddLayerWeights(0, projection, new LoraLayerWeights(
            AHandle: aHandle,
            BHandle: bHandle,
            InputDim: inputDim,
            OutputDim: outputDim));
        return adapter;
    }

    private static void AddOuterProduct(float[] matrix, int rows, int cols, float[] a, float[] b)
    {
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                matrix[r * cols + c] += a[r] * b[c];
    }

    /// <summary>
    /// Reference implementation: step-by-step translation of the DeepSeek-V2
    /// forward pass. Intentionally verbose and non-performant so it reads
    /// directly against the HF modeling_deepseek_v2.py source.
    /// </summary>
    private static void RunReference(Fixture f, Span<float> output)
    {
        int qkHead = f.QkNope + f.QkRope;
        int qTotal = f.NumHeads * qkHead;
        int kvBOut = f.NumHeads * (f.QkNope + f.VHead);
        int oInput = f.NumHeads * f.VHead;
        int halfRope = f.QkRope / 2;
        float scale = 1.0f / MathF.Sqrt(qkHead);

        // 1. Q
        float[] q = new float[f.SeqLen * qTotal];
        for (int t = 0; t < f.SeqLen; t++)
        {
            var hRow = new ReadOnlySpan<float>(f.Hidden, t * f.HiddenSize, f.HiddenSize);
            var qRow = q.AsSpan(t * qTotal, qTotal);
            if (f.QLora > 0)
            {
                float[] latent = new float[f.QLora];
                MatVec(f.QAProj, hRow, latent, f.QLora, f.HiddenSize);
                float[] latentNorm = new float[f.QLora];
                RmsNorm(latent, f.QANorm, f.Eps, latentNorm);
                MatVec(f.QBProj, latentNorm, qRow, qTotal, f.QLora);
            }
            else
            {
                MatVec(f.QProj, hRow, qRow, qTotal, f.HiddenSize);
            }
        }

        // 2. KV — compressed, split, layernorm, expand, per-head split
        float[] kNope = new float[f.SeqLen * f.NumHeads * f.QkNope];
        float[] kPe = new float[f.SeqLen * f.QkRope];
        float[] v = new float[f.SeqLen * f.NumHeads * f.VHead];
        int compressedDim = f.KvLora + f.QkRope;
        for (int t = 0; t < f.SeqLen; t++)
        {
            var hRow = new ReadOnlySpan<float>(f.Hidden, t * f.HiddenSize, f.HiddenSize);
            float[] compressed = new float[compressedDim];
            MatVec(f.KvAProj, hRow, compressed, compressedDim, f.HiddenSize);

            float[] latent = compressed[..f.KvLora];
            float[] kPeVec = compressed[f.KvLora..];

            float[] latentNorm = new float[f.KvLora];
            RmsNorm(latent, f.KvANorm, f.Eps, latentNorm);

            float[] expanded = new float[kvBOut];
            MatVec(f.KvBProj, latentNorm, expanded, kvBOut, f.KvLora);

            int perHead = f.QkNope + f.VHead;
            for (int h = 0; h < f.NumHeads; h++)
            {
                for (int d = 0; d < f.QkNope; d++)
                    kNope[t * f.NumHeads * f.QkNope + h * f.QkNope + d] = expanded[h * perHead + d];
                for (int d = 0; d < f.VHead; d++)
                    v[t * f.NumHeads * f.VHead + h * f.VHead + d] = expanded[h * perHead + f.QkNope + d];
            }
            for (int d = 0; d < f.QkRope; d++)
                kPe[t * f.QkRope + d] = kPeVec[d];
        }

        // 3. RoPE (Norm-pair) on q_pe per head and shared k_pe
        for (int t = 0; t < f.SeqLen; t++)
        {
            for (int h = 0; h < f.NumHeads; h++)
            {
                // q[t, h, qkNope..qkHead)
                int off = t * qTotal + h * qkHead + f.QkNope;
                for (int i = 0; i < halfRope; i++)
                {
                    float a = q[off + 2 * i];
                    float b = q[off + 2 * i + 1];
                    float c = f.CosTable[t * halfRope + i];
                    float s = f.SinTable[t * halfRope + i];
                    q[off + 2 * i] = a * c - b * s;
                    q[off + 2 * i + 1] = b * c + a * s;
                }
            }
            // shared k_pe
            int kpOff = t * f.QkRope;
            for (int i = 0; i < halfRope; i++)
            {
                float a = kPe[kpOff + 2 * i];
                float b = kPe[kpOff + 2 * i + 1];
                float c = f.CosTable[t * halfRope + i];
                float s = f.SinTable[t * halfRope + i];
                kPe[kpOff + 2 * i] = a * c - b * s;
                kPe[kpOff + 2 * i + 1] = b * c + a * s;
            }
        }

        // 4. Attention per head, causal mask, softmax, weighted V
        float[] attn = new float[f.SeqLen * f.NumHeads * f.VHead];
        for (int h = 0; h < f.NumHeads; h++)
        {
            for (int tq = 0; tq < f.SeqLen; tq++)
            {
                float[] scores = new float[f.SeqLen];
                for (int tk = 0; tk < f.SeqLen; tk++)
                {
                    if (tk > tq) { scores[tk] = float.NegativeInfinity; continue; }
                    float dot = 0f;
                    // nope
                    for (int d = 0; d < f.QkNope; d++)
                        dot += q[tq * qTotal + h * qkHead + d]
                             * kNope[tk * f.NumHeads * f.QkNope + h * f.QkNope + d];
                    // rope (kPe shared)
                    for (int d = 0; d < f.QkRope; d++)
                        dot += q[tq * qTotal + h * qkHead + f.QkNope + d]
                             * kPe[tk * f.QkRope + d];
                    scores[tk] = dot * scale;
                }
                // Softmax
                float mx = float.NegativeInfinity;
                for (int i = 0; i < scores.Length; i++) if (scores[i] > mx) mx = scores[i];
                float sum = 0f;
                for (int i = 0; i < scores.Length; i++)
                {
                    scores[i] = MathF.Exp(scores[i] - mx);
                    sum += scores[i];
                }
                if (sum > 0f) for (int i = 0; i < scores.Length; i++) scores[i] /= sum;

                // Weighted V
                for (int d = 0; d < f.VHead; d++)
                {
                    float s = 0f;
                    for (int tk = 0; tk <= tq; tk++)
                        s += scores[tk] * v[tk * f.NumHeads * f.VHead + h * f.VHead + d];
                    attn[tq * f.NumHeads * f.VHead + h * f.VHead + d] = s;
                }
            }
        }

        // 5. o_proj
        for (int t = 0; t < f.SeqLen; t++)
        {
            var attnRow = new ReadOnlySpan<float>(attn, t * oInput, oInput);
            var outRow = output.Slice(t * f.HiddenSize, f.HiddenSize);
            MatVec(f.OProj, attnRow, outRow, f.HiddenSize, oInput);
        }
    }

    private static void MatVec(
        ReadOnlySpan<float> w, ReadOnlySpan<float> x, Span<float> y, int m, int k)
    {
        for (int i = 0; i < m; i++)
        {
            float s = 0f;
            for (int j = 0; j < k; j++)
                s += w[i * k + j] * x[j];
            y[i] = s;
        }
    }

    private static void RmsNorm(
        ReadOnlySpan<float> input, ReadOnlySpan<float> weight, float eps, Span<float> output)
    {
        float sum = 0f;
        for (int i = 0; i < input.Length; i++) sum += input[i] * input[i];
        float rms = MathF.Sqrt(sum / input.Length + eps);
        float inv = 1f / rms;
        for (int i = 0; i < input.Length; i++) output[i] = input[i] * inv * weight[i];
    }

    private static (float[] cos, float[] sin) PrecomputeRopeTables(int maxSeq, int dim, float theta)
    {
        int half = dim / 2;
        float[] cos = new float[maxSeq * half];
        float[] sin = new float[maxSeq * half];
        for (int pos = 0; pos < maxSeq; pos++)
        {
            for (int i = 0; i < half; i++)
            {
                float freq = 1.0f / MathF.Pow(theta, 2.0f * i / dim);
                float angle = pos * freq;
                cos[pos * half + i] = MathF.Cos(angle);
                sin[pos * half + i] = MathF.Sin(angle);
            }
        }
        return (cos, sin);
    }

    private static float[] RandomArr(System.Random rng, int n, float scale)
    {
        float[] arr = new float[n];
        for (int i = 0; i < n; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return arr;
    }

    private static float[] FillArr(System.Random rng, int n, float center, float jitter)
    {
        float[] arr = new float[n];
        for (int i = 0; i < n; i++)
            arr[i] = center + (float)((rng.NextDouble() * 2.0 - 1.0) * jitter);
        return arr;
    }

    private static void AssertSpansClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float tol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            Assert.True(diff <= tol,
                $"index {i}: expected={expected[i]} actual={actual[i]} diff={diff} (tol={tol})");
        }
    }
}
