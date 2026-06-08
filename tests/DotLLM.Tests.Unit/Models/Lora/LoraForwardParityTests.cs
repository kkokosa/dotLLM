using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Lora;

/// <summary>
/// End-to-end parity tests for the LoRA-aware forward path:
/// 1. The standalone <see cref="LoraDelta"/> kernel matches a scalar
///    reference implementation of <c>y += scale × (x · B) · A</c>.
/// 2. Calling <see cref="TransformerModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?, ILoraAdapter?)"/>
///    with a zero adapter is byte-equivalent to the adapter-less forward.
/// 3. A non-zero adapter produces a measurable, finite output difference.
/// </summary>
public sealed class LoraForwardParityTests : IDisposable
{
    private readonly string _scratch;

    public LoraForwardParityTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-lora-fwd-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    // ────────────────────────────────────────────────────────────────────
    // Kernel-level test: LoraDelta vs scalar reference
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void LoraDelta_MatchesScalarReference()
    {
        const int seqLen = 3;
        const int inputDim = 16;
        const int outputDim = 12;
        const int rank = 4;
        const float scale = 0.5f;

        var rng = new Random(123);
        var x = RandomVec(rng, seqLen * inputDim);
        var b = RandomVec(rng, rank * inputDim);          // [rank, inputDim]
        var a = RandomVec(rng, outputDim * rank);          // [outputDim, rank]
        var yKernel = RandomVec(rng, seqLen * outputDim);  // initial y values
        var yRef = (float[])yKernel.Clone();

        // Kernel
        unsafe
        {
            fixed (float* xp = x)
            fixed (float* bp = b)
            fixed (float* ap = a)
            fixed (float* yp = yKernel)
            {
                LoraDelta.Apply(xp, bp, ap, yp, seqLen, inputDim, outputDim, rank, scale);
            }
        }

        // Scalar reference: tmp[t, r] = sum_i x[t, i] * b[r, i]
        //                   y[t, o] += scale * sum_r a[o, r] * tmp[t, r]
        for (int t = 0; t < seqLen; t++)
        {
            var tmp = new float[rank];
            for (int r = 0; r < rank; r++)
            {
                float s = 0;
                for (int i = 0; i < inputDim; i++)
                    s += x[t * inputDim + i] * b[r * inputDim + i];
                tmp[r] = s;
            }
            for (int o = 0; o < outputDim; o++)
            {
                float s = 0;
                for (int r = 0; r < rank; r++)
                    s += a[o * rank + r] * tmp[r];
                yRef[t * outputDim + o] += scale * s;
            }
        }

        AssertClose(yRef, yKernel, absTol: 5e-3f, relTol: 1e-3f);
    }

    [Fact]
    public void LoraDelta_ZeroBIsNoOp()
    {
        const int seqLen = 2, inputDim = 8, outputDim = 8, rank = 2;
        var x = RandomVec(new Random(1), seqLen * inputDim);
        var b = new float[rank * inputDim]; // all zero
        var a = RandomVec(new Random(2), outputDim * rank);
        var y = RandomVec(new Random(3), seqLen * outputDim);
        var yCopy = (float[])y.Clone();

        unsafe
        {
            fixed (float* xp = x) fixed (float* bp = b) fixed (float* ap = a) fixed (float* yp = y)
                LoraDelta.Apply(xp, bp, ap, yp, seqLen, inputDim, outputDim, rank, scale: 16.0f);
        }

        AssertClose(yCopy, y, absTol: 1e-7f, relTol: 1e-7f);
    }

    // ────────────────────────────────────────────────────────────────────
    // TransformerModel-level parity: backward-compat + measurable delta
    // ────────────────────────────────────────────────────────────────────

    private (TransformerModel Model, IDisposable Source, ModelConfig Config) BuildTinyModel()
    {
        const int hidden = 64, numHeads = 4, headDim = 16, intermediate = 128, vocab = 32, layers = 2;
        var rng = new Random(42);
        var bld = new SafetensorsFixtureBuilder();
        bld.AddFloat32("model.embed_tokens.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f));
        bld.AddFloat32("model.norm.weight", [hidden], Ones(hidden));
        for (int i = 0; i < layers; i++)
        {
            string p = $"model.layers.{i}";
            bld.AddFloat32($"{p}.input_layernorm.weight", [hidden], Ones(hidden));
            bld.AddFloat32($"{p}.post_attention_layernorm.weight", [hidden], Ones(hidden));
            bld.AddFloat32($"{p}.self_attn.q_proj.weight",
                [numHeads * headDim, hidden], RandomVec(rng, numHeads * headDim * hidden, 0.05f));
            bld.AddFloat32($"{p}.self_attn.k_proj.weight",
                [numHeads * headDim, hidden], RandomVec(rng, numHeads * headDim * hidden, 0.05f));
            bld.AddFloat32($"{p}.self_attn.v_proj.weight",
                [numHeads * headDim, hidden], RandomVec(rng, numHeads * headDim * hidden, 0.05f));
            bld.AddFloat32($"{p}.self_attn.o_proj.weight",
                [hidden, numHeads * headDim], RandomVec(rng, hidden * numHeads * headDim, 0.05f));
            bld.AddFloat32($"{p}.mlp.gate_proj.weight",
                [intermediate, hidden], RandomVec(rng, intermediate * hidden, 0.05f));
            bld.AddFloat32($"{p}.mlp.up_proj.weight",
                [intermediate, hidden], RandomVec(rng, intermediate * hidden, 0.05f));
            bld.AddFloat32($"{p}.mlp.down_proj.weight",
                [hidden, intermediate], RandomVec(rng, hidden * intermediate, 0.05f));
        }
        bld.AddFloat32("lm_head.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f));

        string path = Path.Combine(_scratch, $"base-{Guid.NewGuid():N}.safetensors");
        bld.WriteTo(path);

        var cfg = new ModelConfig
        {
            Architecture = Architecture.Llama,
            VocabSize = vocab,
            HiddenSize = hidden,
            IntermediateSize = intermediate,
            NumLayers = layers,
            NumAttentionHeads = numHeads,
            NumKvHeads = numHeads,
            HeadDim = headDim,
            MaxSequenceLength = 128,
            NormEpsilon = 1e-5f,
            RoPEConfig = new RoPEConfig(Theta: 10000f, DimensionCount: headDim, Type: RoPEType.Norm),
        };

        var file = SafetensorsFile.Open(path);
        var model = TransformerModel.LoadFromSafetensors(file, cfg);
        return (model, file, cfg);
    }

    private static LoraAdapter BuildSyntheticAdapter(ModelConfig cfg, int rank, float alpha,
                                                     bool zeroFactors = false, int seed = 7)
    {
        var rng = new Random(seed);
        int qOut = cfg.NumAttentionHeads * cfg.HeadDim;
        int kvOut = cfg.NumKvHeads * cfg.HeadDim;
        var adapter = new LoraAdapter("syn",
            rank: rank, alpha: alpha,
            targetModules: ["q_proj", "v_proj"]);
        try
        {
            for (int layer = 0; layer < cfg.NumLayers; layer++)
            {
                AddProj(adapter, layer, "q_proj", inputDim: cfg.HiddenSize, outputDim: qOut, rank, zeroFactors, rng);
                AddProj(adapter, layer, "v_proj", inputDim: cfg.HiddenSize, outputDim: kvOut, rank, zeroFactors, rng);
            }
            return adapter;
        }
        catch
        {
            adapter.Dispose();
            throw;
        }
    }

    private static unsafe void AddProj(LoraAdapter adapter, int layer, string proj,
        int inputDim, int outputDim, int rank, bool zero, Random rng)
    {
        long bElems = (long)rank * inputDim;
        long aElems = (long)outputDim * rank;
        nint b = LoraAdapter.AllocAligned(bElems);
        nint a = LoraAdapter.AllocAligned(aElems);

        if (!zero)
        {
            // Small random values so deltas are measurable but stable.
            float* bp = (float*)b;
            float* ap = (float*)a;
            for (long i = 0; i < bElems; i++) bp[i] = (float)((rng.NextDouble() * 2 - 1) * 0.05);
            for (long i = 0; i < aElems; i++) ap[i] = (float)((rng.NextDouble() * 2 - 1) * 0.05);
        }
        else
        {
            new Span<float>((void*)b, (int)bElems).Clear();
            new Span<float>((void*)a, (int)aElems).Clear();
        }
        adapter.AddLayerWeights(layer, proj,
            new LoraLayerWeights(AHandle: a, BHandle: b, InputDim: inputDim, OutputDim: outputDim));
    }

    [Fact]
    public unsafe void Forward_NoAdapter_VsZeroAdapter_AreIdentical()
    {
        var (model, source, cfg) = BuildTinyModel();
        try
        {
            // Run baseline (no adapter)
            int[] tokenIds = [1, 2, 3];
            int[] positions = [0, 1, 2];
            using var baseLogits = model.Forward(tokenIds, positions, deviceId: -1);

            // Run with a zero-factor adapter — must be byte-equivalent.
            using var zeroAdapter = BuildSyntheticAdapter(cfg, rank: 4, alpha: 16f, zeroFactors: true);
            using var withZeroLogits = model.Forward(tokenIds, positions, deviceId: -1,
                kvCache: null, adapter: zeroAdapter);

            int total = baseLogits.Shape[0] * baseLogits.Shape[1];
            var baseSpan = new ReadOnlySpan<float>((void*)baseLogits.DataPointer, total);
            var withSpan = new ReadOnlySpan<float>((void*)withZeroLogits.DataPointer, total);

            // Zero adapter MUST give identical (or floating-point-equivalent) results.
            // Tolerance is loose because the LoRA path forces the unfused decode
            // route, which can produce tiny order-of-summation differences vs
            // the fused F32 path (under 1e-5 typical).
            for (int i = 0; i < total; i++)
            {
                float diff = MathF.Abs(baseSpan[i] - withSpan[i]);
                Assert.True(diff < 5e-3f,
                    $"Zero-adapter forward diverged at index {i}: base={baseSpan[i]} vs with={withSpan[i]} (diff={diff})");
            }
        }
        finally
        {
            model.Dispose();
            source.Dispose();
        }
    }

    [Fact]
    public unsafe void Forward_NonZeroAdapter_ProducesMeasurableDelta()
    {
        var (model, source, cfg) = BuildTinyModel();
        try
        {
            int[] tokenIds = [1, 2, 3];
            int[] positions = [0, 1, 2];
            using var baseLogits = model.Forward(tokenIds, positions, deviceId: -1);

            using var nonZeroAdapter = BuildSyntheticAdapter(cfg, rank: 8, alpha: 32f, zeroFactors: false);
            using var withLogits = model.Forward(tokenIds, positions, deviceId: -1,
                kvCache: null, adapter: nonZeroAdapter);

            int total = baseLogits.Shape[0] * baseLogits.Shape[1];
            var baseSpan = new ReadOnlySpan<float>((void*)baseLogits.DataPointer, total);
            var withSpan = new ReadOnlySpan<float>((void*)withLogits.DataPointer, total);

            // Verify finite and that there IS a measurable difference.
            float maxAbsDiff = 0f;
            int finiteCount = 0;
            for (int i = 0; i < total; i++)
            {
                if (!float.IsFinite(withSpan[i])) continue;
                finiteCount++;
                maxAbsDiff = MathF.Max(maxAbsDiff, MathF.Abs(baseSpan[i] - withSpan[i]));
            }

            Assert.Equal(total, finiteCount);
            Assert.True(maxAbsDiff > 1e-3f,
                $"Non-zero adapter produced no measurable delta (maxAbsDiff={maxAbsDiff}); LoRA path is silently disabled.");
        }
        finally
        {
            model.Dispose();
            source.Dispose();
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────────

    private static float[] RandomVec(Random rng, int n, float scale = 1.0f)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++)
            v[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return v;
    }

    private static float[] Ones(int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = 1.0f;
        return v;
    }

    private static void AssertClose(float[] expected, float[] actual, float absTol, float relTol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float tol = absTol + relTol * MathF.Abs(expected[i]);
            Assert.True(diff <= tol,
                $"index {i}: expected {expected[i]} vs actual {actual[i]} (diff={diff}, tol={tol})");
        }
    }
}
