using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using DotLLM.Core.Diagnostics;
using DotLLM.Diagnostics;
using Xunit;

namespace DotLLM.Tests.Unit.Diagnostics;

/// <summary>
/// Unit tests for <see cref="SparseAutoencoder"/>, <see cref="SaeHook"/>, and <see cref="SaeLoader"/>.
/// </summary>
/// <remarks>
/// All tests are model-free — they build synthetic SAEs with hand-computable weights and
/// drive the hook via the <see cref="HookRegistry"/> directly. The real-model integration
/// path is exercised end-to-end by <c>DotLLM.Sample.Interpretability</c> and is left out of
/// the unit suite to avoid a GGUF dependency for diagnostics tests.
/// </remarks>
public sealed class SaeHookTests
{
    // ---------------------------------------------------------------------
    // Pure-math: TopK + L2Distance — no SAE needed.
    // ---------------------------------------------------------------------

    [Fact]
    public void TopK_ReturnsHighestMagnitudesDescending()
    {
        var values = new float[] { 0.1f, 0.4f, 0.05f, 0.3f, 0.15f };
        SaeMath.TopK(values, 3, out var indices, out var top);

        Assert.Equal(new[] { 1, 3, 4 }, indices);
        Assert.Equal(new[] { 0.4f, 0.3f, 0.15f }, top);
    }

    [Fact]
    public void TopK_ClampsKToVectorLength()
    {
        var values = new float[] { 0.2f, 0.8f };
        SaeMath.TopK(values, k: 10, out var indices, out var top);
        Assert.Equal(2, indices.Length);
        Assert.Equal(2, top.Length);
    }

    [Fact]
    public void L2Distance_SymmetricNonNegativeZeroOnEquality()
    {
        var a = new float[] { 1, 2, 3, 4 };
        var b = new float[] { 1, 2, 3, 4 };
        Assert.Equal(0f, SaeMath.L2Distance(a, b), precision: 6);

        var c = new float[] { 0, 0, 0, 0 };
        Assert.Equal(MathF.Sqrt(1 + 4 + 9 + 16), SaeMath.L2Distance(a, c), precision: 5);
    }

    // ---------------------------------------------------------------------
    // Synthetic SAE: hand-computed Encode/Decode against the documented formula.
    // ---------------------------------------------------------------------

    [Fact]
    public void Encode_ProducesReluLinearOutput_AgainstHandComputedWeights()
    {
        // d_in=2, d_sae=3 — small enough to compute by hand.
        // W_enc [d_in, d_sae] = [[1, 0, -1], [0, 1, 1]]  → row-major: [1,0,-1, 0,1,1]
        // b_enc [d_sae]       = [0, 0, 0]
        // W_dec [d_sae, d_in] = [[1, 0], [0, 1], [-1, 1]] → row-major: [1,0, 0,1, -1,1]
        // b_dec [d_in]        = [0, 0]
        //
        // activation = [2, 3]
        // pre-relu = activation @ W_enc + b_enc
        //          = [2*1 + 3*0, 2*0 + 3*1, 2*(-1) + 3*1]
        //          = [2, 3, 1]
        // ReLU = [2, 3, 1] (all non-negative)
        // Top-3 descending: indices [1, 0, 2], values [3, 2, 1]

        using var sae = new SparseAutoencoder(
            hiddenSize: 2, featureCount: 3,
            wEnc: new float[] { 1, 0, -1, 0, 1, 1 },
            bEnc: new float[] { 0, 0, 0 },
            wDec: new float[] { 1, 0, 0, 1, -1, 1 },
            bDec: new float[] { 0, 0 },
            topK: 3);

        var (indices, values) = sae.Encode(new float[] { 2, 3 });
        Assert.Equal(new[] { 1, 0, 2 }, indices);
        Assert.Equal(new[] { 3f, 2f, 1f }, values);
    }

    [Fact]
    public void Encode_ReluClampsNegativeOutputs()
    {
        // activation = [-2, -3] with the same weights drives all pre-ReLU outputs negative
        // → after ReLU all zero. Top-K still returns K entries (some zero).
        using var sae = new SparseAutoencoder(
            hiddenSize: 2, featureCount: 3,
            wEnc: new float[] { 1, 0, -1, 0, 1, 1 },
            bEnc: new float[] { 0, 0, 0 },
            wDec: new float[] { 1, 0, 0, 1, -1, 1 },
            bDec: new float[] { 0, 0 },
            topK: 3);

        var (_, values) = sae.Encode(new float[] { -2, -3 });
        Assert.All(values, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Decode_AppliesBDecPlusScaledRows()
    {
        // Using indices [1, 0, 2], values [3, 2, 1] from the encode test:
        // recon = b_dec + 3 * W_dec[1] + 2 * W_dec[0] + 1 * W_dec[2]
        //       = [0,0] + 3*[0,1] + 2*[1,0] + 1*[-1,1]
        //       = [2 - 1, 3 + 1] = [1, 4]
        using var sae = new SparseAutoencoder(
            hiddenSize: 2, featureCount: 3,
            wEnc: new float[] { 1, 0, -1, 0, 1, 1 },
            bEnc: new float[] { 0, 0, 0 },
            wDec: new float[] { 1, 0, 0, 1, -1, 1 },
            bDec: new float[] { 0, 0 },
            topK: 3);

        Span<float> output = new float[2];
        sae.Decode(new[] { 1, 0, 2 }, new[] { 3f, 2f, 1f }, output);
        Assert.Equal(1f, output[0], precision: 6);
        Assert.Equal(4f, output[1], precision: 6);
    }

    [Fact]
    public void EncodeDecodeRoundTrip_OnIdentitySaeReconstructsExactly()
    {
        // Identity SAE: d_in=d_sae=4, W_enc=W_dec=I, biases zero, ReLU keeps non-negative.
        // For a non-negative input, encode→decode is exact.
        int n = 4;
        var I = new float[n * n];
        for (int i = 0; i < n; i++) I[i * n + i] = 1.0f;

        using var sae = new SparseAutoencoder(
            hiddenSize: n, featureCount: n,
            wEnc: I, bEnc: new float[n], wDec: I, bDec: new float[n],
            topK: n);

        var input = new float[] { 0.5f, 1.5f, 2.5f, 3.5f };
        Span<float> recon = new float[n];

        var (indices, values) = sae.Encode(input);
        sae.Decode(indices, values, recon);

        for (int i = 0; i < n; i++)
            Assert.Equal(input[i], recon[i], precision: 6);

        Assert.Equal(0f, sae.ReconstructionError(input), precision: 5);
    }

    [Fact]
    public void TopK_RespectsKParameter_AndDropsLowerMagnitudeFeatures()
    {
        // d_in=4, d_sae=8 with deliberately sparsity-inducing weights: each input dim
        // drives exactly one feature strongly. With TopK=2, only the two strongest fire.
        int dIn = 4, dSae = 8;
        var wEnc = new float[dIn * dSae];
        for (int i = 0; i < dIn; i++) wEnc[i * dSae + i] = 1.0f; // input dim i → feature i, weight 1
        // Add a weak negative dependency to a high-index feature so the next-strongest
        // post-ReLU is clearly the input value itself, not noise.
        var wDec = new float[dSae * dIn];
        for (int i = 0; i < dIn; i++) wDec[i * dIn + i] = 1.0f;

        using var sae = new SparseAutoencoder(
            hiddenSize: dIn, featureCount: dSae,
            wEnc: wEnc, bEnc: new float[dSae], wDec: wDec, bDec: new float[dIn],
            topK: 2);

        var input = new float[] { 0.5f, 4.0f, 1.0f, 2.0f }; // strongest features: 1 (=4), 3 (=2)
        var (indices, values) = sae.Encode(input);
        Assert.Equal(2, indices.Length);
        Assert.Equal(2, values.Length);
        Assert.Equal(1, indices[0]);
        Assert.Equal(4.0f, values[0]);
        Assert.Equal(3, indices[1]);
        Assert.Equal(2.0f, values[1]);
    }

    [Fact]
    public void ApplyBDecToInput_SubtractsBeforeEncode()
    {
        // With b_dec = [1, 1] and apply_b_dec_to_input = true, encode sees activation - b_dec.
        // activation = [3, 4] → pre-encode = [2, 3] → same as the first hand-computed test.
        using var sae = new SparseAutoencoder(
            hiddenSize: 2, featureCount: 3,
            wEnc: new float[] { 1, 0, -1, 0, 1, 1 },
            bEnc: new float[] { 0, 0, 0 },
            wDec: new float[] { 1, 0, 0, 1, -1, 1 },
            bDec: new float[] { 1, 1 },
            topK: 3,
            applyBDecToInput: true);

        var (indices, values) = sae.Encode(new float[] { 3, 4 });
        Assert.Equal(new[] { 1, 0, 2 }, indices);
        Assert.Equal(new[] { 3f, 2f, 1f }, values);
    }

    [Fact]
    public void Encode_ThrowsOnLengthMismatch()
    {
        using var sae = MakeIdentitySae(4);
        Assert.Throws<ArgumentException>(() => sae.Encode(new float[3]));
    }

    [Fact]
    public void Decode_ThrowsOnOutOfRangeFeatureIndex()
    {
        using var sae = MakeIdentitySae(4);
        Span<float> output = new float[4];
        Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            sae.Decode(new[] { 99 }, new[] { 1.0f }, new float[4]);
        });
    }

    [Fact]
    public void Dispose_FreesUnmanagedAndPreventsReuse()
    {
        var sae = MakeIdentitySae(4);
        sae.Dispose();
        Assert.Throws<ObjectDisposedException>(() => sae.Encode(new float[4]));

        // Idempotent.
        sae.Dispose();
    }

    // ---------------------------------------------------------------------
    // Hook integration: fire via HookRegistry, verify SaeResult population.
    // ---------------------------------------------------------------------

    [Fact]
    public void Hook_HookPointIsPostLayer()
    {
        using var sae = MakeIdentitySae(4);
        var hook = new SaeHook(sae);
        Assert.Equal(HookPoint.PostLayer, hook.HookPoint);
    }

    [Fact]
    public void Hook_FiresThroughRegistry_AndPopulatesSaeResult()
    {
        using var sae = MakeIdentitySae(4);
        var hook = new SaeHook(sae, new SaeConfig { TopK = 2 });

        var registry = new HookRegistry();
        registry.Register(hook);

        // Simulate a PostLayer fire for layer 0, position 0 with a non-negative activation.
        Span<float> activation = new float[] { 0.5f, 4.0f, 1.0f, 2.0f };
        var ctx = new HookContext(LayerIndex: 0, TokenPosition: 0, SequenceId: 1, CurrentStep: 0);
        registry.Fire(HookPoint.PostLayer, activation, in ctx);

        var results = hook.GetResults();
        var result = Assert.Single(results);
        Assert.Equal(0, result.LayerIndex);
        Assert.Equal(0, result.TokenPosition);
        Assert.Equal(2, result.FeatureIndices.Length);
        Assert.Equal(1, result.FeatureIndices[0]); // feature 1 fires at 4.0
        Assert.Equal(4.0f, result.FeatureMagnitudes[0]);
        Assert.Equal(3, result.FeatureIndices[1]); // feature 3 fires at 2.0
        Assert.Equal(2.0f, result.FeatureMagnitudes[1]);
        Assert.Equal(4, result.ActiveFeatureCount); // all 4 inputs > 0 → 4 features fire
    }

    [Fact]
    public void Hook_OnActivation_FiltersOutNonAnalyzedLayers()
    {
        using var sae = MakeIdentitySae(4);
        var hook = new SaeHook(sae, new SaeConfig
        {
            Layers = LogitLensLayerSelector.Specific(new[] { 1 }),
        });

        hook.OnActivation(new float[] { 1, 0, 0, 0 }, new HookContext(0, 0, 0, 0));
        hook.OnActivation(new float[] { 0, 1, 0, 0 }, new HookContext(1, 0, 0, 0));
        hook.OnActivation(new float[] { 0, 0, 1, 0 }, new HookContext(2, 0, 0, 0));

        Assert.Equal(1, hook.CaptureCount);
        Assert.Contains((1, 0), hook.CapturedKeys);
    }

    [Fact]
    public void Hook_OnActivation_FiltersOutNonAnalyzedTokenPositions()
    {
        using var sae = MakeIdentitySae(4);
        var hook = new SaeHook(sae, new SaeConfig
        {
            TokenPositions = new[] { 2 },
        });

        hook.OnActivation(new float[] { 1, 0, 0, 0 }, new HookContext(0, 0, 0, 0));
        hook.OnActivation(new float[] { 0, 1, 0, 0 }, new HookContext(0, 1, 0, 0));
        hook.OnActivation(new float[] { 0, 0, 1, 0 }, new HookContext(0, 2, 0, 0));

        Assert.Equal(1, hook.CaptureCount);
        Assert.Contains((0, 2), hook.CapturedKeys);
    }

    [Fact]
    public void Hook_OnActivation_RejectsLengthMismatch()
    {
        using var sae = MakeIdentitySae(4);
        var hook = new SaeHook(sae);

        // Out-of-spec activation: wrong length for the bound SAE. Should throw inside the
        // hook (not corrupt state silently) so misconfigured layer/SAE pairings surface.
        Assert.Throws<InvalidOperationException>(() =>
            hook.OnActivation(new float[3], new HookContext(0, 0, 0, 0)));
    }

    [Fact]
    public void Hook_ReturnsContinue_NeverReplaces()
    {
        // Read-only contract: the initial SAE hook never returns a Replace result. Steering
        // is an explicit future feature.
        using var sae = MakeIdentitySae(4);
        var hook = new SaeHook(sae);
        var result = hook.OnActivation(new float[] { 1, 2, 3, 4 }, new HookContext(0, 0, 0, 0));
        Assert.IsType<HookResult.ContinueResult>(result);
    }

    [Fact]
    public void Hook_GetResults_OrdersByPositionThenLayer()
    {
        using var sae = MakeIdentitySae(4);
        var hook = new SaeHook(sae);
        hook.OnActivation(new float[] { 0, 0, 0, 1 }, new HookContext(2, 1, 0, 0));
        hook.OnActivation(new float[] { 1, 0, 0, 0 }, new HookContext(0, 0, 0, 0));
        hook.OnActivation(new float[] { 0, 1, 0, 0 }, new HookContext(1, 0, 0, 0));

        var results = hook.GetResults();
        Assert.Equal(3, results.Count);
        Assert.Equal((0, 0), (results[0].TokenPosition, results[0].LayerIndex));
        Assert.Equal((0, 1), (results[1].TokenPosition, results[1].LayerIndex));
        Assert.Equal((1, 2), (results[2].TokenPosition, results[2].LayerIndex));
    }

    // ---------------------------------------------------------------------
    // Reconstruction-error bounds: synthetic high- and low-quality SAEs.
    // ---------------------------------------------------------------------

    [Fact]
    public void ReconstructionError_IdentitySae_IsZeroOnPositiveInputs()
    {
        using var sae = MakeIdentitySae(8);
        var input = new float[] { 0.1f, 1.0f, 0.5f, 2.0f, 0.3f, 0.7f, 1.5f, 0.9f };
        Assert.Equal(0f, sae.ReconstructionError(input), precision: 4);
    }

    [Fact]
    public void ReconstructionError_TopKTruncationProducesMeasurableLoss()
    {
        // d_in=4, d_sae=4 identity-ish SAE but TopK=2 — we drop the two smallest features.
        // input = [0.1, 1.0, 0.5, 2.0]; top-2 retained = [1.0, 2.0] (indices 1, 3);
        // dropped magnitudes = [0.1, 0.5]; reconstruction = [0, 1, 0, 2];
        // L2 error = sqrt(0.1^2 + 0.5^2) = sqrt(0.26) ≈ 0.5099.
        int n = 4;
        var I = new float[n * n];
        for (int i = 0; i < n; i++) I[i * n + i] = 1.0f;
        using var sae = new SparseAutoencoder(
            hiddenSize: n, featureCount: n,
            wEnc: I, bEnc: new float[n], wDec: I, bDec: new float[n],
            topK: 2);

        var input = new float[] { 0.1f, 1.0f, 0.5f, 2.0f };
        float err = sae.ReconstructionError(input);
        Assert.Equal(MathF.Sqrt(0.01f + 0.25f), err, precision: 4);
    }

    // ---------------------------------------------------------------------
    // SaeLoader: round-trip a synthetic safetensors buffer.
    // ---------------------------------------------------------------------

    [Fact]
    public void Loader_RoundTripsSyntheticSafetensorsBuffer()
    {
        // Build a safetensors buffer in memory: 4 F32 tensors (W_enc, b_enc, W_dec, b_dec)
        // with known values, then load and assert encode produces the same output as a
        // span-constructed SAE.
        int dIn = 2, dSae = 3;
        var wEnc = new float[] { 1, 0, -1, 0, 1, 1 };          // shape [d_in, d_sae]
        var bEnc = new float[] { 0, 0, 0 };                     // shape [d_sae]
        var wDec = new float[] { 1, 0, 0, 1, -1, 1 };          // shape [d_sae, d_in]
        var bDec = new float[] { 0, 0 };                        // shape [d_in]

        byte[] safetensorsBytes = BuildSafetensors(
            ("W_enc", new long[] { dIn, dSae }, wEnc),
            ("b_enc", new long[] { dSae }, bEnc),
            ("W_dec", new long[] { dSae, dIn }, wDec),
            ("b_dec", new long[] { dIn }, bDec));

        using var sae = SaeLoader.LoadFromBytes(safetensorsBytes, cfg: null, topK: 3);

        Assert.Equal(dIn, sae.HiddenSize);
        Assert.Equal(dSae, sae.FeatureCount);

        var (indices, values) = sae.Encode(new float[] { 2, 3 });
        Assert.Equal(new[] { 1, 0, 2 }, indices);
        Assert.Equal(new[] { 3f, 2f, 1f }, values);
    }

    [Fact]
    public void Loader_HonoursApplyBDecToInputFromCfg()
    {
        int dIn = 2, dSae = 3;
        var wEnc = new float[] { 1, 0, -1, 0, 1, 1 };
        var bEnc = new float[] { 0, 0, 0 };
        var wDec = new float[] { 1, 0, 0, 1, -1, 1 };
        var bDec = new float[] { 1, 1 };

        byte[] safetensorsBytes = BuildSafetensors(
            ("W_enc", new long[] { dIn, dSae }, wEnc),
            ("b_enc", new long[] { dSae }, bEnc),
            ("W_dec", new long[] { dSae, dIn }, wDec),
            ("b_dec", new long[] { dIn }, bDec));

        var cfg = new SaeCfg(DIn: dIn, DSae: dSae, ApplyBDecToInput: true, HookPoint: "blocks.0.hook_resid_post");
        using var sae = SaeLoader.LoadFromBytes(safetensorsBytes, cfg, topK: 3);

        // activation = [3, 4]; (3-1)=2, (4-1)=3 → same encode as the unbiased hand-computed test.
        var (indices, values) = sae.Encode(new float[] { 3, 4 });
        Assert.Equal(new[] { 1, 0, 2 }, indices);
        Assert.Equal(new[] { 3f, 2f, 1f }, values);
    }

    [Fact]
    public void Loader_RejectsBufferMissingRequiredTensor()
    {
        int dIn = 2, dSae = 3;
        byte[] safetensorsBytes = BuildSafetensors(
            // Missing W_enc on purpose.
            ("b_enc", new long[] { dSae }, new float[dSae]),
            ("W_dec", new long[] { dSae, dIn }, new float[dSae * dIn]),
            ("b_dec", new long[] { dIn }, new float[dIn]));

        Assert.Throws<InvalidDataException>(() => SaeLoader.LoadFromBytes(safetensorsBytes));
    }

    [Fact]
    public void Loader_RejectsConfigShapeMismatch()
    {
        int dIn = 2, dSae = 3;
        byte[] safetensorsBytes = BuildSafetensors(
            ("W_enc", new long[] { dIn, dSae }, new float[dIn * dSae]),
            ("b_enc", new long[] { dSae }, new float[dSae]),
            ("W_dec", new long[] { dSae, dIn }, new float[dSae * dIn]),
            ("b_dec", new long[] { dIn }, new float[dIn]));

        // cfg.json declares a different d_in — should fail rather than silently disagree.
        var cfg = new SaeCfg(DIn: 999, DSae: null, ApplyBDecToInput: false, HookPoint: null);
        Assert.Throws<InvalidDataException>(() => SaeLoader.LoadFromBytes(safetensorsBytes, cfg));
    }

    [Fact]
    public void Loader_RejectsUnsupportedDType()
    {
        // Hand-craft a safetensors buffer where W_enc claims F16 — should refuse with a
        // clear error message rather than silently misinterpret bytes.
        var headerJson = """
            {
              "W_enc": {"dtype":"F16","shape":[2,3],"data_offsets":[0,12]},
              "b_enc": {"dtype":"F32","shape":[3],"data_offsets":[12,24]},
              "W_dec": {"dtype":"F32","shape":[3,2],"data_offsets":[24,48]},
              "b_dec": {"dtype":"F32","shape":[2],"data_offsets":[48,56]}
            }
            """;
        var headerBytes = Encoding.UTF8.GetBytes(headerJson);
        var data = new byte[56];
        var buf = new byte[8 + headerBytes.Length + data.Length];
        BinaryPrimitives.WriteUInt64LittleEndian(buf.AsSpan(0, 8), (ulong)headerBytes.Length);
        headerBytes.CopyTo(buf.AsSpan(8));
        data.CopyTo(buf.AsSpan(8 + headerBytes.Length));

        Assert.Throws<NotSupportedException>(() => SaeLoader.LoadFromBytes(buf));
    }

    [Fact]
    public void SaeCfg_ParseJson_ReadsKnownFields()
    {
        var cfg = SaeCfg.ParseJson("""
            { "d_in": 768, "d_sae": 24576, "apply_b_dec_to_input": true, "hook_name": "blocks.5.hook_resid_post" }
            """);
        Assert.Equal(768, cfg.DIn);
        Assert.Equal(24576, cfg.DSae);
        Assert.True(cfg.ApplyBDecToInput);
        Assert.Equal("blocks.5.hook_resid_post", cfg.HookPoint);
    }

    [Fact]
    public void SaeCfg_ParseJson_DefaultsApplyBDecToFalseWhenAbsent()
    {
        var cfg = SaeCfg.ParseJson("""{ "d_in": 4, "d_sae": 8 }""");
        Assert.False(cfg.ApplyBDecToInput);
    }

    // ---------------------------------------------------------------------
    // Helpers.
    // ---------------------------------------------------------------------

    private static SparseAutoencoder MakeIdentitySae(int n)
    {
        var I = new float[n * n];
        for (int i = 0; i < n; i++) I[i * n + i] = 1.0f;
        return new SparseAutoencoder(
            hiddenSize: n, featureCount: n,
            wEnc: I, bEnc: new float[n], wDec: I, bDec: new float[n],
            topK: n);
    }

    /// <summary>
    /// Builds a synthetic safetensors buffer from F32 tensors in the order given. Headers
    /// are produced in canonical { name: { dtype, shape, data_offsets } } form; data ranges
    /// are tightly packed back-to-back starting at offset 0 within the data region.
    /// </summary>
    private static byte[] BuildSafetensors(params (string Name, long[] Shape, float[] Data)[] tensors)
    {
        // Compute offsets relative to the data section.
        var entries = new List<(string Name, long[] Shape, float[] Data, long Begin, long End)>();
        long cursor = 0;
        foreach (var (name, shape, data) in tensors)
        {
            long bytes = data.Length * sizeof(float);
            entries.Add((name, shape, data, cursor, cursor + bytes));
            cursor += bytes;
        }
        long dataBytes = cursor;

        // Build header JSON manually so test fixtures match the same format the loader parses.
        var sb = new StringBuilder();
        sb.Append('{');
        for (int i = 0; i < entries.Count; i++)
        {
            if (i > 0) sb.Append(',');
            var e = entries[i];
            sb.Append('"').Append(e.Name).Append("\":{");
            sb.Append("\"dtype\":\"F32\",");
            sb.Append("\"shape\":[");
            for (int s = 0; s < e.Shape.Length; s++)
            {
                if (s > 0) sb.Append(',');
                sb.Append(e.Shape[s]);
            }
            sb.Append("],");
            sb.Append("\"data_offsets\":[").Append(e.Begin).Append(',').Append(e.End).Append(']');
            sb.Append('}');
        }
        sb.Append('}');

        var headerBytes = Encoding.UTF8.GetBytes(sb.ToString());
        var buf = new byte[8 + headerBytes.Length + dataBytes];
        BinaryPrimitives.WriteUInt64LittleEndian(buf.AsSpan(0, 8), (ulong)headerBytes.Length);
        headerBytes.CopyTo(buf.AsSpan(8));

        int dataBase = 8 + headerBytes.Length;
        foreach (var e in entries)
        {
            var dst = MemoryMarshal.Cast<byte, float>(buf.AsSpan(dataBase + (int)e.Begin, e.Data.Length * sizeof(float)));
            e.Data.AsSpan().CopyTo(dst);
        }

        return buf;
    }
}
