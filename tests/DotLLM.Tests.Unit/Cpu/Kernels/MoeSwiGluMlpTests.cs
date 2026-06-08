using System.Runtime.InteropServices;
using DotLLM.Core.Lora;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Unit tests for <see cref="MoeSwiGluMlp"/>. Exercises the dense-routing
/// top-k kernel at hidden=8, intermediate=16, 4 experts, top-2 against a
/// hand-rolled reference (forward-order <c>torch.topk</c> semantics,
/// softmax-then-renormalise gating weights, per-expert SwiGLU MLP).
/// </summary>
public sealed unsafe class MoeSwiGluMlpTests
{
    private const int Hidden = 8;
    private const int Intermediate = 16;
    private const int NumExperts = 4;
    private const int TopK = 2;
    private const int SeqLen = 3;

    /// <summary>
    /// Top-k selection is stable on ties: lower-indexed expert wins.
    /// Guards the tiebreaker contract documented in the kernel.
    /// </summary>
    [Fact]
    public void SelectTopK_StableTies_LowerIndexWins()
    {
        // Equal probabilities: every slot ties. Top-2 should pick indices 0,1.
        float[] probs = [0.25f, 0.25f, 0.25f, 0.25f];
        Span<int> idx = stackalloc int[TopK];
        Span<float> prob = stackalloc float[TopK];
        MoeSwiGluMlp.SelectTopK(probs, idx, prob);

        Assert.Equal(0, idx[0]);
        Assert.Equal(1, idx[1]);
        Assert.Equal(0.25f, prob[0]);
        Assert.Equal(0.25f, prob[1]);
    }

    /// <summary>
    /// Non-tied: top-k picks the largest, then next-largest, in descending
    /// probability order.
    /// </summary>
    [Fact]
    public void SelectTopK_StrictOrder_PicksLargestThenNext()
    {
        float[] probs = [0.05f, 0.4f, 0.15f, 0.4f]; // indices 1,3 tie as largest
        Span<int> idx = stackalloc int[TopK];
        Span<float> prob = stackalloc float[TopK];
        MoeSwiGluMlp.SelectTopK(probs, idx, prob);

        Assert.Equal(1, idx[0]);   // tie → lower index
        Assert.Equal(3, idx[1]);
        Assert.Equal(0.4f, prob[0]);
        Assert.Equal(0.4f, prob[1]);
    }

    /// <summary>
    /// End-to-end sanity: MoE kernel output matches a scalar reference
    /// that replicates Mixtral's
    /// <c>softmax → topk → renormalise → weighted sum of per-expert SwiGLU</c>.
    /// </summary>
    [Fact]
    public void Execute_MatchesScalarReference()
    {
        var rng = new Random(12345);
        float[] hidden = RandomF32(rng, SeqLen * Hidden, -1f, 1f);
        float[] gate = RandomF32(rng, NumExperts * Hidden, -0.5f, 0.5f);
        float[][] w1 = new float[NumExperts][];
        float[][] w2 = new float[NumExperts][];
        float[][] w3 = new float[NumExperts][];
        for (int e = 0; e < NumExperts; e++)
        {
            w1[e] = RandomF32(rng, Intermediate * Hidden, -0.5f, 0.5f);
            w3[e] = RandomF32(rng, Intermediate * Hidden, -0.5f, 0.5f);
            w2[e] = RandomF32(rng, Hidden * Intermediate, -0.5f, 0.5f);
        }

        float[] actual = new float[SeqLen * Hidden];
        float[] expected = ReferenceMoe(hidden, gate, w1, w2, w3, TopK);

        using var pin = new Pinned(w1, w2, w3);
        MoeSwiGluMlp.Execute(
            hidden, gate, pin.W1, pin.W2, pin.W3, actual,
            NumExperts, TopK, Hidden, Intermediate, SeqLen);

        for (int i = 0; i < actual.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) < 1e-4f,
                $"[i={i}] actual={actual[i]} expected={expected[i]} diff={actual[i] - expected[i]}");
    }

    [Fact]
    public void Execute_PerExpertGateProjLora_MatchesEquivalentMergedWeight()
    {
        var rng = new Random(20260428);
        float[] hidden = RandomF32(rng, SeqLen * Hidden, -0.5f, 0.5f);
        float[] gate = new float[NumExperts * Hidden]; // uniform routing: stable top-k selects experts 0 and 1.
        float[][] w1 = new float[NumExperts][];
        float[][] w2 = new float[NumExperts][];
        float[][] w3 = new float[NumExperts][];
        for (int e = 0; e < NumExperts; e++)
        {
            w1[e] = RandomF32(rng, Intermediate * Hidden, -0.25f, 0.25f);
            w3[e] = RandomF32(rng, Intermediate * Hidden, -0.25f, 0.25f);
            w2[e] = RandomF32(rng, Hidden * Intermediate, -0.25f, 0.25f);
        }

        float[] b = Enumerable.Range(0, Hidden).Select(i => (i + 1) * 0.004f).ToArray();
        float[] a = Enumerable.Range(0, Intermediate).Select(i => (i - 5) * 0.003f).ToArray();
        using var adapter = BuildRankOneAdapter("mlp.experts.0.gate_proj", Hidden, Intermediate, b, a);

        float[] actual = new float[SeqLen * Hidden];
        using (var pin = new Pinned(w1, w2, w3))
        {
            MoeSwiGluMlp.Execute(
                hidden, gate, pin.W1, pin.W2, pin.W3, actual,
                NumExperts, TopK, Hidden, Intermediate, SeqLen,
                loraAdapter: adapter,
                loraLayer: 0);
        }

        float[][] mergedW1 = w1.Select(row => (float[])row.Clone()).ToArray();
        AddOuterProduct(mergedW1[0], Intermediate, Hidden, a, b);
        float[] expected = new float[SeqLen * Hidden];
        using (var pin = new Pinned(mergedW1, w2, w3))
        {
            MoeSwiGluMlp.Execute(
                hidden, gate, pin.W1, pin.W2, pin.W3, expected,
                NumExperts, TopK, Hidden, Intermediate, SeqLen);
        }

        for (int i = 0; i < actual.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) < 1e-4f,
                $"[i={i}] actual={actual[i]} expected={expected[i]} diff={actual[i] - expected[i]}");
    }

    /// <summary>
    /// One-hot router output (after softmax) → only one expert fires with
    /// weight 1.0. Output must equal that expert's dense SwiGLU output.
    /// </summary>
    [Fact]
    public void Execute_OneHotRouter_EquivalentToSingleExpert()
    {
        var rng = new Random(9001);
        // Make gate weights such that expert #2 wins by a landslide.
        // We set gate row 2 to match hidden direction; others to zero.
        float[] hidden = RandomF32(rng, Hidden, -0.1f, 0.1f);
        // Normalize hidden vs. gate-row-2 to force expert-2 dominance.
        float[] gate = new float[NumExperts * Hidden];
        for (int j = 0; j < Hidden; j++) gate[2 * Hidden + j] = hidden[j] * 1000f;
        // All other gate rows remain zero → dot with hidden = 0.

        float[][] w1 = new float[NumExperts][];
        float[][] w2 = new float[NumExperts][];
        float[][] w3 = new float[NumExperts][];
        for (int e = 0; e < NumExperts; e++)
        {
            w1[e] = RandomF32(rng, Intermediate * Hidden, -0.5f, 0.5f);
            w3[e] = RandomF32(rng, Intermediate * Hidden, -0.5f, 0.5f);
            w2[e] = RandomF32(rng, Hidden * Intermediate, -0.5f, 0.5f);
        }

        float[] actual = new float[Hidden];
        using (var pin = new Pinned(w1, w2, w3))
        {
            MoeSwiGluMlp.Execute(
                hidden, gate, pin.W1, pin.W2, pin.W3, actual,
                NumExperts, TopK, Hidden, Intermediate, seqLen: 1);
        }

        // Expected: after softmax the top-2 are {expert 2, expert 0 (or any
        // other expert on stable tiebreaker)}. Because expert 2's logit is
        // huge and the others are 0, softmax ≈ [0,0,1,0]. Top-2 gathers
        // (expert 2 with prob≈1, expert X with prob≈0). Renormalise: expert 2
        // weight → ~1.0, other → ~0. Output ≈ dense SwiGLU of expert 2.
        float[] denseExpertOut = DenseSwiGlu(hidden, w1[2], w2[2], w3[2]);

        for (int j = 0; j < Hidden; j++)
            Assert.True(Math.Abs(actual[j] - denseExpertOut[j]) < 1e-3f,
                $"[j={j}] actual={actual[j]} expected={denseExpertOut[j]}");
    }

    /// <summary>
    /// Zero gate weights → uniform softmax → top-k renormalises to 1/k per
    /// selected expert. For 4 experts top-2 with stable ties (picks indices
    /// 0 and 1): output = 0.5 × SwiGLU_0(hidden) + 0.5 × SwiGLU_1(hidden).
    /// </summary>
    [Fact]
    public void Execute_UniformRouter_EquivalentToAverageOfTopKExperts()
    {
        var rng = new Random(42);
        float[] hidden = RandomF32(rng, Hidden, -1f, 1f);
        float[] gate = new float[NumExperts * Hidden]; // zeros → uniform softmax.

        float[][] w1 = new float[NumExperts][];
        float[][] w2 = new float[NumExperts][];
        float[][] w3 = new float[NumExperts][];
        for (int e = 0; e < NumExperts; e++)
        {
            w1[e] = RandomF32(rng, Intermediate * Hidden, -0.5f, 0.5f);
            w3[e] = RandomF32(rng, Intermediate * Hidden, -0.5f, 0.5f);
            w2[e] = RandomF32(rng, Hidden * Intermediate, -0.5f, 0.5f);
        }

        float[] actual = new float[Hidden];
        using (var pin = new Pinned(w1, w2, w3))
        {
            MoeSwiGluMlp.Execute(
                hidden, gate, pin.W1, pin.W2, pin.W3, actual,
                NumExperts, TopK, Hidden, Intermediate, seqLen: 1);
        }

        // Uniform softmax over 4 = [0.25, 0.25, 0.25, 0.25]. Top-2 picks {0,1}
        // (stable tiebreak, lower index wins), renormalised → [0.5, 0.5].
        float[] e0 = DenseSwiGlu(hidden, w1[0], w2[0], w3[0]);
        float[] e1 = DenseSwiGlu(hidden, w1[1], w2[1], w3[1]);
        for (int j = 0; j < Hidden; j++)
        {
            float expected = 0.5f * e0[j] + 0.5f * e1[j];
            Assert.True(Math.Abs(actual[j] - expected) < 1e-4f,
                $"[j={j}] actual={actual[j]} expected={expected}");
        }
    }

    /// <summary>
    /// Qwen-MoE with shared expert (no sigmoid gate) and
    /// <c>norm_topk_prob=true</c>: output must equal Mixtral routed output +
    /// dense SwiGLU shared-expert output, added per token.
    /// </summary>
    [Fact]
    public void ExecuteWithSharedExpert_UnGated_AddsDenseSharedToRouted()
    {
        const int sharedIntermediate = 12; // deliberately != Intermediate so we catch mis-sized scratch
        var rng = new Random(77);
        float[] hidden = RandomF32(rng, SeqLen * Hidden, -0.5f, 0.5f);
        float[] gate = RandomF32(rng, NumExperts * Hidden, -0.3f, 0.3f);
        float[][] w1 = new float[NumExperts][];
        float[][] w2 = new float[NumExperts][];
        float[][] w3 = new float[NumExperts][];
        for (int e = 0; e < NumExperts; e++)
        {
            w1[e] = RandomF32(rng, Intermediate * Hidden, -0.3f, 0.3f);
            w3[e] = RandomF32(rng, Intermediate * Hidden, -0.3f, 0.3f);
            w2[e] = RandomF32(rng, Hidden * Intermediate, -0.3f, 0.3f);
        }
        float[] sharedW1 = RandomF32(rng, sharedIntermediate * Hidden, -0.3f, 0.3f);
        float[] sharedW3 = RandomF32(rng, sharedIntermediate * Hidden, -0.3f, 0.3f);
        float[] sharedW2 = RandomF32(rng, Hidden * sharedIntermediate, -0.3f, 0.3f);

        // Reference: Mixtral routed (renormalised) + per-token shared SwiGLU, no sigmoid.
        float[] expected = ReferenceMoe(hidden, gate, w1, w2, w3, TopK);
        for (int t = 0; t < SeqLen; t++)
        {
            float[] x = hidden.AsSpan(t * Hidden, Hidden).ToArray();
            float[] s = DenseSwiGluVar(x, sharedW1, sharedW2, sharedW3, Hidden, sharedIntermediate);
            for (int h = 0; h < Hidden; h++) expected[t * Hidden + h] += s[h];
        }

        float[] actual = new float[SeqLen * Hidden];
        using (var pin = new Pinned(w1, w2, w3))
        using (var sharedPin = new PinnedShared(sharedW1, sharedW2, sharedW3))
        {
            MoeSwiGluMlp.ExecuteWithSharedExpert(
                hidden, gate, pin.W1, pin.W2, pin.W3, actual,
                NumExperts, TopK, Hidden, Intermediate, SeqLen,
                normTopKProb: true,
                sharedGateProj: sharedPin.W1, sharedUpProj: sharedPin.W3, sharedDownProj: sharedPin.W2,
                sharedIntermediateSize: sharedIntermediate,
                sharedExpertGate: ReadOnlySpan<float>.Empty);
        }

        for (int i = 0; i < actual.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) < 1e-4f,
                $"[i={i}] actual={actual[i]} expected={expected[i]} diff={actual[i] - expected[i]}");
    }

    /// <summary>
    /// Qwen1.5-MoE variant: shared expert <b>with</b> a per-token sigmoid gate
    /// (<c>sigmoid(hidden . shared_expert_gate)</c> scales the shared output)
    /// and <c>norm_topk_prob=false</c> (raw softmax values as gating weights).
    /// </summary>
    [Fact]
    public void ExecuteWithSharedExpert_WithSigmoidGate_AndNoRenorm_MatchesReference()
    {
        const int sharedIntermediate = 12;
        var rng = new Random(123);
        float[] hidden = RandomF32(rng, SeqLen * Hidden, -0.5f, 0.5f);
        float[] gate = RandomF32(rng, NumExperts * Hidden, -0.3f, 0.3f);
        float[][] w1 = new float[NumExperts][];
        float[][] w2 = new float[NumExperts][];
        float[][] w3 = new float[NumExperts][];
        for (int e = 0; e < NumExperts; e++)
        {
            w1[e] = RandomF32(rng, Intermediate * Hidden, -0.3f, 0.3f);
            w3[e] = RandomF32(rng, Intermediate * Hidden, -0.3f, 0.3f);
            w2[e] = RandomF32(rng, Hidden * Intermediate, -0.3f, 0.3f);
        }
        float[] sharedW1 = RandomF32(rng, sharedIntermediate * Hidden, -0.3f, 0.3f);
        float[] sharedW3 = RandomF32(rng, sharedIntermediate * Hidden, -0.3f, 0.3f);
        float[] sharedW2 = RandomF32(rng, Hidden * sharedIntermediate, -0.3f, 0.3f);
        float[] sharedGate = RandomF32(rng, Hidden, -0.5f, 0.5f);

        // Reference: routed with normTopKProb=false (raw softmax sums), plus
        // sigmoid(hidden . sharedGate) * dense shared expert per token.
        float[] expected = ReferenceMoe(hidden, gate, w1, w2, w3, TopK, normTopKProb: false);
        for (int t = 0; t < SeqLen; t++)
        {
            float[] x = hidden.AsSpan(t * Hidden, Hidden).ToArray();
            float[] s = DenseSwiGluVar(x, sharedW1, sharedW2, sharedW3, Hidden, sharedIntermediate);
            float logit = 0f;
            for (int h = 0; h < Hidden; h++) logit += sharedGate[h] * x[h];
            float scale = 1.0f / (1.0f + MathF.Exp(-logit));
            for (int h = 0; h < Hidden; h++) expected[t * Hidden + h] += scale * s[h];
        }

        float[] actual = new float[SeqLen * Hidden];
        using (var pin = new Pinned(w1, w2, w3))
        using (var sharedPin = new PinnedShared(sharedW1, sharedW2, sharedW3))
        {
            MoeSwiGluMlp.ExecuteWithSharedExpert(
                hidden, gate, pin.W1, pin.W2, pin.W3, actual,
                NumExperts, TopK, Hidden, Intermediate, SeqLen,
                normTopKProb: false,
                sharedGateProj: sharedPin.W1, sharedUpProj: sharedPin.W3, sharedDownProj: sharedPin.W2,
                sharedIntermediateSize: sharedIntermediate,
                sharedExpertGate: sharedGate);
        }

        for (int i = 0; i < actual.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) < 1e-4f,
                $"[i={i}] actual={actual[i]} expected={expected[i]} diff={actual[i] - expected[i]}");
    }

    /// <summary>
    /// Calling <see cref="MoeSwiGluMlp.ExecuteWithSharedExpert"/> with
    /// <c>sharedIntermediateSize=0</c> and empty shared pointer spans must
    /// produce an output byte-identical to the plain Mixtral path — the
    /// shared-expert overload is a strict superset of the routed-only kernel.
    /// </summary>
    [Fact]
    public void ExecuteWithSharedExpert_DisabledShared_MatchesMixtral()
    {
        var rng = new Random(999);
        float[] hidden = RandomF32(rng, SeqLen * Hidden, -0.5f, 0.5f);
        float[] gate = RandomF32(rng, NumExperts * Hidden, -0.3f, 0.3f);
        float[][] w1 = new float[NumExperts][];
        float[][] w2 = new float[NumExperts][];
        float[][] w3 = new float[NumExperts][];
        for (int e = 0; e < NumExperts; e++)
        {
            w1[e] = RandomF32(rng, Intermediate * Hidden, -0.3f, 0.3f);
            w3[e] = RandomF32(rng, Intermediate * Hidden, -0.3f, 0.3f);
            w2[e] = RandomF32(rng, Hidden * Intermediate, -0.3f, 0.3f);
        }

        float[] plain = new float[SeqLen * Hidden];
        float[] shared = new float[SeqLen * Hidden];
        using (var pin = new Pinned(w1, w2, w3))
        {
            MoeSwiGluMlp.Execute(
                hidden, gate, pin.W1, pin.W2, pin.W3, plain,
                NumExperts, TopK, Hidden, Intermediate, SeqLen);
            MoeSwiGluMlp.ExecuteWithSharedExpert(
                hidden, gate, pin.W1, pin.W2, pin.W3, shared,
                NumExperts, TopK, Hidden, Intermediate, SeqLen,
                normTopKProb: true,
                sharedGateProj: ReadOnlySpan<nint>.Empty,
                sharedUpProj: ReadOnlySpan<nint>.Empty,
                sharedDownProj: ReadOnlySpan<nint>.Empty,
                sharedIntermediateSize: 0,
                sharedExpertGate: ReadOnlySpan<float>.Empty);
        }

        for (int i = 0; i < plain.Length; i++)
            Assert.Equal(plain[i], shared[i]);
    }

    /// <summary>
    /// DeepSeek-V2/V3 convention: multiple shared experts with no sigmoid gate.
    /// Output must equal Mixtral routed sum + sum of dense SwiGLU outputs across
    /// all shared experts, added per token.
    /// </summary>
    [Fact]
    public void MoeSwiGluMlp_MultiSharedExpert_SumsOverSharedExperts()
    {
        const int sharedIntermediate = 12;
        const int numSharedExperts = 2;
        var rng = new Random(88);
        float[] hidden = RandomF32(rng, SeqLen * Hidden, -0.5f, 0.5f);
        float[] gate = RandomF32(rng, NumExperts * Hidden, -0.3f, 0.3f);
        float[][] w1 = new float[NumExperts][];
        float[][] w2 = new float[NumExperts][];
        float[][] w3 = new float[NumExperts][];
        for (int e = 0; e < NumExperts; e++)
        {
            w1[e] = RandomF32(rng, Intermediate * Hidden, -0.3f, 0.3f);
            w3[e] = RandomF32(rng, Intermediate * Hidden, -0.3f, 0.3f);
            w2[e] = RandomF32(rng, Hidden * Intermediate, -0.3f, 0.3f);
        }
        // Per-shared-expert SwiGLU weights (different for each).
        float[][] sharedW1 = new float[numSharedExperts][];
        float[][] sharedW2 = new float[numSharedExperts][];
        float[][] sharedW3 = new float[numSharedExperts][];
        for (int k = 0; k < numSharedExperts; k++)
        {
            sharedW1[k] = RandomF32(rng, sharedIntermediate * Hidden, -0.3f, 0.3f);
            sharedW3[k] = RandomF32(rng, sharedIntermediate * Hidden, -0.3f, 0.3f);
            sharedW2[k] = RandomF32(rng, Hidden * sharedIntermediate, -0.3f, 0.3f);
        }

        // Reference: Mixtral routed (renormalised) + SUM over shared experts (no gate).
        float[] expected = ReferenceMoe(hidden, gate, w1, w2, w3, TopK);
        for (int t = 0; t < SeqLen; t++)
        {
            float[] x = hidden.AsSpan(t * Hidden, Hidden).ToArray();
            for (int k = 0; k < numSharedExperts; k++)
            {
                float[] s = DenseSwiGluVar(x, sharedW1[k], sharedW2[k], sharedW3[k],
                    Hidden, sharedIntermediate);
                for (int h = 0; h < Hidden; h++) expected[t * Hidden + h] += s[h];
            }
        }

        float[] actual = new float[SeqLen * Hidden];
        using (var pin = new Pinned(w1, w2, w3))
        using (var sharedPin = new PinnedSharedMulti(sharedW1, sharedW2, sharedW3))
        {
            MoeSwiGluMlp.ExecuteWithSharedExpert(
                hidden, gate, pin.W1, pin.W2, pin.W3, actual,
                NumExperts, TopK, Hidden, Intermediate, SeqLen,
                normTopKProb: true,
                sharedGateProj: sharedPin.W1, sharedUpProj: sharedPin.W3, sharedDownProj: sharedPin.W2,
                sharedIntermediateSize: sharedIntermediate,
                sharedExpertGate: ReadOnlySpan<float>.Empty);
        }

        for (int i = 0; i < actual.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) < 1e-4f,
                $"[i={i}] actual={actual[i]} expected={expected[i]} diff={actual[i] - expected[i]}");
    }

    /// <summary>
    /// With one shared expert in array form, the output must match the
    /// single-shared-expert semantics exactly (bit-identical) — proves the
    /// numSharedExperts=1 path in the array-based kernel is a faithful
    /// rewrite of the pre-migration scalar shared path.
    /// </summary>
    [Fact]
    public void MoeSwiGluMlp_MultiSharedExpert_MatchesSingleSharedReference()
    {
        const int sharedIntermediate = 12;
        var rng = new Random(54321);
        float[] hidden = RandomF32(rng, SeqLen * Hidden, -0.5f, 0.5f);
        float[] gate = RandomF32(rng, NumExperts * Hidden, -0.3f, 0.3f);
        float[][] w1 = new float[NumExperts][];
        float[][] w2 = new float[NumExperts][];
        float[][] w3 = new float[NumExperts][];
        for (int e = 0; e < NumExperts; e++)
        {
            w1[e] = RandomF32(rng, Intermediate * Hidden, -0.3f, 0.3f);
            w3[e] = RandomF32(rng, Intermediate * Hidden, -0.3f, 0.3f);
            w2[e] = RandomF32(rng, Hidden * Intermediate, -0.3f, 0.3f);
        }
        float[] sharedW1 = RandomF32(rng, sharedIntermediate * Hidden, -0.3f, 0.3f);
        float[] sharedW3 = RandomF32(rng, sharedIntermediate * Hidden, -0.3f, 0.3f);
        float[] sharedW2 = RandomF32(rng, Hidden * sharedIntermediate, -0.3f, 0.3f);

        // Single-shared path (the existing Qwen-style call).
        float[] singleShared = new float[SeqLen * Hidden];
        using (var pin = new Pinned(w1, w2, w3))
        using (var sharedPin = new PinnedShared(sharedW1, sharedW2, sharedW3))
        {
            MoeSwiGluMlp.ExecuteWithSharedExpert(
                hidden, gate, pin.W1, pin.W2, pin.W3, singleShared,
                NumExperts, TopK, Hidden, Intermediate, SeqLen,
                normTopKProb: true,
                sharedGateProj: sharedPin.W1, sharedUpProj: sharedPin.W3, sharedDownProj: sharedPin.W2,
                sharedIntermediateSize: sharedIntermediate,
                sharedExpertGate: ReadOnlySpan<float>.Empty);
        }

        // Multi-shared path with length-1 arrays — must be bit-identical.
        float[] multiSharedK1 = new float[SeqLen * Hidden];
        float[][] sharedW1Arr = new[] { sharedW1 };
        float[][] sharedW2Arr = new[] { sharedW2 };
        float[][] sharedW3Arr = new[] { sharedW3 };
        using (var pin = new Pinned(w1, w2, w3))
        using (var sharedPin = new PinnedSharedMulti(sharedW1Arr, sharedW2Arr, sharedW3Arr))
        {
            MoeSwiGluMlp.ExecuteWithSharedExpert(
                hidden, gate, pin.W1, pin.W2, pin.W3, multiSharedK1,
                NumExperts, TopK, Hidden, Intermediate, SeqLen,
                normTopKProb: true,
                sharedGateProj: sharedPin.W1, sharedUpProj: sharedPin.W3, sharedDownProj: sharedPin.W2,
                sharedIntermediateSize: sharedIntermediate,
                sharedExpertGate: ReadOnlySpan<float>.Empty);
        }

        for (int i = 0; i < singleShared.Length; i++)
            Assert.Equal(singleShared[i], multiSharedK1[i]);
    }

    // ──────────────────── Reference implementation ────────────────────

    /// <summary>
    /// Scalar-loop reference: exact replica of Mixtral's MoE block for
    /// cross-checking. Not performance-tuned — just algorithmically correct.
    /// When <paramref name="normTopKProb"/> is false, the raw softmax-derived
    /// top-k probabilities are used directly as gating weights (no renormalisation)
    /// — this is the Qwen1.5-MoE convention.
    /// </summary>
    private static float[] ReferenceMoe(
        float[] hidden, float[] gate,
        float[][] w1, float[][] w2, float[][] w3,
        int topk,
        bool normTopKProb = true)
    {
        int seqLen = hidden.Length / Hidden;
        float[] output = new float[seqLen * Hidden];
        for (int t = 0; t < seqLen; t++)
        {
            ReadOnlySpan<float> x = hidden.AsSpan(t * Hidden, Hidden);

            // Router logits + full softmax.
            float[] logits = new float[NumExperts];
            for (int e = 0; e < NumExperts; e++)
                for (int h = 0; h < Hidden; h++)
                    logits[e] += gate[e * Hidden + h] * x[h];
            float[] probs = ScalarSoftmax(logits);

            // Top-k (stable: lower index wins on ties).
            int[] idx = new int[topk];
            float[] p = new float[topk];
            for (int slot = 0; slot < topk; slot++)
            {
                int bestI = -1; float bestV = float.NegativeInfinity;
                for (int i = 0; i < NumExperts; i++)
                {
                    bool claimed = false;
                    for (int s = 0; s < slot; s++) if (idx[s] == i) { claimed = true; break; }
                    if (claimed) continue;
                    if (probs[i] > bestV) { bestV = probs[i]; bestI = i; }
                }
                idx[slot] = bestI; p[slot] = bestV;
            }

            // Renormalise top-k by sum (Mixtral + Qwen3-MoE). Qwen1.5-MoE
            // leaves the raw values.
            if (normTopKProb)
            {
                float sum = 0f;
                for (int s = 0; s < topk; s++) sum += p[s];
                for (int s = 0; s < topk; s++) p[s] = sum > 0 ? p[s] / sum : 0f;
            }

            // Sum weighted expert outputs.
            Span<float> acc = output.AsSpan(t * Hidden, Hidden);
            for (int s = 0; s < topk; s++)
            {
                int e = idx[s];
                float[] dense = DenseSwiGlu(x.ToArray(), w1[e], w2[e], w3[e]);
                for (int h = 0; h < Hidden; h++) acc[h] += p[s] * dense[h];
            }
        }
        return output;
    }

    /// <summary>
    /// Dense SwiGLU MLP with explicit hidden / intermediate sizes — mirrors
    /// <see cref="DenseSwiGlu"/> but parametric so we can use a different
    /// intermediate width for the shared-expert branch.
    /// </summary>
    private static float[] DenseSwiGluVar(float[] x, float[] w1, float[] w2, float[] w3,
                                          int hidden, int intermediate)
    {
        float[] gate = new float[intermediate];
        float[] up = new float[intermediate];
        for (int i = 0; i < intermediate; i++)
        {
            float g = 0f, u = 0f;
            for (int h = 0; h < hidden; h++)
            {
                g += w1[i * hidden + h] * x[h];
                u += w3[i * hidden + h] * x[h];
            }
            gate[i] = g; up[i] = u;
        }
        float[] silu = new float[intermediate];
        for (int i = 0; i < intermediate; i++)
        {
            float s = gate[i] * (1f / (1f + MathF.Exp(-gate[i])));
            silu[i] = s * up[i];
        }
        float[] outBuf = new float[hidden];
        for (int h = 0; h < hidden; h++)
        {
            float d = 0f;
            for (int i = 0; i < intermediate; i++) d += w2[h * intermediate + i] * silu[i];
            outBuf[h] = d;
        }
        return outBuf;
    }

    private static float[] DenseSwiGlu(float[] x, float[] w1, float[] w2, float[] w3)
    {
        // gate[i] = w1[i,:] . x,  up[i] = w3[i,:] . x
        float[] gate = new float[Intermediate];
        float[] up = new float[Intermediate];
        for (int i = 0; i < Intermediate; i++)
        {
            float g = 0f, u = 0f;
            for (int h = 0; h < Hidden; h++)
            {
                g += w1[i * Hidden + h] * x[h];
                u += w3[i * Hidden + h] * x[h];
            }
            gate[i] = g; up[i] = u;
        }
        // silu = SiLu(gate) * up
        float[] silu = new float[Intermediate];
        for (int i = 0; i < Intermediate; i++)
        {
            float s = gate[i] * (1f / (1f + MathF.Exp(-gate[i])));
            silu[i] = s * up[i];
        }
        // out = w2 @ silu  → [Hidden]
        float[] outBuf = new float[Hidden];
        for (int h = 0; h < Hidden; h++)
        {
            float d = 0f;
            for (int i = 0; i < Intermediate; i++) d += w2[h * Intermediate + i] * silu[i];
            outBuf[h] = d;
        }
        return outBuf;
    }

    private static float[] ScalarSoftmax(float[] logits)
    {
        float max = logits[0];
        for (int i = 1; i < logits.Length; i++) if (logits[i] > max) max = logits[i];
        float[] y = new float[logits.Length];
        float sum = 0f;
        for (int i = 0; i < logits.Length; i++) { y[i] = MathF.Exp(logits[i] - max); sum += y[i]; }
        for (int i = 0; i < logits.Length; i++) y[i] /= sum;
        return y;
    }

    private static float[] RandomF32(Random rng, int count, float lo, float hi)
    {
        float[] arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = (float)(rng.NextDouble() * (hi - lo) + lo);
        return arr;
    }

    private static LoraAdapter BuildRankOneAdapter(
        string projection,
        int inputDim,
        int outputDim,
        float[] b,
        float[] a)
    {
        var adapter = new LoraAdapter("moe-test", rank: 1, alpha: 1f, targetModules: [projection]);
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
    /// Pins per-expert weight arrays and surfaces <c>nint</c> pointers for the
    /// kernel signature. Using <see cref="GCHandle"/> (pinned) because the
    /// kernel takes <c>ReadOnlySpan&lt;nint&gt;</c> rather than a nested fixed.
    /// </summary>
    private sealed class Pinned : IDisposable
    {
        private readonly GCHandle[] _handles;
        public readonly nint[] W1;
        public readonly nint[] W2;
        public readonly nint[] W3;
        public Pinned(float[][] w1, float[][] w2, float[][] w3)
        {
            _handles = new GCHandle[w1.Length + w2.Length + w3.Length];
            W1 = new nint[w1.Length];
            W2 = new nint[w2.Length];
            W3 = new nint[w3.Length];
            int h = 0;
            for (int e = 0; e < w1.Length; e++)
            {
                _handles[h] = GCHandle.Alloc(w1[e], GCHandleType.Pinned);
                W1[e] = _handles[h].AddrOfPinnedObject();
                h++;
                _handles[h] = GCHandle.Alloc(w2[e], GCHandleType.Pinned);
                W2[e] = _handles[h].AddrOfPinnedObject();
                h++;
                _handles[h] = GCHandle.Alloc(w3[e], GCHandleType.Pinned);
                W3[e] = _handles[h].AddrOfPinnedObject();
                h++;
            }
        }
        public void Dispose()
        {
            foreach (var h in _handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <summary>
    /// Pins a single shared-expert's three weight arrays and exposes them as
    /// length-1 nint arrays — the kernel's shared-expert API takes pointer
    /// spans whether there's one or many experts.
    /// </summary>
    private sealed class PinnedShared : IDisposable
    {
        private readonly GCHandle[] _handles;
        public readonly nint[] W1;
        public readonly nint[] W2;
        public readonly nint[] W3;
        public PinnedShared(float[] w1, float[] w2, float[] w3)
        {
            _handles = new GCHandle[3];
            _handles[0] = GCHandle.Alloc(w1, GCHandleType.Pinned);
            _handles[1] = GCHandle.Alloc(w2, GCHandleType.Pinned);
            _handles[2] = GCHandle.Alloc(w3, GCHandleType.Pinned);
            W1 = [_handles[0].AddrOfPinnedObject()];
            W2 = [_handles[1].AddrOfPinnedObject()];
            W3 = [_handles[2].AddrOfPinnedObject()];
        }
        public void Dispose()
        {
            foreach (var h in _handles) if (h.IsAllocated) h.Free();
        }
    }

    /// <summary>
    /// Pins an arbitrary number of shared-expert weight triples and exposes
    /// them as parallel nint arrays — used to exercise the multi-shared-expert
    /// code path (DeepSeek-V2/V3 <c>n_shared_experts &gt; 1</c>).
    /// </summary>
    private sealed class PinnedSharedMulti : IDisposable
    {
        private readonly GCHandle[] _handles;
        public readonly nint[] W1;
        public readonly nint[] W2;
        public readonly nint[] W3;
        public PinnedSharedMulti(float[][] w1, float[][] w2, float[][] w3)
        {
            int n = w1.Length;
            _handles = new GCHandle[n * 3];
            W1 = new nint[n];
            W2 = new nint[n];
            W3 = new nint[n];
            int h = 0;
            for (int k = 0; k < n; k++)
            {
                _handles[h] = GCHandle.Alloc(w1[k], GCHandleType.Pinned);
                W1[k] = _handles[h].AddrOfPinnedObject(); h++;
                _handles[h] = GCHandle.Alloc(w2[k], GCHandleType.Pinned);
                W2[k] = _handles[h].AddrOfPinnedObject(); h++;
                _handles[h] = GCHandle.Alloc(w3[k], GCHandleType.Pinned);
                W3[k] = _handles[h].AddrOfPinnedObject(); h++;
            }
        }
        public void Dispose()
        {
            foreach (var h in _handles) if (h.IsAllocated) h.Free();
        }
    }
}
