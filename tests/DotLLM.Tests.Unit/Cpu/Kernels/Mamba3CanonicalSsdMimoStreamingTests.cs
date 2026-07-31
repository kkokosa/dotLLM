using DotLLM.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Kernel-level tests for <see cref="Mamba3CanonicalSsd.ExecuteMimoStreaming"/>
/// — verifies the MIMO streaming analog of
/// <see cref="Mamba3CanonicalSsd.ExecuteMimo"/>:
/// <list type="bullet">
///   <item><description>
///     Streaming with a single full-chunk schedule and zero kState/vState
///     reproduces <see cref="Mamba3CanonicalSsd.ExecuteMimo"/> bit-for-bit
///     (the boundary adjustment is a no-op in that configuration).
///   </description></item>
///   <item><description>
///     Splitting into multiple chunks with kState/vState threading matches
///     the one-shot call within F32-reorder noise, for several schedules.
///   </description></item>
/// </list>
/// </summary>
/// <remarks>
/// Directly drives the kernel without the block-level in_proj / norm / RoPE
/// plumbing — inputs are deterministic ramps over the kernel's canonical
/// shape space ([T, R, H, N] B/C, [T, H, P] V, [H, R, P] mimo_z/mimo_o).
/// </remarks>
public sealed class Mamba3CanonicalSsdMimoStreamingTests
{
    // Tiny-but-realistic dims (fast to brute-force enumerate in a test).
    // R=3 to catch rank-coupling bugs that a R=2 fixture might miss.
    private const int SeqLen = 8;
    private const int NRank = 3;
    private const int NHead = 4;
    private const int HeadDim = 4;
    private const int DState = 6;

    // F32 reorder tol: boundary adjustment re-sums O(R·P·N) entries per head
    // across chunk edges; a few-ULP drift is expected on the state path.
    private const float AbsTol = 1e-5f;
    private const float RelTol = 1e-4f;

    private readonly ITestOutputHelper _output;
    public Mamba3CanonicalSsdMimoStreamingTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Streaming_SingleChunk_ZeroState_MatchesOneShot()
    {
        // Empty kState/vState is the "no prior chunk" path — the boundary
        // adjustment is a no-op, so the streaming kernel must equal
        // ExecuteMimo byte-for-byte.
        float[] v = Ramp(SeqLen * NHead * HeadDim, 0.03f, 0);
        float[] q = Ramp(SeqLen * NRank * NHead * DState, 0.02f, 1);
        float[] k = Ramp(SeqLen * NRank * NHead * DState, 0.02f, 2);
        float[] qkSum = Ramp(SeqLen * NHead, 0.1f, 3);
        float[] scale = Ramp(SeqLen * NHead, 0.05f, 4);
        float[] gamma = Ramp(SeqLen * NHead, 0.05f, 5);
        float[] adt = new float[SeqLen * NHead];
        // adt must be negative (clamped to -a_floor). Use small-magnitude neg.
        for (int i = 0; i < adt.Length; i++)
            adt[i] = -0.02f - 0.001f * i;
        float[] dt = Ramp(SeqLen * NHead, 0.02f, 6);
        float[] trap = Ramp(SeqLen * NHead, 0.5f, 7);
        float[] d = Ramp(NHead, 0.1f, 8);
        float[] z = Ramp(SeqLen * NHead * HeadDim, 0.1f, 9);
        float[] mimoZ = Ramp(NHead * NRank * HeadDim, 0.3f, 10);
        float[] mimoO = Ramp(NHead * NRank * HeadDim, 0.3f, 11);

        float[] stateA = new float[NHead * HeadDim * DState];
        float[] yA = new float[SeqLen * NHead * HeadDim];
        Mamba3CanonicalSsd.ExecuteMimo(
            stateA, v, q, k, qkSum, scale, gamma, adt, d, z, mimoZ, mimoO,
            yA, yPerRank: Span<float>.Empty,
            SeqLen, NRank, NHead, HeadDim, DState);

        float[] stateB = new float[NHead * HeadDim * DState];
        float[] yB = new float[SeqLen * NHead * HeadDim];
        // Zero streaming buffers — no prior chunk.
        float[] kState = new float[NRank * NHead * DState];
        float[] vState = new float[NHead * HeadDim];
        Mamba3CanonicalSsd.ExecuteMimoStreaming(
            stateB, v, q, k, qkSum, scale, gamma, adt, dt, trap, d, z, mimoZ, mimoO,
            kState, vState,
            yB, yPerRank: Span<float>.Empty,
            SeqLen, NRank, NHead, HeadDim, DState);

        // Bit-equal (no reorder — both code paths run the same scan once).
        for (int i = 0; i < yA.Length; i++) Assert.Equal(yA[i], yB[i]);
        for (int i = 0; i < stateA.Length; i++) Assert.Equal(stateA[i], stateB[i]);
    }

    [Theory]
    [InlineData(new[] { 8 })]
    [InlineData(new[] { 4, 4 })]
    [InlineData(new[] { 2, 2, 2, 2 })]
    [InlineData(new[] { 1, 1, 1, 1, 1, 1, 1, 1 })]
    [InlineData(new[] { 3, 5 })]
    [InlineData(new[] { 5, 3 })]
    [InlineData(new[] { 1, 2, 5 })]
    public void Streaming_SplitChunks_MatchesOneShot(int[] schedule)
    {
        int sum = 0; foreach (int s in schedule) sum += s;
        Assert.Equal(SeqLen, sum);

        float[] v = Ramp(SeqLen * NHead * HeadDim, 0.03f, 0);
        float[] q = Ramp(SeqLen * NRank * NHead * DState, 0.02f, 1);
        float[] k = Ramp(SeqLen * NRank * NHead * DState, 0.02f, 2);
        float[] qkSum = Ramp(SeqLen * NHead, 0.1f, 3);
        float[] gamma = Ramp(SeqLen * NHead, 0.05f, 5);
        float[] adt = new float[SeqLen * NHead];
        for (int i = 0; i < adt.Length; i++)
            adt[i] = -0.02f - 0.001f * i;
        float[] dt = Ramp(SeqLen * NHead, 0.02f, 6);
        // `trap` here is the already-sigmoided "trap" (0..1). Ramp into (0.1, 0.9).
        float[] trap = new float[SeqLen * NHead];
        for (int i = 0; i < trap.Length; i++)
            trap[i] = 0.3f + 0.4f * MathF.Cos(0.61803f * (i + 1));
        float[] d = Ramp(NHead, 0.1f, 8);
        float[] z = Ramp(SeqLen * NHead * HeadDim, 0.1f, 9);
        float[] mimoZ = Ramp(NHead * NRank * HeadDim, 0.3f, 10);
        float[] mimoO = Ramp(NHead * NRank * HeadDim, 0.3f, 11);

        // For the ONE-SHOT call we must compute scale = γ + shifted_γ over the
        // full T tokens, with shifted_γ[T-1] = 0. For the STREAMING call each
        // chunk sees its own local shifted_γ[last] = 0 — the boundary
        // adjustment reconstructs the missing term from kState/vState.
        float[] scaleFull = new float[SeqLen * NHead];
        for (int t = 0; t < SeqLen; t++)
        {
            for (int h = 0; h < NHead; h++)
            {
                float gm = gamma[t * NHead + h];
                float sh = 0f;
                if (t + 1 < SeqLen)
                {
                    int next = (t + 1) * NHead + h;
                    sh = dt[next] * (1f - trap[next]);
                }
                scaleFull[t * NHead + h] = gm + sh;
            }
        }

        // ── One-shot ────────────────────────────────────────────────────
        float[] stateOne = new float[NHead * HeadDim * DState];
        float[] yOne = new float[SeqLen * NHead * HeadDim];
        Mamba3CanonicalSsd.ExecuteMimo(
            stateOne, v, q, k, qkSum, scaleFull, gamma, adt, d, z, mimoZ, mimoO,
            yOne, yPerRank: Span<float>.Empty,
            SeqLen, NRank, NHead, HeadDim, DState);

        // ── Streaming (schedule) ────────────────────────────────────────
        float[] stateS = new float[NHead * HeadDim * DState];
        float[] yS = new float[SeqLen * NHead * HeadDim];
        float[] kState = new float[NRank * NHead * DState];
        float[] vState = new float[NHead * HeadDim];

        int offset = 0;
        foreach (int chunkLen in schedule)
        {
            // Chunk-local scale: shifted_γ[chunkEnd] = 0 by construction.
            float[] scaleChunk = new float[chunkLen * NHead];
            for (int t = 0; t < chunkLen; t++)
            {
                int tAbs = offset + t;
                for (int h = 0; h < NHead; h++)
                {
                    float gm = gamma[tAbs * NHead + h];
                    float sh = 0f;
                    if (t + 1 < chunkLen)
                    {
                        int nextAbs = (tAbs + 1) * NHead + h;
                        sh = dt[nextAbs] * (1f - trap[nextAbs]);
                    }
                    scaleChunk[t * NHead + h] = gm + sh;
                }
            }

            // Slice the per-token spans for this chunk window.
            var vC = new ReadOnlySpan<float>(v, offset * NHead * HeadDim,
                                             chunkLen * NHead * HeadDim);
            var qC = new ReadOnlySpan<float>(q, offset * NRank * NHead * DState,
                                             chunkLen * NRank * NHead * DState);
            var kC = new ReadOnlySpan<float>(k, offset * NRank * NHead * DState,
                                             chunkLen * NRank * NHead * DState);
            var qkC = new ReadOnlySpan<float>(qkSum, offset * NHead, chunkLen * NHead);
            var gammaC = new ReadOnlySpan<float>(gamma, offset * NHead, chunkLen * NHead);
            var adtC = new ReadOnlySpan<float>(adt, offset * NHead, chunkLen * NHead);
            var dtC = new ReadOnlySpan<float>(dt, offset * NHead, chunkLen * NHead);
            var trapC = new ReadOnlySpan<float>(trap, offset * NHead, chunkLen * NHead);
            var zC = new ReadOnlySpan<float>(z, offset * NHead * HeadDim,
                                             chunkLen * NHead * HeadDim);

            float[] yChunk = new float[chunkLen * NHead * HeadDim];
            Mamba3CanonicalSsd.ExecuteMimoStreaming(
                stateS, vC, qC, kC, qkC, scaleChunk, gammaC, adtC, dtC, trapC,
                d, zC, mimoZ, mimoO, kState, vState,
                yChunk, yPerRank: Span<float>.Empty,
                chunkLen, NRank, NHead, HeadDim, DState);

            Array.Copy(yChunk, 0, yS, offset * NHead * HeadDim,
                       chunkLen * NHead * HeadDim);
            offset += chunkLen;
        }
        Assert.Equal(SeqLen, offset);

        string label = string.Join("+", schedule);
        (float yMaxAbs, float yMaxRel) = Drift(yOne, yS);
        (float stateMaxAbs, float stateMaxRel) = Drift(stateOne, stateS);
        _output.WriteLine(
            $"{label}  y: abs={yMaxAbs:E3} rel={yMaxRel:E3}  "
            + $"ssm: abs={stateMaxAbs:E3} rel={stateMaxRel:E3}");

        AssertClose($"{label} y", yOne, yS);
        AssertClose($"{label} ssm_state", stateOne, stateS);
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    /// <summary>
    /// kState/vState are a pair; supplying only one is a caller error and must be
    /// reported as such rather than as a confusing "too small" message.
    /// </summary>
    [Fact]
    public void Streaming_HalfConfiguredBoundaryBuffers_Throws()
    {
        var args = MakeMinimalArgs();
        float[] kState = new float[NRank * NHead * DState];

        var ex = Assert.Throws<ArgumentException>(() =>
            args.Invoke(kState, [], state: new float[NHead * HeadDim * DState]));
        Assert.Contains("both", ex.Message);
    }

    /// <summary>
    /// The chunk-boundary adjustment mutates state before the inner scan validates
    /// anything, so an undersized state must be rejected up front — not surface as
    /// IndexOutOfRangeException from a half-applied boundary update.
    /// </summary>
    [Fact]
    public void Streaming_UndersizedState_ThrowsArgumentException()
    {
        var args = MakeMinimalArgs();
        float[] kState = new float[NRank * NHead * DState];
        float[] vState = new float[NHead * HeadDim];

        var ex = Assert.Throws<ArgumentException>(() =>
            args.Invoke(kState, vState, state: new float[NHead * HeadDim * DState - 1]));
        Assert.Equal("state", ex.ParamName);
    }

    /// <summary>Well-formed inputs for the argument-validation tests.</summary>
    private static StreamingArgs MakeMinimalArgs()
    {
        float[] adt = new float[SeqLen * NHead];
        for (int i = 0; i < adt.Length; i++) adt[i] = -0.02f - 0.001f * i;

        return new StreamingArgs(
            v: Ramp(SeqLen * NHead * HeadDim, 0.03f, 0),
            q: Ramp(SeqLen * NRank * NHead * DState, 0.02f, 1),
            k: Ramp(SeqLen * NRank * NHead * DState, 0.02f, 2),
            qkSum: Ramp(SeqLen * NHead, 0.1f, 3),
            scale: Ramp(SeqLen * NHead, 0.05f, 4),
            gamma: Ramp(SeqLen * NHead, 0.05f, 5),
            adt: adt,
            dt: Ramp(SeqLen * NHead, 0.02f, 6),
            trap: Ramp(SeqLen * NHead, 0.5f, 7),
            d: Ramp(NHead, 0.1f, 8),
            z: Ramp(SeqLen * NHead * HeadDim, 0.1f, 9),
            mimoZ: Ramp(NHead * NRank * HeadDim, 0.3f, 10),
            mimoO: Ramp(NHead * NRank * HeadDim, 0.3f, 11));
    }

    private sealed record StreamingArgs(
        float[] v, float[] q, float[] k, float[] qkSum, float[] scale,
        float[] gamma, float[] adt, float[] dt, float[] trap, float[] d,
        float[] z, float[] mimoZ, float[] mimoO)
    {
        public void Invoke(float[] kState, float[] vState, float[] state) =>
            Mamba3CanonicalSsd.ExecuteMimoStreaming(
                state, v, q, k, qkSum, scale, gamma, adt, dt, trap, d, z, mimoZ, mimoO,
                kState, vState,
                new float[SeqLen * NHead * HeadDim], yPerRank: Span<float>.Empty,
                SeqLen, NRank, NHead, HeadDim, DState);
    }

    private static float[] Ramp(int count, float amplitude, int seed)
    {
        float[] r = new float[count];
        for (int i = 0; i < count; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            r[i] = amplitude * MathF.Cos(phi);
        }
        return r;
    }

    private static (float maxAbs, float maxRel) Drift(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float maxAbs = 0f, maxRel = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            float diff = MathF.Abs(a[i] - b[i]);
            float rel = diff / (MathF.Abs(a[i]) + 1e-12f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
        }
        return (maxAbs, maxRel);
    }

    private static void AssertClose(string label, float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float absDiff = MathF.Abs(e - a);
            float relDiff = absDiff / (MathF.Abs(e) + 1e-12f);
            if (absDiff > AbsTol && relDiff > RelTol)
            {
                Assert.Fail(
                    $"{label}[{i}]: expected={e:F8} actual={a:F8} "
                    + $"abs={absDiff:E3} rel={relDiff:E3} "
                    + $"(tol abs={AbsTol:E0} rel={RelTol:E0})");
            }
        }
    }
}
