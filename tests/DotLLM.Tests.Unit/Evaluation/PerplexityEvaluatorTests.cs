using DotLLM.Core.Evaluation;
using DotLLM.Engine.Evaluation;
using Xunit;

namespace DotLLM.Tests.Unit.Evaluation;

public sealed class PerplexityEvaluatorTests
{
    // Vocab must exceed every token id used as a scoring target.
    private const int Vocab = 64;
    private static readonly int[] Tokens = Enumerable.Range(0, 32).ToArray();

    [Fact]
    public void TeacherForced_AllRowsBackend_UniformLogitsGivesVocabSizePerplexity()
    {
        // Uniform logits => P(any target) = 1/vocab => perplexity == vocab, exactly.
        using var model = new FakePerplexityModel(
            vocabSize: Vocab, maxContextLength: 64, returnsAllRows: true, FakePerplexityModel.Uniform);

        var result = PerplexityEvaluator.Evaluate(
            model, Tokens, new PerplexityOptions(PerplexityMode.TeacherForced, ContextLength: 32, Stride: 32));

        Assert.Equal(Vocab, result.Perplexity, 9);
        Assert.Equal(31, result.ScoredTokens);   // n-1 targets from one pass
    }

    [Fact]
    public void TeacherForced_AllRowsBackend_UsesASingleForwardPass()
    {
        using var model = new FakePerplexityModel(Vocab, 64, returnsAllRows: true, FakePerplexityModel.Uniform);

        PerplexityEvaluator.Evaluate(
            model, Tokens, new PerplexityOptions(PerplexityMode.TeacherForced, 32, 32));

        // The whole point of ReturnsAllRows: one pass scores every target.
        Assert.Single(model.ForwardCalls);
        Assert.Equal(32, model.ForwardCalls[0].Length);
    }

    [Fact]
    public void MeanNll_AndPerplexity_AreConsistent()
    {
        using var model = new FakePerplexityModel(Vocab, 64, returnsAllRows: true, FakePerplexityModel.Uniform);

        var result = PerplexityEvaluator.Evaluate(
            model, Tokens, new PerplexityOptions(PerplexityMode.TeacherForced, 32, 32));

        Assert.Equal(result.Perplexity, Math.Exp(result.MeanNegativeLogLikelihood), 9);
    }

    [Fact]
    public void TeacherForced_LastRowOnlyBackend_MatchesAllRowsBackendExactly()
    {
        // Position-dependent but deterministic logits, so a wrong row/position mapping shows up.
        static float[] Rows(int position, int vocab)
        {
            var row = new float[vocab];
            for (int j = 0; j < vocab; j++) row[j] = (float)Math.Sin((position + 1) * (j + 1) * 0.37);
            return row;
        }

        using var allRows = new FakePerplexityModel(Vocab, 64, returnsAllRows: true, Rows);
        using var lastRow = new FakePerplexityModel(Vocab, 64, returnsAllRows: false, Rows);
        var options = new PerplexityOptions(PerplexityMode.TeacherForced, 32, 32);

        var a = PerplexityEvaluator.Evaluate(allRows, Tokens, options);
        var b = PerplexityEvaluator.Evaluate(lastRow, Tokens, options);

        Assert.Equal(a.Perplexity, b.Perplexity, 9);
        Assert.Equal(a.ScoredTokens, b.ScoredTokens);
    }

    [Fact]
    public void TeacherForced_LastRowOnlyBackend_ReprefixesGrowingWindows()
    {
        using var model = new FakePerplexityModel(Vocab, 64, returnsAllRows: false, FakePerplexityModel.Uniform);

        PerplexityEvaluator.Evaluate(
            model, Tokens, new PerplexityOptions(PerplexityMode.TeacherForced, 32, 32));

        // One forward per scored target, each one token longer than the last.
        Assert.Equal(31, model.ForwardCalls.Count);
        for (int i = 0; i < model.ForwardCalls.Count; i++)
            Assert.Equal(i + 1, model.ForwardCalls[i].Length);
    }

    [Fact]
    public void SlidingWindow_TilesScoredTokensWithoutGapsOrOverlap()
    {
        var tokens = Enumerable.Range(0, 40).ToArray();
        using var model = new FakePerplexityModel(Vocab, 64, returnsAllRows: true, FakePerplexityModel.Uniform);

        // L=16, S=8 => windows start at 0, 8, 16, 24; each scores its last 8 targets.
        var result = PerplexityEvaluator.Evaluate(
            model, tokens, new PerplexityOptions(PerplexityMode.SlidingWindow, ContextLength: 16, Stride: 8));

        Assert.Equal(4, result.WindowCount);
        Assert.Equal(32, result.ScoredTokens);        // 4 windows x 8 targets
        Assert.Equal(Vocab, result.Perplexity, 9);    // uniform logits

        Assert.Equal(4, model.ForwardCalls.Count);
        Assert.All(model.ForwardCalls, w => Assert.Equal(16, w.Length));
        Assert.Equal(0, model.ForwardCalls[0][0]);
        Assert.Equal(8, model.ForwardCalls[1][0]);
        Assert.Equal(16, model.ForwardCalls[2][0]);
        Assert.Equal(24, model.ForwardCalls[3][0]);
    }

    [Fact]
    public void SlidingWindow_EvaluatesEachWindowAsAnIndependentSequenceFromPositionZero()
    {
        // llama.cpp evaluates every chunk as a fresh sequence with positions restarting at 0.
        // That is also what allows a corpus longer than the model's max sequence length to be
        // scored at all -- absolute positions would run past it and throw.
        var seenPositions = new List<int>();

        float[] Rows(int position, int vocab)
        {
            seenPositions.Add(position);
            var row = new float[vocab];
            row[0] = 1f;
            return row;
        }

        var tokens = new int[40];
        using var model = new FakePerplexityModel(Vocab, maxContextLength: 16, returnsAllRows: true, Rows);

        PerplexityEvaluator.Evaluate(
            model, tokens, new PerplexityOptions(PerplexityMode.SlidingWindow, 16, Stride: 8));

        // Never a position at or beyond the window length, however far into the corpus we are.
        Assert.All(seenPositions, p => Assert.InRange(p, 0, 15));
        Assert.Contains(0, seenPositions);
    }

    [Fact]
    public void SlidingWindow_ScoresTargetsFromTheCorrectRow()
    {
        // Row i of a window predicts the window's token i+1. A row/target off-by-one shows up as
        // a confident model scoring badly.
        static float[] Rows(int position, int vocab)
        {
            var row = new float[vocab];
            row[(position + 1) % vocab] = 20f;   // argmax == the id this row should predict
            return row;
        }

        // Window-relative: token at window offset j has id j % Vocab, so row j's target is
        // (j+1) % Vocab -- exactly the argmax above. Stride == context keeps windows aligned to
        // the same offsets, so this holds for every window.
        var tokens = new int[64];
        for (int i = 0; i < tokens.Length; i++) tokens[i] = (i % 16) % Vocab;

        using var model = new FakePerplexityModel(Vocab, 16, returnsAllRows: true, Rows);
        var result = PerplexityEvaluator.Evaluate(
            model, tokens, new PerplexityOptions(PerplexityMode.SlidingWindow, 16, Stride: 16, UnscoredPrefix: 8));

        Assert.True(result.MeanNegativeLogLikelihood < 0.01,
            $"expected confident predictions, got mean NLL {result.MeanNegativeLogLikelihood}");
    }

    [Fact]
    public void SlidingWindow_RejectsUnscoredPrefixLeavingNothingToScore()
    {
        using var model = new FakePerplexityModel(Vocab, 64, returnsAllRows: true, FakePerplexityModel.Uniform);
        Assert.Throws<ArgumentException>(() => PerplexityEvaluator.Evaluate(
            model, Tokens,
            new PerplexityOptions(PerplexityMode.SlidingWindow, ContextLength: 16, Stride: 16, UnscoredPrefix: 16)));
    }

    [Fact]
    public void LlamaCppDefault_UsesNonOverlappingChunksScoringTheSecondHalf()
    {
        var tokens = Enumerable.Range(0, 40).ToArray();
        using var model = new FakePerplexityModel(Vocab, 64, returnsAllRows: true, FakePerplexityModel.Uniform);

        // L=16 => chunks at 0, 16 (32 would need tokens up to 48). Each scores 7, not 8:
        // llama.cpp's count is n_ctx - n_ctx/2 - 1.
        var result = PerplexityEvaluator.Evaluate(
            model, tokens, PerplexityOptions.LlamaCppDefault(contextLength: 16));

        Assert.Equal(2, result.WindowCount);
        Assert.Equal(14, result.ScoredTokens);   // 2 chunks x 7 scored
        Assert.Equal(2, model.ForwardCalls.Count);
        Assert.Equal(0, model.ForwardCalls[0][0]);
        Assert.Equal(16, model.ForwardCalls[1][0]);   // advances by the FULL window, not by 8
    }

    [Fact]
    public void LlamaCppDefault_ScoresOneFewerTargetThanHalfTheWindow()
    {
        // llama.cpp: first = n_ctx/2, then `count += n_ctx - first - 1`, and the target for row j is
        // token j+1 — so the token AT n_ctx/2 is context only and is never scored. Scoring it too
        // would give n_ctx/2 targets per chunk instead of n_ctx/2 - 1: the same name, a different
        // measurement, and a figure that is not comparable to a published one.
        foreach (int context in new[] { 16, 64, 512 })
        {
            var options = PerplexityOptions.LlamaCppDefault(context);
            Assert.Equal(context / 2 + 1, options.UnscoredPrefix);
            Assert.Equal(context / 2 - 1, context - options.UnscoredPrefix);
        }
    }

    [Fact]
    public void StandardError_MatchesTheSampleVarianceOfPerTokenNll()
    {
        // Every target is token 0, and every row's only non-zero logit is at index 0 — so the
        // target is always the argmax and its NLL depends solely on the row's peak height. That
        // makes the whole set of per-token NLLs predictable in closed form, without reaching
        // into the evaluator or replaying tensors.
        static float[] Rows(int position, int vocab)
        {
            var row = new float[vocab];
            row[0] = position % 2 == 0 ? 2f : 0f;
            return row;
        }

        static double Nll(int position)
        {
            double peak = position % 2 == 0 ? 2.0 : 0.0;
            return Math.Log(Math.Exp(peak) + (Vocab - 1)) - peak;
        }

        var tokens = new int[40];   // all zero
        using var model = new FakePerplexityModel(Vocab, 64, returnsAllRows: true, Rows);

        var result = PerplexityEvaluator.Evaluate(model, tokens, PerplexityOptions.LlamaCppDefault(16));

        // L=16 => windows at 0 and 16; prefix 9 scores targets [s+9, s+16), i.e. rows 8..14.
        var expectedNlls = new List<double>();
        for (int window = 0; window < 2; window++)
            for (int row = 8; row <= 14; row++)
                expectedNlls.Add(Nll(row));

        double mean = expectedNlls.Average();
        double variance = expectedNlls.Sum(v => (v - mean) * (v - mean)) / expectedNlls.Count;
        double expected = Math.Sqrt(variance / (expectedNlls.Count - 1)) * Math.Exp(mean);

        Assert.Equal(expectedNlls.Count, result.ScoredTokens);
        Assert.Equal(mean, result.MeanNegativeLogLikelihood, 9);
        Assert.Equal(expected, result.StandardError, 9);
        Assert.True(result.StandardError > 0);
    }

    [Fact]
    public void StandardError_IsZeroWhenEveryScoredTokenHasTheSameNll()
    {
        // A uniform distribution gives every target an identical NLL: zero variance, so the error
        // bar must be exactly 0 rather than a NaN out of a negative rounding residue.
        var tokens = Enumerable.Range(0, 40).ToArray();
        using var model = new FakePerplexityModel(Vocab, 64, returnsAllRows: true, FakePerplexityModel.Uniform);

        var result = PerplexityEvaluator.Evaluate(model, tokens, PerplexityOptions.LlamaCppDefault(16));

        Assert.Equal(0, result.StandardError);
    }


    [Fact]
    public void LlamaCppDefault_AndContiguousTiling_ScoreDifferentTokenSets()
    {
        // The bug this guards: both schemes score the same COUNT, so a count check alone cannot
        // distinguish them. Only a position-dependent signal reveals the different token sets.
        // Row at position p predicts token p+1. Give the correct target a confidence that VARIES
        // with p, so the mean NLL depends on which positions were scored, not merely how many.
        static float[] Rows(int position, int vocab)
        {
            var row = new float[vocab];
            row[(position + 1) % vocab] = 1f + (position % 7);
            return row;
        }

        var tokens = new int[64];
        for (int i = 0; i < tokens.Length; i++) tokens[i] = i % Vocab;

        using var chunked = new FakePerplexityModel(Vocab, 64, returnsAllRows: true, Rows);
        using var tiled = new FakePerplexityModel(Vocab, 64, returnsAllRows: true, Rows);

        var a = PerplexityEvaluator.Evaluate(chunked, tokens, PerplexityOptions.LlamaCppDefault(16));
        var b = PerplexityEvaluator.Evaluate(
            tiled, tokens, new PerplexityOptions(PerplexityMode.SlidingWindow, 16, Stride: 8));

        Assert.NotEqual(a.Perplexity, b.Perplexity, 6);
    }
}
