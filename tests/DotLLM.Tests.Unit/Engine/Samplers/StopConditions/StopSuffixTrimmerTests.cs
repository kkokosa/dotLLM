using System.Collections.Generic;
using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers.StopConditions;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers.StopConditions;

/// <summary>
/// Regression tests for <see cref="StopSuffixTrimmer"/>. The bug it replaces:
/// when a <see cref="StopStringCondition"/> matched, <c>TextGenerator</c> removed
/// the entire last token from the generated-id list. If the stop string was a
/// strict suffix of the last token (e.g. token <c>"ld&lt;|im_end|&gt;"</c>,
/// stop <c>"&lt;|im_end|&gt;"</c>) the user lost the <c>"ld"</c> prefix and got
/// <c>"Hello, wor"</c> instead of <c>"Hello, world"</c>. See upstream issue #107
/// item 1.
/// </summary>
public class StopSuffixTrimmerTests
{
    /// <summary>
    /// Headline test: stop string is a STRICT SUFFIX of the last token's decoded
    /// text. The trimmer must preserve the token-prefix that precedes the stop
    /// string. A test that uses a stop string equal to the whole token does not
    /// discriminate, since the buggy code happens to do the right thing in that
    /// degenerate case.
    /// </summary>
    [Fact]
    public void TrimMatchedSuffix_StopStringIsStrictSuffixOfLastToken_PreservesPrefix()
    {
        // Simulates the issue's example: full decoded text where the last token
        // contributed "ld<|im_end|>" — the stop string is just "<|im_end|>".
        const string fullText = "Hello, world<|im_end|>";
        var conditions = new List<IStopCondition>
        {
            new StopStringCondition("<|im_end|>"),
        };

        string trimmed = StopSuffixTrimmer.TrimMatchedSuffix(fullText, conditions);

        Assert.Equal("Hello, world", trimmed);
    }

    /// <summary>
    /// Ordering discriminator: an EOS condition registered BEFORE the matching
    /// <see cref="StopStringCondition"/>. <c>TextGenerator</c> decides whether to keep
    /// the last token via <see cref="StopSuffixTrimmer.MatchedSuffixLength"/> over all
    /// conditions, not via the type of the first condition to return <c>Stop</c> — with
    /// the latter, this registration order would drop the whole last token and lose the
    /// <c>"ld"</c> prefix again. A list with the stop-string condition first does not
    /// discriminate, since both formulations agree there.
    /// </summary>
    [Fact]
    public void MatchedSuffixLength_NonStopStringConditionListedFirst_StillMatches()
    {
        const string fullText = "Hello, world<|im_end|>";
        var conditions = new List<IStopCondition>
        {
            new EosStopCondition(eosTokenId: 2),
            new MaxTokensStopCondition(maxTokens: 100),
            new StopStringCondition("<|im_end|>"),
        };

        Assert.Equal("<|im_end|>".Length, StopSuffixTrimmer.MatchedSuffixLength(fullText, conditions));
        Assert.Equal("Hello, world", StopSuffixTrimmer.TrimMatchedSuffix(fullText, conditions));
    }

    /// <summary>
    /// Reports the matched-suffix character count for callers that need it.
    /// </summary>
    [Fact]
    public void MatchedSuffixLength_StopStringSuffix_ReturnsStopStringLength()
    {
        const string fullText = "Hello, world<|im_end|>";
        var conditions = new List<IStopCondition>
        {
            new StopStringCondition("<|im_end|>"),
        };

        int len = StopSuffixTrimmer.MatchedSuffixLength(fullText, conditions);

        Assert.Equal("<|im_end|>".Length, len);
    }

    /// <summary>
    /// When multiple stop strings match the same tail, the longest match wins —
    /// otherwise nested patterns such as <c>"&lt;|end|&gt;"</c> vs
    /// <c>"&lt;|im_end|&gt;"</c> would leave the longer pattern's leading chars
    /// in the output.
    /// </summary>
    [Fact]
    public void TrimMatchedSuffix_MultipleMatches_TrimsLongest()
    {
        const string fullText = "Done<|im_end|>";
        var conditions = new List<IStopCondition>
        {
            new StopStringCondition("<|end|>"), // not a suffix
            new StopStringCondition("<|im_end|>"), // matches, longer
            new StopStringCondition("end|>"), // suffix but shorter
        };

        string trimmed = StopSuffixTrimmer.TrimMatchedSuffix(fullText, conditions);

        Assert.Equal("Done", trimmed);
    }

    /// <summary>
    /// When no stop string is a suffix of the text the original text is returned
    /// unchanged (defensive — caller may invoke with finishReason == Stop for
    /// reasons other than a stop-string match, e.g. EOS token).
    /// </summary>
    [Fact]
    public void TrimMatchedSuffix_NoMatch_ReturnsOriginal()
    {
        const string fullText = "Hello world";
        var conditions = new List<IStopCondition>
        {
            new StopStringCondition("<|im_end|>"),
        };

        string trimmed = StopSuffixTrimmer.TrimMatchedSuffix(fullText, conditions);

        Assert.Equal(fullText, trimmed);
        // Reference equality is asserted on purpose, not as an accident of the current
        // implementation: the no-match path is the common case (every EOS / max-tokens
        // stop reaches it), so returning the input instance rather than re-allocating an
        // identical string is part of the contract. If a refactor breaks this, the trim
        // has started allocating on a hot path and should be revisited, not the test.
        Assert.Same(fullText, trimmed);
    }

    /// <summary>
    /// Non-stop-string conditions (EOS, MaxTokens) are ignored by the trimmer —
    /// only <see cref="StopStringCondition"/> contributes a suffix-match length.
    /// </summary>
    [Fact]
    public void TrimMatchedSuffix_NonStopStringConditions_AreIgnored()
    {
        const string fullText = "Hello, world<|im_end|>";
        var conditions = new List<IStopCondition>
        {
            new EosStopCondition(eosTokenId: 2),
            new MaxTokensStopCondition(maxTokens: 100),
        };

        string trimmed = StopSuffixTrimmer.TrimMatchedSuffix(fullText, conditions);

        Assert.Equal(fullText, trimmed);
    }

    /// <summary>
    /// Trimmer never leaves a dangling high surrogate at the end of the output —
    /// if the trim would land between a surrogate pair, it extends to include
    /// the high half.
    /// </summary>
    [Fact]
    public void TrimMatchedSuffix_TrimWouldSplitSurrogatePair_TrimsBothHalves()
    {
        // U+1F600 (😀) is a surrogate pair. Imagine a stop string that happens to
        // start with the low surrogate — the trim would otherwise land between
        // the high and low halves of the emoji, leaving an invalid UTF-16 result.
        string fullText = "abc😀STOP";
        var conditions = new List<IStopCondition>
        {
            new StopStringCondition("\uDE00STOP"), // low-surrogate + STOP — 5 chars
        };

        string trimmed = StopSuffixTrimmer.TrimMatchedSuffix(fullText, conditions);

        Assert.Equal("abc", trimmed);
    }
}
