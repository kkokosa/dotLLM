using System;
using System.Collections.Generic;
using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers.StopConditions;

/// <summary>
/// Character-level helper for handling stop-string matches that overlap only the
/// suffix of the last generated token.
/// </summary>
/// <remarks>
/// <para>
/// Before this helper existed, <c>TextGenerator</c> handled a <see cref="StopStringCondition"/>
/// match by removing the last token from the generated-id list entirely. That over-trimmed
/// when the stop string was a strict suffix of the last token's decoded text — e.g. when
/// the last token decoded to <c>"ld&lt;|im_end|&gt;"</c> and the stop string was
/// <c>"&lt;|im_end|&gt;"</c>, the user lost the <c>"ld"</c> prefix of the token.
/// </para>
/// <para>
/// The correct behaviour is to keep every token in the id list and instead trim the
/// stop-string suffix from the <em>decoded text</em> at the character level. This helper
/// performs that suffix match against the full set of registered stop strings, returning
/// the longest matching length (so overlapping or nested stop strings — e.g.
/// <c>"&lt;|end|&gt;"</c> vs <c>"&lt;|im_end|&gt;"</c> — trim correctly).
/// </para>
/// </remarks>
internal static class StopSuffixTrimmer
{
    /// <summary>
    /// Finds the longest stop string from <paramref name="conditions"/> that is a suffix
    /// of <paramref name="text"/> and returns its character count. Returns 0 when no stop
    /// string is a suffix of the text.
    /// </summary>
    /// <param name="text">The fully decoded generated text.</param>
    /// <param name="conditions">All registered stop conditions. Only
    /// <see cref="StopStringCondition"/> entries participate; others are ignored.</param>
    /// <returns>The character length of the longest matching stop-string suffix, or 0.</returns>
    public static int MatchedSuffixLength(ReadOnlySpan<char> text, IReadOnlyList<IStopCondition> conditions)
    {
        int longest = 0;
        for (int i = 0; i < conditions.Count; i++)
        {
            if (conditions[i] is StopStringCondition ssc)
            {
                string stop = ssc.StopString;
                if (stop.Length == 0 || stop.Length > text.Length) continue;
                if (text.EndsWith(stop.AsSpan(), StringComparison.Ordinal) && stop.Length > longest)
                {
                    longest = stop.Length;
                }
            }
        }
        return longest;
    }

    /// <summary>
    /// Returns <paramref name="text"/> with the longest matching stop-string suffix removed
    /// at the character boundary. Defensive against malformed UTF-16: if the trim point
    /// would land inside a surrogate pair, the trim is extended to include the high
    /// surrogate so the result is always valid UTF-16.
    /// </summary>
    /// <param name="text">The fully decoded generated text.</param>
    /// <param name="conditions">All registered stop conditions.</param>
    /// <returns>Trimmed text. Returns <paramref name="text"/> unchanged when no stop string matches.</returns>
    public static string TrimMatchedSuffix(string text, IReadOnlyList<IStopCondition> conditions)
    {
        int len = MatchedSuffixLength(text.AsSpan(), conditions);
        if (len == 0) return text;

        int end = text.Length - len;
        // If the trim boundary lands between a high+low surrogate pair, extend the
        // trim so the result remains valid UTF-16 — never leave a dangling high
        // surrogate at the end of the output.
        if (end > 0 && char.IsHighSurrogate(text[end - 1]) && end < text.Length && char.IsLowSurrogate(text[end]))
        {
            end--;
        }
        return text.Substring(0, end);
    }
}
