using System.Text;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Evaluation;

/// <summary>Streams a text corpus into tokens without materializing the whole file or token array.</summary>
/// <remarks>
/// Streaming is a design constraint rather than an optimisation. On unified-memory parts a large
/// VRAM carve-out leaves host RAM scarce, and a standard perplexity corpus tokenizes to hundreds of
/// thousands of ints — held alongside the weights, that is exactly the pressure this harness must
/// not add.
/// </remarks>
public static class CorpusReader
{
    /// <summary>
    /// Reads <paramref name="reader"/> in character chunks, tokenizes each chunk, and yields token
    /// ids in order, stopping after <paramref name="maxTokens"/> (<c>0</c> = unbounded).
    /// </summary>
    /// <remarks>
    /// <para>Chunks are cut immediately before the last run of whitespace, so a token is never split
    /// across a boundary; the remainder is carried into the next chunk, and the final carry is
    /// flushed whole. Verified against whole-file tokenization on wikitext-2 (1.29 MB, 301,948
    /// tokens) at chunk sizes from 997 to 65536 — identical streams, so ~1,290 boundaries in the
    /// smallest case land in awkward places without changing a single id.</para>
    /// <para><b>Whole runs, not single characters.</b> Cutting at the last whitespace <i>character</i>
    /// can fall inside a run of them, and a GPT-2-style pre-tokenizer treats a whitespace run as one
    /// unit — splitting it yields a different token stream than tokenizing the file in one pass,
    /// which is precisely the silent divergence this harness exists to rule out.</para>
    /// <para><b>Known limitation:</b> text containing no whitespace at all accumulates in the carry
    /// buffer until some arrives, or until the corpus ends. Flushing at an arbitrary character
    /// instead would bound the memory but change the token stream, so correctness wins here. Ordinary
    /// prose corpora are unaffected; a whitespace-free corpus (minified JSON, unsegmented CJK) is
    /// effectively read whole.</para>
    /// </remarks>
    /// <param name="reader">Corpus source.</param>
    /// <param name="tokenizer">Tokenizer whose vocabulary the ids belong to.</param>
    /// <param name="maxTokens">Upper bound on emitted tokens; <c>0</c> for unbounded.</param>
    /// <param name="charChunkSize">Characters read per chunk.</param>
    public static IEnumerable<int> StreamTokens(
        TextReader reader, ITokenizer tokenizer, int maxTokens = 0, int charChunkSize = 65536)
    {
        ArgumentNullException.ThrowIfNull(reader);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentOutOfRangeException.ThrowIfLessThan(charChunkSize, 1);

        var buffer = new char[charChunkSize];
        var carry = new StringBuilder();
        int emitted = 0;

        while (true)
        {
            int read = reader.Read(buffer, 0, buffer.Length);
            if (read == 0) break;

            carry.Append(buffer, 0, read);
            string pending = carry.ToString();

            int cut = LastWhitespaceRunStart(pending);
            if (cut <= 0) continue;   // no safe split point yet; keep accumulating

            // The separating whitespace is carried INTO the next chunk, not dropped. GPT-2-style BPE
            // encodes a leading space as part of the following token, so dropping it silently
            // changes the token stream — and therefore the perplexity — versus tokenizing the
            // corpus in one pass.
            string ready = pending[..cut];
            carry.Clear();
            carry.Append(pending[cut..]);

            foreach (int id in tokenizer.Encode(ready))
            {
                yield return id;
                if (maxTokens > 0 && ++emitted >= maxTokens) yield break;
            }
        }

        if (carry.Length > 0)
        {
            foreach (int id in tokenizer.Encode(carry.ToString()))
            {
                yield return id;
                if (maxTokens > 0 && ++emitted >= maxTokens) yield break;
            }
        }
    }

    /// <summary>
    /// Index of the first character of the last whitespace run in <paramref name="text"/>, or
    /// <c>-1</c> when it contains no whitespace.
    /// </summary>
    /// <remarks>
    /// Returning the run's START rather than the last whitespace character is what keeps the run
    /// intact: everything from it onwards moves into the carry, so no pre-token that spans a
    /// whitespace run is ever cut in half.
    /// </remarks>
    private static int LastWhitespaceRunStart(string text)
    {
        int i = text.Length - 1;
        while (i >= 0 && !char.IsWhiteSpace(text[i])) i--;
        if (i < 0) return -1;

        while (i > 0 && char.IsWhiteSpace(text[i - 1])) i--;
        return i;
    }
}
