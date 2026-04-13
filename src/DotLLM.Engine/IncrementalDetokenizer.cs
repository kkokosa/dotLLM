using System.Runtime.InteropServices;
using System.Text;
using DotLLM.Tokenizers;

namespace DotLLM.Engine;

/// <summary>
/// Incrementally detokenizes a generated token stream in O(1) amortized time per token.
/// </summary>
/// <remarks>
/// <para>
/// The naive pattern <c>_tokenizer.Decode(generatedIds)</c> called on every decode step is
/// O(n²) in generation length. This type keeps a small sliding window of the most recent tokens
/// and a committed <see cref="StringBuilder"/> for everything before the window; each
/// <see cref="Append(int)"/> decodes only the window (bounded-size) and promotes fully-stable
/// prefix text into the committed buffer.
/// </para>
/// <para>
/// Correctness with partial UTF-8 / SentencePiece byte-tokens: we only evict a token when
/// <c>Decode(window[1..])</c> is a suffix of <c>Decode(window)</c>. When the tail differs —
/// usually because the oldest token contributed bytes to a multi-byte UTF-8 codepoint or an
/// accumulating byte-token run — we leave the window alone and try again after the next token.
/// A hard cap prevents unbounded growth in pathological cases by force-committing the current
/// window text.
/// </para>
/// </remarks>
internal sealed class IncrementalDetokenizer
{
    private const int SoftWindowLimit = 4;
    private const int HardWindowLimit = 32;

    private readonly ITokenizer _tokenizer;
    private readonly StringBuilder _committed;
    private readonly List<int> _window;
    private string _windowText;
    private int _deltaBaseline;

    public IncrementalDetokenizer(ITokenizer tokenizer, int initialCapacity = 1024)
    {
        _tokenizer = tokenizer;
        _committed = new StringBuilder(initialCapacity);
        _window = new List<int>(HardWindowLimit + 1);
        _windowText = string.Empty;
    }

    /// <summary>Total number of decoded characters (committed + window).</summary>
    public int Length => _committed.Length + _windowText.Length;

    /// <summary>Adds a token and advances the decoded state. Amortized O(1) per call.</summary>
    public void Append(int tokenId)
    {
        _window.Add(tokenId);
        _windowText = _tokenizer.Decode(CollectionsMarshal.AsSpan(_window), stripBosSpace: false);

        while (_window.Count > SoftWindowLimit)
        {
            if (TryEvictOldest())
                continue;

            if (_window.Count > HardWindowLimit)
            {
                // Pathological case: leading byte-token run that never resolves cleanly.
                // Force-commit to prevent unbounded memory growth. Rare by construction.
                _committed.Append(_windowText);
                _window.Clear();
                _windowText = string.Empty;
            }
            break;
        }
    }

    private bool TryEvictOldest()
    {
        var tail = _tokenizer.Decode(
            CollectionsMarshal.AsSpan(_window).Slice(1), stripBosSpace: false);

        if (tail.Length > _windowText.Length)
            return false;

        var windowSpan = _windowText.AsSpan();
        if (!windowSpan[(windowSpan.Length - tail.Length)..].SequenceEqual(tail))
            return false;

        int commitLen = _windowText.Length - tail.Length;
        if (commitLen > 0)
            _committed.Append(_windowText, 0, commitLen);
        _window.RemoveAt(0);
        _windowText = tail;
        return true;
    }

    /// <summary>
    /// Returns a tail view over the last <paramref name="maxChars"/> characters of the decoded text.
    /// The view aliases <see cref="_windowText"/> when possible (zero allocation), otherwise writes
    /// into <paramref name="scratch"/>.
    /// </summary>
    /// <param name="maxChars">Maximum number of trailing characters to expose.</param>
    /// <param name="scratch">Scratch buffer used when the tail spans the committed/window boundary.
    /// Must be at least <c>min(maxChars, Length)</c> in length.</param>
    public ReadOnlySpan<char> GetTailView(int maxChars, Span<char> scratch)
    {
        int total = Length;
        int take = Math.Min(maxChars, total);
        if (take == 0)
            return default;

        if (take <= _windowText.Length)
            return _windowText.AsSpan(_windowText.Length - take);

        if (scratch.Length < take)
            throw new ArgumentException("Scratch buffer too small for requested tail.", nameof(scratch));

        int fromWindow = _windowText.Length;
        int fromCommitted = take - fromWindow;
        int committedStart = _committed.Length - fromCommitted;
        _committed.CopyTo(committedStart, scratch[..fromCommitted], fromCommitted);
        _windowText.AsSpan().CopyTo(scratch.Slice(fromCommitted, fromWindow));
        return scratch[..take];
    }

    /// <summary>
    /// Returns the text appended since the previous call to <see cref="TakeDelta"/>
    /// (or since construction). Advances the baseline to the current length.
    /// </summary>
    public string TakeDelta()
    {
        int total = Length;
        if (total == _deltaBaseline)
            return string.Empty;

        string delta = SliceRange(_deltaBaseline, total);
        _deltaBaseline = total;
        return delta;
    }

    /// <summary>Materializes the full decoded text. O(<see cref="Length"/>). Use sparingly.</summary>
    public override string ToString()
    {
        if (_committed.Length == 0)
            return _windowText;
        if (_windowText.Length == 0)
            return _committed.ToString();
        return string.Create(
            _committed.Length + _windowText.Length,
            this,
            static (span, self) =>
            {
                self._committed.CopyTo(0, span[..self._committed.Length], self._committed.Length);
                self._windowText.AsSpan().CopyTo(span[self._committed.Length..]);
            });
    }

    private string SliceRange(int start, int endExclusive)
    {
        int length = endExclusive - start;
        if (length == 0) return string.Empty;

        int committedLen = _committed.Length;

        if (start >= committedLen)
            return _windowText.Substring(start - committedLen, length);

        if (endExclusive <= committedLen)
            return _committed.ToString(start, length);

        // Spans committed/window boundary.
        int fromCommitted = committedLen - start;
        int fromWindow = length - fromCommitted;
        return string.Create(length, (self: this, start, fromCommitted, fromWindow),
            static (span, s) =>
            {
                s.self._committed.CopyTo(s.start, span[..s.fromCommitted], s.fromCommitted);
                s.self._windowText.AsSpan(0, s.fromWindow).CopyTo(span[s.fromCommitted..]);
            });
    }
}
