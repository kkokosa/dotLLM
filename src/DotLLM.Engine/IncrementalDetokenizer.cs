using System.Buffers;
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
/// <para>
/// Zero-allocation hot path: window decode and tail decode both go through
/// <see cref="ITokenizer.TryDecode"/> into <see cref="ArrayPool{T}.Shared"/>-rented char buffers
/// that grow-and-re-rent on overflow. <see cref="Dispose"/> returns the buffers; callers should
/// hold the instance in a <c>using</c> or otherwise invoke <see cref="Dispose"/> deterministically.
/// </para>
/// </remarks>
internal sealed class IncrementalDetokenizer : IDisposable
{
    private const int SoftWindowLimit = 4;
    private const int HardWindowLimit = 32;
    private const int InitialWindowBufSize = 64;

    private readonly ITokenizer _tokenizer;
    private readonly StringBuilder _committed;
    private readonly List<int> _window;
    private char[] _windowBuf;
    private int _windowLen;
    private char[] _tailBuf;
    private int _deltaBaseline;

    public IncrementalDetokenizer(ITokenizer tokenizer, int initialCapacity = 1024)
    {
        _tokenizer = tokenizer;
        _committed = new StringBuilder(initialCapacity);
        _window = new List<int>(HardWindowLimit + 1);
        _windowBuf = ArrayPool<char>.Shared.Rent(InitialWindowBufSize);
        _tailBuf = ArrayPool<char>.Shared.Rent(InitialWindowBufSize);
        _windowLen = 0;
    }

    /// <summary>Total number of decoded characters (committed + window).</summary>
    public int Length => _committed.Length + _windowLen;

    /// <summary>Adds a token and advances the decoded state. Amortized O(1) per call.</summary>
    public void Append(int tokenId)
    {
        _window.Add(tokenId);
        DecodeWindowInto(ref _windowBuf, out _windowLen);

        while (_window.Count > SoftWindowLimit)
        {
            if (TryEvictOldest())
                continue;

            if (_window.Count > HardWindowLimit)
            {
                // Pathological case: leading byte-token run that never resolves cleanly.
                // Force-commit to prevent unbounded memory growth. Rare by construction.
                _committed.Append(_windowBuf, 0, _windowLen);
                _window.Clear();
                _windowLen = 0;
            }
            break;
        }
    }

    private void DecodeWindowInto(ref char[] buffer, out int written)
    {
        var ids = CollectionsMarshal.AsSpan(_window);
        while (true)
        {
            if (_tokenizer.TryDecode(ids, stripBosSpace: false, buffer, out written))
                return;
            // Overflow: grow and retry. ArrayPool.Rent rounds up to a power of two,
            // so growth converges quickly even on multi-byte glyph runs.
            char[] larger = ArrayPool<char>.Shared.Rent(buffer.Length * 2);
            ArrayPool<char>.Shared.Return(buffer);
            buffer = larger;
        }
    }

    private void DecodeTailInto(out int written)
    {
        var ids = CollectionsMarshal.AsSpan(_window).Slice(1);
        while (true)
        {
            if (_tokenizer.TryDecode(ids, stripBosSpace: false, _tailBuf, out written))
                return;
            char[] larger = ArrayPool<char>.Shared.Rent(_tailBuf.Length * 2);
            ArrayPool<char>.Shared.Return(_tailBuf);
            _tailBuf = larger;
        }
    }

    private bool TryEvictOldest()
    {
        DecodeTailInto(out int tailLen);

        if (tailLen > _windowLen)
            return false;

        ReadOnlySpan<char> windowSpan = _windowBuf.AsSpan(0, _windowLen);
        ReadOnlySpan<char> tailSpan = _tailBuf.AsSpan(0, tailLen);
        if (!windowSpan[(windowSpan.Length - tailLen)..].SequenceEqual(tailSpan))
            return false;

        int commitLen = _windowLen - tailLen;
        if (commitLen > 0)
            _committed.Append(_windowBuf, 0, commitLen);
        _window.RemoveAt(0);
        // Shift the surviving tail to the front of _windowBuf so subsequent appends see
        // [0, _windowLen) as the current decoded window.
        if (tailLen > 0)
            _tailBuf.AsSpan(0, tailLen).CopyTo(_windowBuf);
        _windowLen = tailLen;
        return true;
    }

    /// <summary>
    /// Returns a tail view over the last <paramref name="maxChars"/> characters of the decoded text.
    /// The view aliases the window buffer when possible (zero allocation), otherwise writes
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

        if (take <= _windowLen)
            return _windowBuf.AsSpan(_windowLen - take, take);

        if (scratch.Length < take)
            throw new ArgumentException("Scratch buffer too small for requested tail.", nameof(scratch));

        int fromWindow = _windowLen;
        int fromCommitted = take - fromWindow;
        int committedStart = _committed.Length - fromCommitted;
        _committed.CopyTo(committedStart, scratch[..fromCommitted], fromCommitted);
        _windowBuf.AsSpan(0, fromWindow).CopyTo(scratch.Slice(fromCommitted, fromWindow));
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
            return _windowLen == 0 ? string.Empty : new string(_windowBuf, 0, _windowLen);
        if (_windowLen == 0)
            return _committed.ToString();
        return string.Create(
            _committed.Length + _windowLen,
            this,
            static (span, self) =>
            {
                self._committed.CopyTo(0, span[..self._committed.Length], self._committed.Length);
                self._windowBuf.AsSpan(0, self._windowLen).CopyTo(span[self._committed.Length..]);
            });
    }

    /// <summary>Returns pooled buffers to <see cref="ArrayPool{T}.Shared"/>. Idempotent.</summary>
    public void Dispose()
    {
        if (_windowBuf is { Length: > 0 } wb)
        {
            ArrayPool<char>.Shared.Return(wb);
            _windowBuf = [];
        }
        if (_tailBuf is { Length: > 0 } tb)
        {
            ArrayPool<char>.Shared.Return(tb);
            _tailBuf = [];
        }
        _windowLen = 0;
    }

    private string SliceRange(int start, int endExclusive)
    {
        int length = endExclusive - start;
        if (length == 0) return string.Empty;

        int committedLen = _committed.Length;

        if (start >= committedLen)
            return new string(_windowBuf, start - committedLen, length);

        if (endExclusive <= committedLen)
            return _committed.ToString(start, length);

        // Spans committed/window boundary.
        int fromCommitted = committedLen - start;
        int fromWindow = length - fromCommitted;
        return string.Create(length, (self: this, start, fromCommitted, fromWindow),
            static (span, s) =>
            {
                s.self._committed.CopyTo(s.start, span[..s.fromCommitted], s.fromCommitted);
                s.self._windowBuf.AsSpan(0, s.fromWindow).CopyTo(span[s.fromCommitted..]);
            });
    }
}
