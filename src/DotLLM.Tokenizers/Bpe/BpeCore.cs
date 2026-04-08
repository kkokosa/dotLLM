using System.Buffers;
using System.Runtime.CompilerServices;
using System.Text;

namespace DotLLM.Tokenizers.Bpe;

/// <summary>
/// Strategy interface for BPE encoding variants (SentencePiece, tiktoken).
/// Implementations are internal; the public entry point is <see cref="BpeTokenizer"/>.
/// </summary>
internal interface IBpeEncoding
{
    int[] Encode(string text);
    /// <summary>
    /// Encodes text as a continuation segment (no BOS space prepend).
    /// Used for text segments that follow a special token during pre-splitting.
    /// Default implementation falls back to <see cref="Encode"/>.
    /// </summary>
    int[] EncodeSegment(string text) => Encode(text);
    string Decode(ReadOnlySpan<int> tokenIds);
    string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
    string DecodeToken(int tokenId);
}

/// <summary>Mutable doubly-linked symbol node used during the BPE merge loop.</summary>
internal struct Symbol
{
    public int Prev;     // index of previous live symbol; -1 = head
    public int Next;     // index of next live symbol; -1 = tail
    public int TokenId;
    public bool Deleted;
}

/// <summary>
/// Immutable bigram candidate queued during the BPE merge loop.
/// Carries expected token IDs so stale entries can be detected and discarded.
/// </summary>
internal readonly struct BgramEntry(int left, int right, int mergedId, int expectedLeft, int expectedRight)
{
    public int Left { get; } = left;
    public int Right { get; } = right;
    public int MergedId { get; } = mergedId;
    /// <summary>TokenId the left symbol must still have for this entry to be valid.</summary>
    public int ExpectedLeft { get; } = expectedLeft;
    /// <summary>TokenId the right symbol must still have for this entry to be valid.</summary>
    public int ExpectedRight { get; } = expectedRight;
}

/// <summary>
/// Shared static utilities used by both <see cref="SentencePieceEncoding"/>
/// and <see cref="Gpt2TiktokenEncoding"/>.
/// </summary>
internal static class BpeCore
{
    /// <summary>Returns true if <paramref name="token"/> is in the <c>&lt;0xNN&gt;</c> byte-literal format.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool TryParseByteLiteral(string token, out byte value)
    {
        if (token.Length == 6
            && token[0] == '<' && token[1] == '0' && token[2] == 'x'
            && token[5] == '>'
            && IsHexDigit(token[3]) && IsHexDigit(token[4]))
        {
            value = (byte)(HexValue(token[3]) << 4 | HexValue(token[4]));
            return true;
        }
        value = 0;
        return false;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool IsByteToken(string token, out byte value) =>
        TryParseByteLiteral(token, out value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool IsHexDigit(char c) =>
        (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int HexValue(char c) =>
        c >= 'a' ? c - 'a' + 10 : c >= 'A' ? c - 'A' + 10 : c - '0';

    /// <summary>Collects non-deleted symbol token IDs into a new array.</summary>
    internal static int[] CollectTokenIds(Symbol[] symbols, int symbolCount)
    {
        int count = 0;
        for (int i = 0; i < symbolCount; i++)
            if (!symbols[i].Deleted) count++;

        int[] result = new int[count];
        int ri = 0;
        for (int i = 0; i < symbolCount; i++)
            if (!symbols[i].Deleted) result[ri++] = symbols[i].TokenId;
        return result;
    }

    /// <summary>Appends non-deleted symbol token IDs to an existing list (zero intermediate allocation).</summary>
    internal static void CollectTokenIds(Symbol[] symbols, int symbolCount, List<int> dest)
    {
        for (int i = 0; i < symbolCount; i++)
            if (!symbols[i].Deleted) dest.Add(symbols[i].TokenId);
    }

    /// <summary>Appends buffered bytes as UTF-8 to <paramref name="sb"/> and resets the count.</summary>
    internal static void FlushByteBuffer(StringBuilder sb, byte[]? buffer, ref int count)
    {
        if (count == 0) return;
        sb.Append(Encoding.UTF8.GetString(buffer!, 0, count));
        count = 0;
    }

    /// <summary>
    /// Builds a 256-entry byte→token-ID mapping from <c>&lt;0xNN&gt;</c> vocab entries.
    /// Entries with no matching byte token are set to -1.
    /// </summary>
    internal static int[] BuildByteToTokenId(string[] tokens)
    {
        int[] byteToTokenId = new int[256];
        byteToTokenId.AsSpan().Fill(-1);
        for (int i = 0; i < tokens.Length; i++)
        {
            if (TryParseByteLiteral(tokens[i], out byte b))
                byteToTokenId[b] = i;
        }
        return byteToTokenId;
    }
}
