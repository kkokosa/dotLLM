using System.Runtime.CompilerServices;

namespace DotLLM.Core.Constraints;

/// <summary>
/// Compact bit vector over the full vocabulary for constrained decoding.
/// Uses managed <c>long[]</c> — this is a transient constraint mask, not tensor data.
/// A 128K vocabulary requires only 16KB.
/// </summary>
public struct TokenMask
{
    private readonly long[] _bits;

    /// <summary>
    /// Creates a token mask for a vocabulary of the given size.
    /// All tokens are initially disallowed.
    /// </summary>
    /// <param name="vocabSize">Total vocabulary size.</param>
    public TokenMask(int vocabSize)
    {
        _bits = new long[(vocabSize + 63) / 64];
    }

    /// <summary>Marks a token as allowed.</summary>
    /// <param name="tokenId">Token ID to allow.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Allow(int tokenId)
    {
        _bits[tokenId >> 6] |= 1L << (tokenId & 63);
    }

    /// <summary>Marks a token as disallowed.</summary>
    /// <param name="tokenId">Token ID to disallow.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Disallow(int tokenId)
    {
        _bits[tokenId >> 6] &= ~(1L << (tokenId & 63));
    }

    /// <summary>Checks whether a token is allowed.</summary>
    /// <param name="tokenId">Token ID to check.</param>
    /// <returns>True if the token is allowed.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly bool IsAllowed(int tokenId)
    {
        return (_bits[tokenId >> 6] & (1L << (tokenId & 63))) != 0;
    }

    /// <summary>Marks all tokens as allowed.</summary>
    public void AllowAll()
    {
        Array.Fill(_bits, ~0L);
    }

    /// <summary>Marks all tokens as disallowed.</summary>
    public void DisallowAll()
    {
        Array.Clear(_bits);
    }

    /// <summary>Gets the underlying bit storage as a span for vectorized masking.</summary>
    public readonly ReadOnlySpan<long> AsSpan() => _bits;
}
