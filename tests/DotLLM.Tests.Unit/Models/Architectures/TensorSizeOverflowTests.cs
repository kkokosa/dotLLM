using System.Globalization;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Regression tests for the long-to-int narrowing used when allocating managed
/// arrays for bias tensors loaded from GGUF. Without the `checked` cast, an
/// <c>ElementCount</c> of 2^32 silently wrapped to <c>0</c>, producing a
/// zero-length array and corrupt downstream data with no error. See upstream
/// issue #107 item 3.
/// </summary>
public class TensorSizeOverflowTests
{
    /// <summary>
    /// 2^32 (4,294,967,296) wraps to <c>0</c> when narrowed without a `checked`
    /// cast — a positive-but-wrong value that bypasses every later sanity check.
    /// The fixed code must throw <see cref="OverflowException"/> at the narrowing
    /// site, not allocate a zero-element array.
    /// </summary>
    [Fact]
    public void ToInt32SizeChecked_2Pow32_ThrowsInsteadOfSilentlyWrappingToZero()
    {
        long elementCount = 1L << 32; // 4,294,967,296 — wraps to 0 unchecked.
        // Explicitly unchecked: this line documents the ORIGINAL buggy behaviour, so it must
        // keep wrapping even if the project ever turns on CheckForOverflowUnderflow (which
        // would otherwise make it throw and fail the test for the wrong reason).
        Assert.Equal(0, unchecked((int)elementCount)); // confirms the silent-wrap scenario.

        var ex = Assert.Throws<OverflowException>(
            () => TransformerWeights.ToInt32SizeChecked(elementCount, "blk.0.attn_q.bias"));

        Assert.Contains("blk.0.attn_q.bias", ex.Message);
        Assert.Contains("4294967296", ex.Message);
    }

    /// <summary>
    /// <c>int.MaxValue + 1</c> wraps to <c>int.MinValue</c> unchecked, then a
    /// later <c>new float[negative]</c> would throw <see cref="OverflowException"/>
    /// anyway — but with a confusing stack frame far from the actual cause. The
    /// fix throws at the cast site with a useful error message.
    /// </summary>
    [Fact]
    public void ToInt32SizeChecked_BeyondInt32MaxValue_Throws()
    {
        long elementCount = (long)int.MaxValue + 1;
        var ex = Assert.Throws<OverflowException>(
            () => TransformerWeights.ToInt32SizeChecked(elementCount, "blk.0.attn_v.bias"));

        Assert.Contains("blk.0.attn_v.bias", ex.Message);
        Assert.Contains("2147483648", ex.Message);
    }

    /// <summary>
    /// A negative count is in <see cref="int"/> range, so a bare <c>checked</c> cast lets it
    /// through and the failure surfaces much later as <c>new float[negative]</c> — an
    /// <see cref="OverflowException"/> naming neither the tensor nor the count. It is
    /// reachable: <c>TensorShape.ElementCount</c> multiplies dimensions in unchecked
    /// <see cref="long"/> arithmetic, so a crafted GGUF with individually-valid dimensions
    /// whose product passes <see cref="long.MaxValue"/> can wrap negative.
    /// </summary>
    [Theory]
    [InlineData(-1L)]
    [InlineData(-4096L)]
    [InlineData(long.MinValue)]
    public void ToInt32SizeChecked_NegativeCount_Throws(long elementCount)
    {
        var ex = Assert.Throws<OverflowException>(
            () => TransformerWeights.ToInt32SizeChecked(elementCount, "blk.0.ffn_up.bias"));

        Assert.Contains("blk.0.ffn_up.bias", ex.Message);
        Assert.Contains(elementCount.ToString(CultureInfo.InvariantCulture), ex.Message);
    }

    /// <summary>
    /// Sizes that fit in <see cref="int"/> pass through unchanged.
    /// </summary>
    [Theory]
    [InlineData(0L)]
    [InlineData(1L)]
    [InlineData(4096L)]
    [InlineData(2_147_483_647L)] // int.MaxValue exactly
    public void ToInt32SizeChecked_WithinInt32Range_ReturnsValue(long elementCount)
    {
        int actual = TransformerWeights.ToInt32SizeChecked(elementCount, "blk.0.attn_q.bias");
        Assert.Equal((int)elementCount, actual);
    }
}
