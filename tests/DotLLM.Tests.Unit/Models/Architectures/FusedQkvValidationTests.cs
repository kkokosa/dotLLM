using System.IO;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Regression tests for fused QKV shape validation. Without these guards, mismatched
/// GGUF metadata (e.g. wrong NumKvHeads / HeadDim) silently passes through
/// <see cref="TransformerWeights"/> and produces pointer offsets past the tensor's
/// allocated bytes — corrupting later layers and crashing inference. See upstream
/// issue #107 item 2.
/// </summary>
public class FusedQkvValidationTests
{
    /// <summary>
    /// Verifies the validator throws when the fused QKV weight's output dim does not
    /// match <c>qDim + 2 * kvDim</c>. Without validation, this mismatch would cause
    /// the K and V split pointers to read past the tensor's allocated rows.
    /// </summary>
    [Fact]
    public void ValidateFusedQkvShape_MismatchedOutputDim_Throws()
    {
        // Config says we expect Q=512 (8 heads x 64), KV=128 (2 heads x 64), total rows = 768.
        // But the tensor on disk claims only 700 rows — typical mismatch from wrong
        // NumKvHeads metadata in the GGUF.
        var shape = new TensorShape(512, 700); // [input_dim=512, output_dim=700]
        int qDim = 512;
        int kvDim = 128;

        var ex = Assert.Throws<InvalidDataException>(
            () => TransformerWeights.ValidateFusedQkvShape(shape, qDim, kvDim, "blk.0.attn_qkv.weight"));

        Assert.Contains("blk.0.attn_qkv.weight", ex.Message);
        Assert.Contains("700", ex.Message);
        Assert.Contains("768", ex.Message); // expected qDim + 2*kvDim
    }

    /// <summary>
    /// Verifies the validator accepts a correctly-sized fused QKV weight tensor.
    /// </summary>
    [Fact]
    public void ValidateFusedQkvShape_MatchingOutputDim_DoesNotThrow()
    {
        int qDim = 512;
        int kvDim = 128;
        var shape = new TensorShape(512, qDim + 2 * kvDim); // [512, 768]

        // Should not throw.
        TransformerWeights.ValidateFusedQkvShape(shape, qDim, kvDim, "blk.0.attn_qkv.weight");
    }

    /// <summary>
    /// Verifies the validator throws on a rank-1 tensor (defensive: fused QKV must be 2-D).
    /// </summary>
    [Fact]
    public void ValidateFusedQkvShape_Rank1Tensor_Throws()
    {
        var shape = new TensorShape(768);
        var ex = Assert.Throws<InvalidDataException>(
            () => TransformerWeights.ValidateFusedQkvShape(shape, 512, 128, "blk.0.attn_qkv.weight"));
        Assert.Contains("rank", ex.Message);
    }

    /// <summary>
    /// Verifies the validator throws on a rank-3 tensor. The split path interprets only
    /// <c>Shape[0]</c>/<c>Shape[1]</c> and ignores trailing dimensions, so accepting a
    /// higher-rank shape would validate against the wrong axis and then split with
    /// pointer arithmetic that runs past the tensor's allocated bytes.
    /// </summary>
    [Fact]
    public void ValidateFusedQkvShape_Rank3Tensor_Throws()
    {
        // Output rows on axis 1 match qDim + 2*kvDim, so only the rank check can reject this.
        var shape = new TensorShape(512, 768, 4);
        var ex = Assert.Throws<InvalidDataException>(
            () => TransformerWeights.ValidateFusedQkvShape(shape, 512, 128, "blk.0.attn_qkv.weight"));
        Assert.Contains("rank", ex.Message);
    }

    /// <summary>
    /// Verifies the bias validator throws when the fused bias's element count does not
    /// match <c>qDim + 2 * kvDim</c>. Without validation, the K/V bias dequant pointers
    /// would read past the allocated bias bytes.
    /// </summary>
    [Fact]
    public void ValidateFusedQkvBiasShape_MismatchedElements_Throws()
    {
        int qDim = 512;
        int kvDim = 128;
        // Tensor on disk claims 700 elements — mismatched.
        var shape = new TensorShape(700);

        var ex = Assert.Throws<InvalidDataException>(
            () => TransformerWeights.ValidateFusedQkvBiasShape(shape, qDim, kvDim, "blk.0.attn_qkv.bias"));

        Assert.Contains("blk.0.attn_qkv.bias", ex.Message);
        Assert.Contains("700", ex.Message);
        Assert.Contains("768", ex.Message);
    }

    /// <summary>
    /// Verifies the bias validator accepts a correctly-sized fused bias tensor.
    /// </summary>
    [Fact]
    public void ValidateFusedQkvBiasShape_MatchingElements_DoesNotThrow()
    {
        int qDim = 512;
        int kvDim = 128;
        var shape = new TensorShape(qDim + 2 * kvDim);

        // Should not throw.
        TransformerWeights.ValidateFusedQkvBiasShape(shape, qDim, kvDim, "blk.0.attn_qkv.bias");
    }
}
