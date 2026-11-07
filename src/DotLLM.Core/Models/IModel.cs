using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;

namespace DotLLM.Core.Models;

/// <summary>
/// A loaded, ready-to-run transformer model.
/// </summary>
public interface IModel : IDisposable
{
    /// <summary>Model configuration.</summary>
    ModelConfig Config { get; }

    /// <summary>Total bytes allocated for inference compute scratch buffers.</summary>
    long ComputeMemoryBytes { get; }

    /// <summary>
    /// Total bytes of committed memory holding repacked (layout-transformed) copies of the model
    /// weights, or 0 when the backend does not repack. This is memory in addition to the
    /// memory-mapped weight file, not a subset of it.
    /// </summary>
    long RepackedWeightBytes => 0;

    /// <summary>
    /// Runs a forward pass through the model.
    /// </summary>
    /// <param name="tokenIds">Input token IDs for this batch.</param>
    /// <param name="positions">Position indices for each token.</param>
    /// <param name="deviceId">Target device for computation.</param>
    /// <returns>Logits tensor of shape [batch, vocab_size].</returns>
    ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId);

    /// <summary>
    /// Runs a forward pass with optional KV-cache for efficient autoregressive decoding.
    /// </summary>
    /// <param name="tokenIds">Input token IDs for this step.</param>
    /// <param name="positions">Position indices for each token.</param>
    /// <param name="deviceId">Target device for computation.</param>
    /// <param name="kvCache">Optional KV-cache. When null, behaves identically to the uncached forward pass.</param>
    /// <returns>Logits tensor of shape [1, vocab_size] or [seqLen, vocab_size].</returns>
    ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache);

    /// <summary>
    /// Runs a forward pass with optional KV-cache, optionally computing output logits only for the final token.
    /// During prompt prefill, only the last token's logits are needed to sample the first generated token,
    /// whereas speculative decoding verification requires logits for all input positions.
    /// </summary>
    /// <param name="tokenIds">Input token IDs for this step.</param>
    /// <param name="positions">Position indices for each token.</param>
    /// <param name="deviceId">Target device for computation.</param>
    /// <param name="kvCache">Optional KV-cache. When null, behaves identically to the uncached forward pass.</param>
    /// <param name="lastLogitOnly">When true, skips final normalization and LM head projection for positions 0..seqLen-2 and returns [1, vocab_size].</param>
    /// <returns>Logits tensor of shape [1, vocab_size] when <paramref name="lastLogitOnly"/> is true, otherwise [seqLen, vocab_size].</returns>
    ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, bool lastLogitOnly)
        => Forward(tokenIds, positions, deviceId, kvCache);
}
