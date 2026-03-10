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
    /// <returns>Logits tensor of shape [1, vocab_size] for the last token.</returns>
    ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache);
}
