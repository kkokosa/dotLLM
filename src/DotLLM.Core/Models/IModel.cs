using DotLLM.Core.Attention;
using DotLLM.Core.Lora;
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

    /// <summary>
    /// Runs a forward pass with optional KV-cache and an optional LoRA adapter.
    /// When <paramref name="adapter"/> is non-null and supplies <c>(layer, proj)</c>
    /// factor pairs that match the current model's projection sites, the runtime
    /// adds the LoRA delta <c>alpha × (x · B) · A</c> to each adapted projection.
    /// </summary>
    /// <param name="tokenIds">Input token IDs for this step.</param>
    /// <param name="positions">Position indices for each token.</param>
    /// <param name="deviceId">Target device for computation.</param>
    /// <param name="kvCache">Optional KV-cache. When null, behaves identically to the uncached forward pass.</param>
    /// <param name="adapter">
    /// Optional LoRA adapter. When null, behaves byte-equivalently to the
    /// adapter-less <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?)"/>
    /// overload (default implementation forwards to it).
    /// </param>
    /// <returns>Logits tensor of shape [seq, vocab_size] for all input positions.</returns>
    ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                    IKvCache? kvCache, ILoraAdapter? adapter)
        => Forward(tokenIds, positions, deviceId, kvCache);

    /// <summary>
    /// Runs a fused forward pass across multiple in-flight sequences.
    /// </summary>
    /// <remarks>
    /// <para>The continuous-batch scheduler calls this once per iteration when 2+ sequences are
    /// active, instead of looping <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?, ILoraAdapter?)"/>
    /// per sequence. Each <paramref name="requests"/> entry carries its own tokens, positions,
    /// and KV-cache — sequences are independent at the attention level (no cross-sequence
    /// attention).</para>
    /// <para>The default implementation simply loops over <c>Forward</c> per request and returns
    /// the results in input order. Implementations can override to fuse the per-sequence GEMVs
    /// into batched GEMMs and avoid the per-iteration kernel-dispatch overhead — this is the
    /// principal continuous-batching throughput win.</para>
    /// <para>The returned tensors follow the same shape contract as <c>Forward</c>: each entry
    /// is <c>[N_i, vocab_size]</c> where <c>N_i</c> matches that request's token count (CPU
    /// model) or <c>[1, vocab_size]</c> for the last token only (GPU/hybrid). The caller is
    /// responsible for disposing each returned tensor.</para>
    /// </remarks>
    /// <param name="requests">One entry per active sequence. Order is preserved in the result.</param>
    /// <param name="deviceId">Target device for computation.</param>
    /// <returns>Logits tensors, one per request, in the same order.</returns>
    IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();

        var results = new ITensor[requests.Count];
        for (int i = 0; i < requests.Count; i++)
        {
            var r = requests[i];
            results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.Adapter);
        }
        return results;
    }
}
