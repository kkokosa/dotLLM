using DotLLM.Core.Attention;
using DotLLM.Core.Lora;

namespace DotLLM.Core.Models;

/// <summary>
/// One sequence's contribution to a batched <see cref="IModel.ForwardBatch"/> call.
/// </summary>
/// <remarks>
/// <para>The scheduler bundles N of these into a single dispatch when more than one sequence
/// is active in the same iteration. Each request carries its own token chunk (single decode
/// step or a prefill slice), its own position offsets into its own KV-cache, and its own
/// optional LoRA adapter.</para>
/// <para>Implementations are free to fuse the batch into a single kernel dispatch (compute-wise
/// optimal) or to fall back to a per-sequence loop. The contract is purely about the API
/// shape — see <see cref="IModel.ForwardBatch"/> for the per-request output layout.</para>
/// </remarks>
public readonly record struct SequenceForwardRequest
{
    /// <summary>Token IDs for this sequence in this batch (decode = 1 token, prefill = N).</summary>
    public required ReadOnlyMemory<int> TokenIds { get; init; }

    /// <summary>Position indices for each token. Same length as <see cref="TokenIds"/>.</summary>
    public required ReadOnlyMemory<int> Positions { get; init; }

    /// <summary>Per-sequence KV-cache handle. Independent across sequences.</summary>
    public required IKvCache KvCache { get; init; }

    /// <summary>Optional per-sequence LoRA adapter. <see langword="null"/> for the base model.</summary>
    public ILoraAdapter? Adapter { get; init; }
}
