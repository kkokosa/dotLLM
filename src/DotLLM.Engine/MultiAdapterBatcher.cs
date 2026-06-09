using DotLLM.Core.Lora;

namespace DotLLM.Engine;

/// <summary>
/// Phase 4c first-cut helper for grouping a batch of inference requests by
/// LoRA adapter identity and dispatching them sequentially per group.
/// </summary>
/// <remarks>
/// <para>
/// True parallelised group dispatch (where the base matmul runs once across
/// all sequences and the per-group LoRA delta is fused on top) is a
/// performance optimisation tracked for Phase 4d / Wave 9. The Phase 4c
/// guarantee is correctness — every request sees its own adapter applied —
/// not throughput.
/// </para>
/// <para>
/// The current server request gate (<c>DotLLM.Server.ServerState</c>)
/// serialises inference requests anyway, so this batcher is used by tests
/// and the future multi-request scheduler to express the partition
/// contract independently of the gate. <see cref="Group{T}"/> is pure
/// and allocation-light.
/// </para>
/// </remarks>
public static class MultiAdapterBatcher
{
    /// <summary>
    /// Partitions a batch of <typeparamref name="T"/> requests by adapter
    /// identity. Requests with no adapter (<c>null</c>) form one group;
    /// each distinct non-null adapter forms its own group keyed by
    /// reference equality (so two distinct registry entries with the same
    /// name are still treated as different groups — the registry guarantees
    /// a single instance per name, so this never bites in practice).
    /// </summary>
    /// <typeparam name="T">Per-request payload type.</typeparam>
    /// <param name="requests">The batch to partition.</param>
    /// <param name="adapterSelector">
    /// Selector function returning the <see cref="ILoraAdapter"/> the
    /// request will run under (or <c>null</c> for the base model).
    /// </param>
    /// <returns>
    /// A list of (adapter, requests) groups, in stable insertion order:
    /// the first group seen for each adapter is yielded first; within
    /// each group the relative order of <paramref name="requests"/> is
    /// preserved. The base-model (<c>null</c>) group, when non-empty,
    /// always yields first so single-batch base inference takes the
    /// fast path.
    /// </returns>
    public static IReadOnlyList<AdapterGroup<T>> Group<T>(
        IReadOnlyList<T> requests,
        Func<T, ILoraAdapter?> adapterSelector)
    {
        ArgumentNullException.ThrowIfNull(requests);
        ArgumentNullException.ThrowIfNull(adapterSelector);

        if (requests.Count == 0)
            return Array.Empty<AdapterGroup<T>>();

        // Reference-keyed dictionary of distinct adapters; the null group is
        // tracked separately so we can yield it first deterministically.
        var byAdapter = new Dictionary<ILoraAdapter, List<T>>(ReferenceEqualityComparer.Instance);
        List<T>? nullGroup = null;
        var adapterOrder = new List<ILoraAdapter>();

        for (int i = 0; i < requests.Count; i++)
        {
            var req = requests[i];
            var adapter = adapterSelector(req);
            if (adapter is null)
            {
                nullGroup ??= new List<T>();
                nullGroup.Add(req);
            }
            else
            {
                if (!byAdapter.TryGetValue(adapter, out var bucket))
                {
                    bucket = new List<T>();
                    byAdapter[adapter] = bucket;
                    adapterOrder.Add(adapter);
                }
                bucket.Add(req);
            }
        }

        var result = new List<AdapterGroup<T>>(byAdapter.Count + (nullGroup is null ? 0 : 1));
        if (nullGroup is not null)
            result.Add(new AdapterGroup<T>(null, nullGroup));
        foreach (var adapter in adapterOrder)
            result.Add(new AdapterGroup<T>(adapter, byAdapter[adapter]));

        return result;
    }
}

/// <summary>
/// One adapter-keyed partition of a multi-adapter batch.
/// </summary>
/// <param name="Adapter">
/// LoRA adapter applied to every request in this group, or <c>null</c>
/// for the base-model group.
/// </param>
/// <param name="Requests">
/// Requests in this group, in the original batch order.
/// </param>
public sealed record AdapterGroup<T>(ILoraAdapter? Adapter, IReadOnlyList<T> Requests);
