using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Tokenizers;

namespace DotLLM.Server;

/// <summary>
/// Shared server state: loaded model metadata, concurrency control, readiness.
/// </summary>
public sealed class ServerState
{
    private readonly SemaphoreSlim _requestGate = new(1, 1);

    /// <summary>Server startup options.</summary>
    public required ServerOptions Options { get; init; }

    /// <summary>Model configuration.</summary>
    public required ModelConfig Config { get; init; }

    /// <summary>Tool call parser for the loaded model.</summary>
    public IToolCallParser? ToolCallParser { get; init; }

    /// <summary>KV-cache configuration.</summary>
    public required KvCacheConfig KvCacheConfig { get; init; }

    /// <summary>KV-cache factory for the loaded model/device.</summary>
    public Func<ModelConfig, int, DotLLM.Core.Attention.IKvCache>? KvCacheFactory { get; init; }

    /// <summary>Whether the server is ready to accept requests.</summary>
    public bool IsReady { get; set; }

    /// <summary>
    /// Executes a request with sequential access control.
    /// Only one request is processed at a time (Step 35 adds batching).
    /// </summary>
    public async Task ExecuteAsync(Func<Task> work, CancellationToken ct)
    {
        await _requestGate.WaitAsync(ct);
        try { await work(); }
        finally { _requestGate.Release(); }
    }
}
