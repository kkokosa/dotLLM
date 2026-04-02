using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;

namespace DotLLM.Server;

/// <summary>
/// Shared server state: loaded model, concurrency control, mutable configuration.
/// Endpoints access live model/generator instances through this object.
/// </summary>
public sealed class ServerState : IDisposable
{
    private readonly SemaphoreSlim _requestGate = new(1, 1);

    /// <summary>Server startup options.</summary>
    public required ServerOptions Options { get; init; }

    /// <summary>Model configuration (updated on model swap).</summary>
    public required ModelConfig Config { get; set; }

    /// <summary>Tool call parser for the loaded model.</summary>
    public IToolCallParser? ToolCallParser { get; set; }

    /// <summary>KV-cache configuration.</summary>
    public KvCacheConfig KvCacheConfig { get; set; }

    /// <summary>KV-cache factory for the loaded model/device.</summary>
    public Func<ModelConfig, int, IKvCache>? KvCacheFactory { get; set; }

    /// <summary>Whether the server is ready to accept requests.</summary>
    public bool IsReady { get; set; }

    // ── Live instances (mutable for model swap) ──

    /// <summary>Currently loaded model.</summary>
    public required IModel Model { get; set; }

    /// <summary>Tokenizer for the loaded model.</summary>
    public required ITokenizer Tokenizer { get; set; }

    /// <summary>Chat template for the loaded model.</summary>
    public required IChatTemplate ChatTemplate { get; set; }

    /// <summary>Text generator wired to the current model.</summary>
    public required TextGenerator Generator { get; set; }

    /// <summary>Mutable sampling parameter defaults (changeable from the UI).</summary>
    public SamplingDefaults SamplingDefaults { get; set; } = new();

    /// <summary>Path of the currently loaded GGUF file.</summary>
    public string LoadedModelPath { get; set; } = "";

    /// <summary>Open GGUF file handle (disposed on model swap).</summary>
    public GgufFile? CurrentGguf { get; set; }

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

    /// <summary>
    /// Swaps the loaded model under the request gate.
    /// Blocks new requests during the swap, disposes the old model, and runs the load action.
    /// </summary>
    public async Task SwapModelAsync(Func<Task> loadAction, CancellationToken ct)
    {
        await _requestGate.WaitAsync(ct);
        IsReady = false;
        try
        {
            Model.Dispose();
            CurrentGguf?.Dispose();
            CurrentGguf = null;

            await loadAction();
            IsReady = true;
        }
        finally { _requestGate.Release(); }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        Model.Dispose();
        CurrentGguf?.Dispose();
        _requestGate.Dispose();
    }
}

/// <summary>
/// Mutable sampling parameter defaults that can be changed from the UI.
/// These serve as defaults when the per-request body does not specify a value.
/// </summary>
public sealed class SamplingDefaults
{
    /// <summary>Sampling temperature. 0 = greedy.</summary>
    public float Temperature { get; set; } = 0.7f;

    /// <summary>Top-P (nucleus) sampling threshold.</summary>
    public float TopP { get; set; } = 1.0f;

    /// <summary>Top-K sampling. 0 = disabled.</summary>
    public int TopK { get; set; }

    /// <summary>Min-P sampling threshold. 0 = disabled.</summary>
    public float MinP { get; set; }

    /// <summary>Repetition penalty factor. 1.0 = disabled.</summary>
    public float RepetitionPenalty { get; set; } = 1.0f;

    /// <summary>Maximum tokens to generate per response.</summary>
    public int MaxTokens { get; set; } = 2048;

    /// <summary>Random seed for reproducibility. Null = non-deterministic.</summary>
    public int? Seed { get; set; }
}
