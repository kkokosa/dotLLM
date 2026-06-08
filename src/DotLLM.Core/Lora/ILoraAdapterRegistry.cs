namespace DotLLM.Core.Lora;

/// <summary>
/// Registry of loaded LoRA adapters. Supports hot-load / hot-unload at
/// runtime so adapter rotation does not require a server restart.
/// </summary>
/// <remarks>
/// <para>
/// Per the dotLLM design (see <c>docs/LORA.md</c>), <see cref="Load"/> is
/// synchronous and expected to complete in &lt;1 s for a typical 7B-scale
/// adapter (10–100 MB on disk). Adapter switching is instant — the registry
/// hands out the same <see cref="ILoraAdapter"/> reference until
/// <see cref="Unload"/> is called.
/// </para>
/// <para>
/// Implementations must be thread-safe for concurrent <see cref="Get"/> and
/// <see cref="List"/> calls; <see cref="Load"/> / <see cref="Unload"/>
/// may serialize internally.
/// </para>
/// </remarks>
public interface ILoraAdapterRegistry : IDisposable
{
    /// <summary>
    /// Loads an adapter from <paramref name="path"/> (a HuggingFace PEFT
    /// directory containing <c>adapter_config.json</c> and
    /// <c>adapter_model.safetensors</c>) and registers it under
    /// <paramref name="name"/>. Throws <see cref="InvalidOperationException"/>
    /// when an adapter with that name is already loaded.
    /// </summary>
    void Load(string name, string path);

    /// <summary>
    /// Unloads the adapter registered under <paramref name="name"/>,
    /// disposing its native buffers. No-op when the name is unknown.
    /// </summary>
    void Unload(string name);

    /// <summary>
    /// Returns the loaded adapter for <paramref name="name"/>, or
    /// <c>null</c> when no such adapter is registered.
    /// </summary>
    ILoraAdapter? Get(string name);

    /// <summary>
    /// Snapshots the names of all currently-loaded adapters. The returned
    /// list is a stable copy — concurrent loads/unloads after the call do
    /// not mutate it.
    /// </summary>
    IReadOnlyList<string> List();
}
