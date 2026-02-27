namespace DotLLM.Engine;

/// <summary>
/// Manages LoRA adapter lifecycle: loading, unloading, and runtime switching.
/// Supports concurrent adapters for multi-tenant serving.
/// </summary>
public interface IAdapterManager
{
    /// <summary>
    /// Loads a LoRA adapter from disk.
    /// </summary>
    /// <param name="name">Unique name for this adapter.</param>
    /// <param name="path">Path to the adapter weights.</param>
    void LoadAdapter(string name, string path);

    /// <summary>
    /// Unloads a previously loaded adapter, freeing its resources.
    /// </summary>
    /// <param name="name">Name of the adapter to unload.</param>
    void UnloadAdapter(string name);

    /// <summary>
    /// Gets a loaded adapter by name.
    /// </summary>
    /// <param name="name">Adapter name.</param>
    /// <returns>The adapter, or null if not loaded.</returns>
    LoraAdapter? GetAdapter(string name);

    /// <summary>
    /// Lists all currently loaded adapter names.
    /// </summary>
    IReadOnlyList<string> ListAdapters();
}
