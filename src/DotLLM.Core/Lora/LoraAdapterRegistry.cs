using System.Collections.Concurrent;

namespace DotLLM.Core.Lora;

/// <summary>
/// Default <see cref="ILoraAdapterRegistry"/> implementation. Keeps loaded
/// adapters in a <see cref="ConcurrentDictionary{TKey, TValue}"/> so reads
/// (per-request adapter lookup) are lock-free; loads/unloads serialise
/// through a process-wide lock since they involve disk I/O and disposal.
/// </summary>
/// <remarks>
/// <para>
/// The registry takes a <see cref="Func{T, TResult}"/> factory rather than
/// hard-binding to the PEFT loader so DotLLM.Core stays free of any
/// SafeTensors dependency — DotLLM.Models supplies the production factory
/// (see <c>PeftAdapterLoader.LoadFromDirectory</c>) and tests may inject
/// synthetic loaders.
/// </para>
/// </remarks>
public sealed class LoraAdapterRegistry : ILoraAdapterRegistry
{
    private readonly ConcurrentDictionary<string, ILoraAdapter> _adapters = new(StringComparer.Ordinal);
    private readonly object _writeLock = new();
    private readonly Func<string, string, ILoraAdapter> _loaderFactory;
    private bool _disposed;

    /// <summary>
    /// Creates a registry that uses <paramref name="loaderFactory"/> to
    /// materialise adapters from disk. The factory receives
    /// <c>(name, path)</c> and must return an owned <see cref="ILoraAdapter"/>
    /// — the registry takes responsibility for disposing it.
    /// </summary>
    public LoraAdapterRegistry(Func<string, string, ILoraAdapter> loaderFactory)
    {
        ArgumentNullException.ThrowIfNull(loaderFactory);
        _loaderFactory = loaderFactory;
    }

    /// <inheritdoc/>
    public void Load(string name, string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentException.ThrowIfNullOrEmpty(path);

        lock (_writeLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_adapters.ContainsKey(name))
                throw new InvalidOperationException($"LoRA adapter '{name}' is already loaded.");

            var adapter = _loaderFactory(name, path);
            if (adapter is null)
                throw new InvalidOperationException(
                    $"LoRA adapter loader returned null for '{name}' at '{path}'.");

            // Verify name consistency — defensive: the factory could mint a
            // different name. Prefer the registry's view since that is the
            // key callers will use.
            if (!StringComparer.Ordinal.Equals(adapter.Name, name))
            {
                adapter.Dispose();
                throw new InvalidOperationException(
                    $"LoRA adapter loader returned name '{adapter.Name}' but '{name}' was requested.");
            }

            if (!_adapters.TryAdd(name, adapter))
            {
                adapter.Dispose();
                throw new InvalidOperationException($"LoRA adapter '{name}' is already loaded (race).");
            }
        }
    }

    /// <inheritdoc/>
    public void Unload(string name)
    {
        if (string.IsNullOrEmpty(name)) return;
        lock (_writeLock)
        {
            if (_disposed) return;
            if (_adapters.TryRemove(name, out var adapter))
                adapter.Dispose();
        }
    }

    /// <inheritdoc/>
    public ILoraAdapter? Get(string name)
    {
        if (string.IsNullOrEmpty(name)) return null;
        return _adapters.TryGetValue(name, out var adapter) ? adapter : null;
    }

    /// <inheritdoc/>
    public IReadOnlyList<string> List()
    {
        // ConcurrentDictionary.Keys allocates a snapshot — perfect for the
        // stable-snapshot contract.
        var keys = new List<string>(_adapters.Keys);
        return keys;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_writeLock)
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var adapter in _adapters.Values)
                adapter.Dispose();
            _adapters.Clear();
        }
    }
}
