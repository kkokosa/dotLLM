using System.Runtime.InteropServices;

namespace DotLLM.Metal.Interop;

internal static partial class MetalNative
{
    /// <summary>Creates a KV-cache backed by persistent MTLResourceStorageModeShared buffers.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_kvcache_create")]
    internal static partial nint KvCacheCreate(
        nint ctx, int numLayers, int numKvHeads, int headDim, int maxSeqLen);

    /// <summary>Destroys the KV-cache and releases all MTLBuffer resources.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_kvcache_destroy")]
    [SuppressGCTransition]
    internal static partial void KvCacheDestroy(nint cache);

    /// <summary>Returns the CPU-writable contents pointer for layer's K buffer (FP16).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_kvcache_key_ptr")]
    [SuppressGCTransition]
    internal static unsafe partial void* KvCacheKeyPtr(nint cache, int layer);

    /// <summary>Returns the CPU-writable contents pointer for layer's V buffer (FP16).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_kvcache_value_ptr")]
    [SuppressGCTransition]
    internal static unsafe partial void* KvCacheValuePtr(nint cache, int layer);

    /// <summary>Sets the current valid length (for rollback / prefix-cache reuse).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_kvcache_set_current_length")]
    [SuppressGCTransition]
    internal static partial void KvCacheSetCurrentLength(nint cache, int length);

}
