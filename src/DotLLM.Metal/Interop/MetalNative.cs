using System.Runtime.InteropServices;

namespace DotLLM.Metal.Interop;

/// <summary>
/// P/Invoke declarations for the native Metal library (libdotllmmetal.dylib).
/// All kernel functions take an opaque context handle as first argument.
/// </summary>
internal static partial class MetalNative
{
    private const string LibName = "dotllmmetal";

    /// <summary>Creates a Metal context (device + command queue + pipeline cache).</summary>
    /// <returns>Opaque pointer, or <c>0</c> on failure.</returns>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_create_context")]
    internal static partial nint CreateContext();

    /// <summary>Destroys the context and releases all Metal resources.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_destroy_context")]
    [SuppressGCTransition]
    internal static partial void DestroyContext(nint ctx);

    /// <summary>
    /// Allocates shared (CPU+GPU visible) memory and returns the .contents pointer.
    /// The backing MTLBuffer is retained by the context for zero-copy kernel access.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_alloc_shared")]
    internal static partial nint AllocShared(nint ctx, nuint bytes);

    /// <summary>
    /// Releases a buffer previously returned by <see cref="AllocShared"/>.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_free_shared")]
    [SuppressGCTransition]
    internal static partial void FreeShared(nint ctx, nint ptr);

    /// <summary>
    /// Wraps a caller-owned page-aligned memory region (e.g. an mmap'd GGUF
    /// file) as a zero-copy MTLBuffer. Pointers within <c>[ptr, ptr+bytes)</c>
    /// become eligible for zero-copy kernel encoding (the kernel-side helper
    /// recovers the MTLBuffer and the offset within it).
    /// Caller MUST keep the memory alive (and not unmap it) until the matching
    /// <see cref="UnregisterBuffer"/> call or context destruction.
    /// Returns 0 on success, negative on failure (alignment or OOM).
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_register_buffer")]
    internal static partial int RegisterBuffer(nint ctx, nint ptr, nuint bytes);

    /// <summary>
    /// Unregisters a region previously passed to <see cref="RegisterBuffer"/>.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_unregister_buffer")]
    [SuppressGCTransition]
    internal static partial void UnregisterBuffer(nint ctx, nint ptr);

    /// <summary>
    /// Opens a batched forward pass. All subsequent kernel calls on this
    /// context encode into one shared command buffer until <see cref="EndForward"/>.
    /// Returns 0 on success, negative on error.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_begin_forward")]
    internal static partial int BeginForward(nint ctx);

    /// <summary>
    /// Closes the active batched forward pass: commits and waits for GPU.
    /// Returns 0 on success, negative on error.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_end_forward")]
    internal static partial int EndForward(nint ctx);

    /// <summary>
    /// Copies <paramref name="bytes"/> from <paramref name="src"/> to
    /// <paramref name="dst"/>. Inside a batched forward this is a GPU-side
    /// blit (zero CPU work, sequenced with surrounding compute kernels);
    /// outside, it falls back to a CPU memcpy.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_buffer_copy")]
    internal static partial int BufferCopy(nint ctx, nint dst, nint src, nuint bytes);
}
