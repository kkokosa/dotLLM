#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dotllm_metal_context dotllm_metal_context;

/// Opaque KV-cache backed by persistent MTLResourceStorageModeShared buffers.
/// On Apple Silicon, Shared storage is physically accessible by both the CPU
/// and the GPU — the C# side writes K/V via the contents pointer and the
/// attention kernel reads directly, with zero copies.
typedef struct dotllm_metal_kvcache dotllm_metal_kvcache;

// Internal record describing a memory region registered with the context.
// Used for range-based lookup of pointers that fall inside a larger zero-copy
// MTLBuffer (e.g. the GGUF mmap base + arbitrary per-tensor offsets).
@class DotLLMRegion;

struct dotllm_metal_context {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;       // pre-compiled dotllm_kernels.metallib, loaded once
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines;

    // Maps shared-memory pointers (MTLBuffer.contents) to their backing MTLBuffer.
    // Populated by dotllm_metal_alloc_shared — exact-pointer matches only.
    NSMapTable<NSValue*, id<MTLBuffer>>* shared_buffers;

    // Registered regions for range-based lookup. Populated by
    // dotllm_metal_register_buffer (used for mmap'd GGUF tensor data).
    // A tensor pointer T is owned by region R iff R.base <= T < R.base + R.length;
    // its offset inside the MTLBuffer is then (T - R.base).
    NSMutableArray<DotLLMRegion*>* regions;

    // Active command-buffer batching state.
    // When `active_enc` != nil, run_* helpers encode into the existing encoder
    // and SKIP the per-kernel commit/waitUntilCompleted. The owning forward
    // pass is responsible for closing it via dotllm_metal_end_forward.
    // When nil, helpers operate in standalone mode (one CB per kernel) — the
    // default path used by unit tests.
    id<MTLCommandBuffer>         active_cmd;
    id<MTLComputeCommandEncoder> active_enc;
};


#ifdef __cplusplus
}
#endif
