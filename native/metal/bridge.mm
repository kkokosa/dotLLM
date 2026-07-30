#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <dlfcn.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>     // getpagesize
#include "dotllm_core.h"
#include "dotllm_metal.h"

// Internal record for a memory region registered via dotllm_metal_register_buffer.
// Holds the MTLBuffer with a strong reference so ARC retains it for the
// region's lifetime.
@interface DotLLMRegion : NSObject
@property (assign, nonatomic) const void*     base;
@property (assign, nonatomic) size_t          length;
@property (strong, nonatomic) id<MTLBuffer>   buffer;
@end
@implementation DotLLMRegion
@end

// Now that DotLLMRegion is defined, pull in the KV cache implementation
// (it pushes regions for each layer's K/V buffer so range-lookup resolves
// writes at arbitrary token offsets).
#include "metal_kv_cache.mm"

// Name of the pre-compiled archive produced by build.sh. Must sit next to
// libdotllmmetal.dylib in the deployment layout. Defined in one place so
// build.sh, the .csproj copy rules and the runtime loader stay in sync.
static NSString* const kKernelLibraryFileName = @"dotllm_kernels.metallib";

// Locate `dotllm_kernels.metallib` next to this dylib using dladdr().
// dl_info.dli_fname is the absolute path to the loaded image that contains
// the function whose address we hand in — i.e. ourselves. This is robust
// to whatever cwd the host process runs in.
static NSString* find_kernel_library(void)
{
    Dl_info info;
    if (dladdr((const void*)&find_kernel_library, &info) == 0 || !info.dli_fname) {
        return nil;
    }
    NSString* dylibPath = @(info.dli_fname);
    NSString* dir       = [dylibPath stringByDeletingLastPathComponent];
    return [dir stringByAppendingPathComponent:kKernelLibraryFileName];
}

dotllm_metal_context* dotllm_metal_create_context(void)
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return nullptr;

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return nullptr;

    // Load the pre-compiled archive once. No runtime MSL→AIR compilation.
    NSString* libPath = find_kernel_library();
    if (!libPath) return nullptr;

    NSError*      error  = nil;
    NSURL*        libURL = [NSURL fileURLWithPath:libPath];
    id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&error];
    if (!library) {
        NSLog(@"dotllm_metal: failed to load %@: %@", libPath, error);
        return nullptr;
    }

    dotllm_metal_context* ctx = new dotllm_metal_context();
    ctx->device    = device;
    ctx->queue     = queue;
    ctx->library   = library;
    ctx->pipelines = [NSMutableDictionary new];
    ctx->shared_buffers = [NSMapTable strongToStrongObjectsMapTable];
    ctx->regions        = [NSMutableArray new];
    return ctx;
}

void dotllm_metal_destroy_context(dotllm_metal_context* ctx)
{
    if (!ctx) return;
    // ARC releases device, queue, pipelines, shared_buffers (and the MTLBuffers
    // they retain) automatically.
    delete ctx;
}

// ── Shared-memory allocation (zero-copy backing for forward state) ────────────
//
// On Apple Silicon, MTLResourceStorageModeShared buffers expose their bytes via
// `.contents` — the same physical memory the GPU reads. A C# pointer obtained
// from dotllm_metal_alloc_shared can be:
//   - Read/written by the CPU directly (it's just RAM).
//   - Looked up to recover the backing MTLBuffer for zero-copy kernel encoding.
// The map keeps a strong reference to the MTLBuffer; calling free_shared (or
// destroying the context) releases it.

extern "C" void* dotllm_metal_alloc_shared(dotllm_metal_context* ctx, size_t bytes)
{
    if (!ctx || bytes == 0) return nullptr;

    id<MTLBuffer> buf = [ctx->device newBufferWithLength:(NSUInteger)bytes
                                                 options:MTLResourceStorageModeShared];
    if (!buf) return nullptr;

    void* ptr = [buf contents];
    if (!ptr) return nullptr;

    // Register the buffer in BOTH maps:
    //   - shared_buffers: fast O(1) exact-match for callers that pass the base pointer
    //   - regions:        range lookup for callers that pass base + offset
    //                     (e.g. reading the last token of HiddenState in prefill).
    [ctx->shared_buffers setObject:buf forKey:[NSValue valueWithPointer:ptr]];

    DotLLMRegion* region = [DotLLMRegion new];
    region.base   = ptr;
    region.length = (size_t)bytes;
    region.buffer = buf;
    [ctx->regions addObject:region];

    return ptr;
}

extern "C" void dotllm_metal_free_shared(dotllm_metal_context* ctx, void* ptr)
{
    if (!ctx || !ptr) return;
    [ctx->shared_buffers removeObjectForKey:[NSValue valueWithPointer:ptr]];
    // Drop the matching region too (linear scan — regions count is small).
    NSUInteger idx = [ctx->regions indexOfObjectPassingTest:
        ^BOOL(DotLLMRegion* r, NSUInteger, BOOL*) { return r.base == ptr; }];
    if (idx != NSNotFound)
        [ctx->regions removeObjectAtIndex:idx];
    // ARC releases the MTLBuffer when removed from both collections.
}

// ── Zero-copy registration of caller-owned memory (e.g. GGUF mmap) ───────────
//
// `newBufferWithBytesNoCopy:length:options:deallocator:` requires both the
// pointer and length to be page-aligned (vm_page_size, usually 16 KiB on
// Apple Silicon). Caller must keep the underlying memory alive for the
// buffer's lifetime — we pass a nil deallocator so Metal never touches it.
//
// Typical use: register the entire GGUF mmap once; per-tensor pointers
// (mmap_base + tensor_offset) are then resolved via lookup_shared_region
// which returns the (buffer, offset) pair for kernel encoding.

extern "C" int dotllm_metal_register_buffer(
    dotllm_metal_context* ctx, const void* ptr, size_t bytes)
{
    if (!ctx || !ptr || bytes == 0) return -1;

    const size_t page = (size_t)getpagesize();
    if (((uintptr_t)ptr & (page - 1)) != 0) {
        NSLog(@"dotllm_metal_register_buffer: pointer %p is not page-aligned (page=%zu)",
              ptr, page);
        return -2;
    }

    // Round length up to the next page multiple — Metal mandates it, and
    // since the trailing pages are owned by the caller (mmap allocates by
    // whole pages anyway), this is safe.
    size_t aligned = (bytes + page - 1) & ~(page - 1);

    id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:(void*)ptr
                                                       length:(NSUInteger)aligned
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    if (!buf) {
        NSLog(@"dotllm_metal_register_buffer: newBufferWithBytesNoCopy failed "
              @"(ptr=%p, bytes=%zu, aligned=%zu)", ptr, bytes, aligned);
        return -3;
    }

    DotLLMRegion* region = [DotLLMRegion new];
    region.base   = ptr;
    region.length = aligned;
    region.buffer = buf;
    [ctx->regions addObject:region];
    return 0;
}

extern "C" void dotllm_metal_unregister_buffer(
    dotllm_metal_context* ctx, const void* ptr)
{
    if (!ctx || !ptr) return;
    NSUInteger idx = [ctx->regions indexOfObjectPassingTest:
        ^BOOL(DotLLMRegion* r, NSUInteger, BOOL*) { return r.base == ptr; }];
    if (idx != NSNotFound)
        [ctx->regions removeObjectAtIndex:idx];
    // ARC releases the MTLBuffer when the region object is removed.
}

// Returns the MTLBuffer backing `ptr` if it was allocated via alloc_shared,
// otherwise nil. Used by kernels to detect GPU-resident buffers and skip the
// CPU→Metal copy. Phase 2 will route most kernels through this lookup.
// For exact-pointer hits (alloc_shared), out_offset is set to 0.
// For range hits (register_buffer), out_offset = ptr - region.base.
static id<MTLBuffer> lookup_shared_buffer(
    dotllm_metal_context* ctx, const void* ptr, NSUInteger* out_offset)
{
    if (out_offset) *out_offset = 0;
    if (!ctx || !ptr) return nil;

    // Fast path: exact-match (alloc_shared pointers).
    id<MTLBuffer> exact = [ctx->shared_buffers
        objectForKey:[NSValue valueWithPointer:(void*)ptr]];
    if (exact) return exact;

    // Slow path: range scan over registered regions. Number of regions is
    // tiny (typically 1: the GGUF mmap) so a linear scan is fine.
    const uintptr_t addr = (uintptr_t)ptr;
    for (DotLLMRegion* r in ctx->regions) {
        const uintptr_t base = (uintptr_t)r.base;
        if (addr >= base && addr < base + r.length) {
            if (out_offset) *out_offset = (NSUInteger)(addr - base);
            return r.buffer;
        }
    }
    return nil;
}

// ── Command-buffer batching (Phase 2b) ───────────────────────────────────────
//
// Opens a long-lived command buffer + compute encoder on the context. While
// active, all run_* helpers skip their per-kernel commit/waitUntilCompleted
// and encode their dispatch into the existing encoder. The caller MUST pair
// every begin_forward with an end_forward in the same thread — otherwise the
// encoder leaks and the next forward will assert.
//
// Returns 0 on success, negative on error.

extern "C" int dotllm_metal_begin_forward(dotllm_metal_context* ctx)
{
    if (!ctx) return -10;
    if (ctx->active_enc != nil || ctx->active_cmd != nil) {
        NSLog(@"dotllm_metal_begin_forward: a forward is already in flight");
        return -2;
    }
    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    if (!cmd) return -8;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return -9;
    ctx->active_cmd = cmd;
    ctx->active_enc = enc;
    return 0;
}

// Closes the active encoder, commits the command buffer, and BLOCKS until the
// GPU finishes. After this call the host can safely read any shared buffer
// the kernels wrote to.
extern "C" int dotllm_metal_end_forward(dotllm_metal_context* ctx)
{
    if (!ctx) return -10;
    if (ctx->active_enc == nil || ctx->active_cmd == nil) {
        NSLog(@"dotllm_metal_end_forward: no forward in flight");
        return -2;
    }
    [ctx->active_enc endEncoding];
    [ctx->active_cmd commit];
    [ctx->active_cmd waitUntilCompleted];
    int rc = (ctx->active_cmd.error != nil) ? -11 : 0;
    ctx->active_enc = nil;
    ctx->active_cmd = nil;
    return rc;
}

// ── GPU buffer-to-buffer copy (Phase 2c — full-forward batching) ─────────────
//
// Encodes a copy from `src` to `dst` (both must be GPU-visible — found in
// shared_buffers or registered regions). In batched mode, the copy is enqueued
// as a blit in the active command buffer; subsequent compute dispatches see
// the result without any CPU sync. In standalone mode, the call opens a
// one-shot blit command buffer and waits — equivalent semantically to memcpy.
//
// Returns 0 on success, negative on failure.

extern "C" int dotllm_metal_buffer_copy(
    dotllm_metal_context* ctx, void* dst, const void* src, size_t bytes)
{
    if (!ctx || !dst || !src || bytes == 0) return -10;

    NSUInteger srcOff = 0, dstOff = 0;
    id<MTLBuffer> srcBuf = lookup_shared_buffer(ctx, src, &srcOff);
    id<MTLBuffer> dstBuf = lookup_shared_buffer(ctx, dst, &dstOff);

    if (!srcBuf || !dstBuf) {
        // Pointers aren't registered. In standalone mode we can fall back to
        // a plain CPU memcpy (both are presumed to be in host RAM). In batched
        // mode this is a programmer error: the source might not be up-to-date
        // yet (kernel encoded but not executed).
        if (ctx->active_enc != nil) {
            NSLog(@"dotllm_metal_buffer_copy: pointer not registered "
                  @"during batched forward (src=%p dst=%p)", src, dst);
            return -2;
        }
        memcpy(dst, src, bytes);
        return 0;
    }

    if (ctx->active_enc != nil) {
        // Batched mode: pause the compute encoder, run a blit, resume compute.
        [ctx->active_enc endEncoding];
        ctx->active_enc = nil;

        id<MTLBlitCommandEncoder> blit = [ctx->active_cmd blitCommandEncoder];
        if (!blit) return -9;
        [blit copyFromBuffer:srcBuf sourceOffset:srcOff
                    toBuffer:dstBuf destinationOffset:dstOff
                        size:(NSUInteger)bytes];
        [blit endEncoding];

        ctx->active_enc = [ctx->active_cmd computeCommandEncoder];
        if (!ctx->active_enc) return -9;
        return 0;
    }

    // Standalone: one-shot command buffer.
    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    if (!cmd) return -8;
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    if (!blit) return -9;
    [blit copyFromBuffer:srcBuf sourceOffset:srcOff
                toBuffer:dstBuf destinationOffset:dstOff
                    size:(NSUInteger)bytes];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return (cmd.error != nil) ? -11 : 0;
}

// ── Encoder lifecycle helpers (Phase 2b) ─────────────────────────────────────
//
// Most run_* helpers follow the same skeleton:
//   1. obtain a compute encoder (either the batched one or a fresh standalone)
//   2. configure pipeline / setBuffer / setBytes / dispatch
//   3. either commit+wait+copyback (standalone) or just leave the dispatch
//      encoded (batched — end_forward will commit later)
//
// We capture the standalone/batched choice in a tiny struct so each kernel
// only needs two extra lines: begin_kernel(...) and finish_kernel(...).

typedef struct {
    id<MTLCommandBuffer>          cmd;   // nil in batched mode
    id<MTLComputeCommandEncoder>  enc;
    bool                          batched;
} dotllm_kernel_scope;

static inline dotllm_kernel_scope begin_kernel(dotllm_metal_context* ctx)
{
    dotllm_kernel_scope s;
    if (ctx->active_enc != nil) {
        s.cmd     = nil;
        s.enc     = ctx->active_enc;
        s.batched = true;
    } else {
        s.cmd     = [ctx->queue commandBuffer];
        s.enc     = (s.cmd != nil) ? [s.cmd computeCommandEncoder] : nil;
        s.batched = false;
    }
    return s;
}

// In standalone mode: endEncoding, commit, wait, return non-zero on error.
// In batched mode: no-op (the active encoder stays open for the next kernel).
// Returns 0 on success.
static inline int finish_kernel(const dotllm_kernel_scope* s)
{
    if (s->batched) return 0;
    [s->enc endEncoding];
    [s->cmd commit];
    [s->cmd waitUntilCompleted];
    return (s->cmd.error != nil) ? -11 : 0;
}

// True when the kernel ran standalone and its `needCopy` memcpy should fire.
// In batched mode the memcpy is unsafe (GPU hasn't run yet); callers must
// guarantee zero-copy hits for outputs when the encoder is active.
static inline bool should_copy_back(const dotllm_kernel_scope* s)
{
    return !s->batched;
}

// Pick a threadgroup size adapted to:
//   (a) the simd width of the device (round up to a full warp — never split one),
//   (b) the amount of work the threadgroup actually has (no point spawning 256
//       threads when only 128 blocks need processing),
//   (c) the kernel's per-pipeline cap (registers/smem may reduce maxTg below
//       1024), and
//   (d) the caller's `cap` — how large the caller is willing to go.
// Callers choose `cap` per kernel: memory-bound elementwise usually 256
// (preserves existing behavior), reduction kernels 512, GEMV 256 (helper will
// auto-shrink when the row has fewer blocks than 256).
static inline NSUInteger pick_tgw(
    id<MTLComputePipelineState> pipeline,
    NSUInteger work_per_group,
    NSUInteger cap)
{
    NSUInteger maxTg = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger warp  = pipeline.threadExecutionWidth;
    if (warp == 0) warp = 32;

    NSUInteger limit = MIN(cap, maxTg);
    NSUInteger ideal = ((work_per_group + warp - 1) / warp) * warp;
    NSUInteger tgw   = MIN(ideal, limit);
    if (tgw < warp) tgw = MIN(warp, limit);
    if (tgw == 0) tgw = 1;
    return tgw;
}

// ── Zero-copy bind helpers (Phase 2) ─────────────────────────────────────────
//
// `bind_input`  : resolves a host pointer to an MTLBuffer the kernel can read.
// `bind_output` : resolves a host pointer to an MTLBuffer the kernel will write.
//
// Hit (pointer found in shared_buffers or registered regions) → returns the
//   underlying MTLBuffer plus the offset within it. No allocation, no copy.
// Miss → allocates a fresh shared MTLBuffer; for inputs we memcpy into it;
//   for outputs we let the caller memcpy back after the GPU finishes.
//
// All bytes are interpreted as a strict count (the buffer is large enough to
// cover [ptr, ptr+bytes)); registration validates the upper bound for us.

static inline id<MTLBuffer> bind_input(
    dotllm_metal_context* ctx, const void* ptr, NSUInteger bytes,
    NSUInteger* out_offset)
{
    NSUInteger offset = 0;
    id<MTLBuffer> hit = lookup_shared_buffer(ctx, ptr, &offset);
    if (hit) {
        if (out_offset) *out_offset = offset;
        return hit;
    }
    if (out_offset) *out_offset = 0;
    return [ctx->device newBufferWithBytes:ptr length:bytes
                                   options:MTLResourceStorageModeShared];
}

// In-place binding (read-modify-write). On hit, reuses the existing MTLBuffer
// + offset (no copy). On miss, wraps the host pointer via newBufferWithBytesNoCopy
// — same constraint as legacy call sites: ptr/length must be page-aligned.
// This is appropriate for buffers the caller has already guaranteed to be
// safe for zero-copy (typically forward-state allocations).
static inline id<MTLBuffer> bind_inout(
    dotllm_metal_context* ctx, void* ptr, NSUInteger bytes,
    NSUInteger* out_offset)
{
    NSUInteger offset = 0;
    id<MTLBuffer> hit = lookup_shared_buffer(ctx, ptr, &offset);
    if (hit) {
        if (out_offset) *out_offset = offset;
        return hit;
    }
    if (out_offset) *out_offset = 0;
    return [ctx->device newBufferWithBytesNoCopy:ptr length:bytes
                                         options:MTLResourceStorageModeShared
                                     deallocator:nil];
}

// needsCopy is set to true when the output pointer is *not* GPU-resident and
// the caller must memcpy from the temp buffer back to the host pointer after
// waitUntilCompleted.
static inline id<MTLBuffer> bind_output(
    dotllm_metal_context* ctx, void* ptr, NSUInteger bytes,
    NSUInteger* out_offset, bool* needs_copy)
{
    NSUInteger offset = 0;
    id<MTLBuffer> hit = lookup_shared_buffer(ctx, ptr, &offset);
    if (hit) {
        if (out_offset) *out_offset = offset;
        if (needs_copy) *needs_copy = false;
        return hit;
    }
    if (out_offset) *out_offset = 0;
    if (needs_copy) *needs_copy = true;
    return [ctx->device newBufferWithLength:bytes
                                    options:MTLResourceStorageModeShared];
}

// `shaderPath` is kept in the signature for backward source compatibility
// with the call sites, but is ignored — all functions now live in the
// pre-loaded dotllm_kernels.metallib. The cache key is just the function name.
static id<MTLComputePipelineState> get_or_create_pipeline(
    dotllm_metal_context* ctx,
    const char* shaderPath,         // unused; retained for call-site compatibility
    const char* functionName)
{
    (void)shaderPath;
    NSString* key = @(functionName);

    id<MTLComputePipelineState> pipeline = ctx->pipelines[key];
    if (pipeline) return pipeline;

    id<MTLFunction> function = [ctx->library newFunctionWithName:key];
    if (!function) {
        NSLog(@"dotllm_metal: function '%s' not found in %@", functionName, kKernelLibraryFileName);
        return nil;
    }

    NSError* error = nil;
    pipeline = [ctx->device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        NSLog(@"dotllm_metal: failed to create pipeline for '%s': %@", functionName, error);
        return nil;
    }

    ctx->pipelines[key] = pipeline;
    return pipeline;
}

static int run_binary_f32_kernel(
    dotllm_metal_context* ctx,
    const char* shaderPath,
    const char* functionName,
    const float* a,
    const float* b,
    float* result,
    uint32_t length)
{
    @autoreleasepool {
        if (!ctx || !a || !b || !result) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, shaderPath, functionName);
        if (!pipeline) return -3;

        NSUInteger bytes = (NSUInteger)length * sizeof(float);
        NSUInteger offA = 0, offB = 0, offR = 0;
        bool needCopyR = false;
        id<MTLBuffer> bufA = bind_input (ctx, a,      bytes, &offA);
        id<MTLBuffer> bufB = bind_input (ctx, b,      bytes, &offB);
        id<MTLBuffer> bufR = bind_output(ctx, result, bytes, &offR, &needCopyR);
        if (!bufA || !bufB || !bufR) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufA offset:offA atIndex:0];
        [enc setBuffer:bufB offset:offB atIndex:1];
        [enc setBuffer:bufR offset:offR atIndex:2];
        [enc setBytes:&length length:sizeof(uint32_t) atIndex:3];

        NSUInteger tgw = pick_tgw(pipeline, (NSUInteger)length, 256);

        [enc dispatchThreads:MTLSizeMake(length, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopyR && should_copy_back(&ks)) memcpy(result, bufR.contents, bytes);
        return 0;
    }
}

static int run_unary_f32_kernel(
    dotllm_metal_context* ctx,
    const char* shaderPath,
    const char* functionName,
    const float* a,
    float* result,
    uint32_t length)
{
    @autoreleasepool {
        if (!ctx || !a || !result) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, shaderPath, functionName);
        if (!pipeline) return -3;

        NSUInteger bytes = (NSUInteger)length * sizeof(float);
        NSUInteger offA = 0, offR = 0;
        bool needCopyR = false;
        id<MTLBuffer> bufA = bind_input (ctx, a,      bytes, &offA);
        id<MTLBuffer> bufR = bind_output(ctx, result, bytes, &offR, &needCopyR);
        if (!bufA || !bufR) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufA offset:offA atIndex:0];
        [enc setBuffer:bufR offset:offR atIndex:2];
        [enc setBytes:&length length:sizeof(uint32_t) atIndex:3];

        NSUInteger tgw = pick_tgw(pipeline, (NSUInteger)length, 256);

        [enc dispatchThreads:MTLSizeMake(length, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopyR && should_copy_back(&ks)) memcpy(result, bufR.contents, bytes);
        return 0;
    }
}

extern "C" int dotllm_metal_add_f32(
    dotllm_metal_context* ctx,
    const float* a,
    const float* b,
    float* result,
    uint32_t length)
{
    return run_binary_f32_kernel(ctx, "add_f32.metal", "add_f32", a, b, result, length);
}

extern "C" int dotllm_metal_add_f16(
    dotllm_metal_context* ctx,
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* result,
    uint32_t length)
{
    @autoreleasepool {
        if (!ctx || !a || !b || !result) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "add.metal", "add_f16");
        if (!pipeline) return -3;

        NSUInteger bytes = (NSUInteger)length * sizeof(uint16_t);
        NSUInteger offA = 0, offB = 0, offR = 0;
        bool needCopyR = false;
        id<MTLBuffer> bufA = bind_input (ctx, a,      bytes, &offA);
        id<MTLBuffer> bufB = bind_input (ctx, b,      bytes, &offB);
        id<MTLBuffer> bufR = bind_output(ctx, result, bytes, &offR, &needCopyR);
        if (!bufA || !bufB || !bufR) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufA offset:offA atIndex:0];
        [enc setBuffer:bufB offset:offB atIndex:1];
        [enc setBuffer:bufR offset:offR atIndex:2];
        [enc setBytes:&length length:sizeof(uint32_t) atIndex:3];

        // Dispatch n/2 threads (vectorized half2); +1 handles the odd-tail guard
        uint32_t dispatch = (length / 2u) + 1u;
        NSUInteger tgw = pick_tgw(pipeline, (NSUInteger)dispatch, 256);

        [enc dispatchThreads:MTLSizeMake(dispatch, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopyR && should_copy_back(&ks)) memcpy(result, bufR.contents, bytes);
        return 0;
    }
}

extern "C" int dotllm_metal_add_f32_f16(
    dotllm_metal_context* ctx,
    const float*    a,
    const uint16_t* b,
    float*          result,
    uint32_t        length)
{
    @autoreleasepool {
        if (!ctx || !a || !b || !result) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "add_f32.metal", "add_f32_f16");
        if (!pipeline) return -3;

        NSUInteger bytesF32 = (NSUInteger)length * sizeof(float);
        NSUInteger bytesF16 = (NSUInteger)length * sizeof(uint16_t);
        NSUInteger offA = 0, offB = 0, offR = 0;
        bool needCopyR = false;
        id<MTLBuffer> bufA = bind_input (ctx, a,      bytesF32, &offA);
        id<MTLBuffer> bufB = bind_input (ctx, b,      bytesF16, &offB);
        id<MTLBuffer> bufR = bind_output(ctx, result, bytesF32, &offR, &needCopyR);
        if (!bufA || !bufB || !bufR) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufA offset:offA atIndex:0];
        [enc setBuffer:bufB offset:offB atIndex:1];
        [enc setBuffer:bufR offset:offR atIndex:2];
        [enc setBytes:&length length:sizeof(uint32_t) atIndex:3];

        NSUInteger tgw = pick_tgw(pipeline, (NSUInteger)length, 256);

        [enc dispatchThreads:MTLSizeMake(length, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopyR && should_copy_back(&ks)) memcpy(result, bufR.contents, bytesF32);
        return 0;
    }
}

extern "C" int dotllm_metal_softmax_f16(
    dotllm_metal_context* ctx,
    const uint16_t* input,
    uint16_t*       output,
    int32_t         rows,
    int32_t         cols)
{
    @autoreleasepool {
        if (!ctx || !input || !output) return -10;
        if (rows <= 0 || cols <= 0)   return 0;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "softmax.metal", "softmax_f16");
        if (!pipeline) return -3;

        NSUInteger inputBytes  = (NSUInteger)(rows * cols) * sizeof(uint16_t);
        NSUInteger outputBytes = inputBytes;

        NSUInteger offIn = 0, offOut = 0;
        bool needCopy = false;
        id<MTLBuffer> bufInput  = bind_input (ctx, input,  inputBytes,  &offIn);
        id<MTLBuffer> bufOutput = bind_output(ctx, output, outputBytes, &offOut, &needCopy);
        if (!bufInput || !bufOutput) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufInput  offset:offIn  atIndex:0];
        [enc setBuffer:bufOutput offset:offOut atIndex:1];
        [enc setBytes:&rows length:sizeof(int32_t) atIndex:2];
        [enc setBytes:&cols length:sizeof(int32_t) atIndex:3];

        // One threadgroup per row, 256 threads per threadgroup
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopy && should_copy_back(&ks)) memcpy(output, bufOutput.contents, outputBytes);
        return 0;
    }
}

extern "C" int dotllm_metal_swiglu_f32(
    dotllm_metal_context* ctx,
    const float* gate,
    const float* up,
    float* result,
    uint32_t length)
{
    return run_binary_f32_kernel(ctx, "swiglu_f32.metal", "swiglu_f32", gate, up, result, length);
}

extern "C" int dotllm_metal_swiglu_f16(
    dotllm_metal_context* ctx,
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t*       result,
    uint32_t        length)
{
    @autoreleasepool {
        if (!ctx || !gate || !up || !result) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "swiglu.metal", "swiglu_f16");
        if (!pipeline) return -3;

        NSUInteger bytes = (NSUInteger)length * sizeof(uint16_t);
        NSUInteger offGate = 0, offUp = 0, offR = 0;
        bool needCopyR = false;
        id<MTLBuffer> bufGate   = bind_input (ctx, gate,   bytes, &offGate);
        id<MTLBuffer> bufUp     = bind_input (ctx, up,     bytes, &offUp);
        id<MTLBuffer> bufResult = bind_output(ctx, result, bytes, &offR, &needCopyR);
        if (!bufGate || !bufUp || !bufResult) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufGate   offset:offGate atIndex:0];
        [enc setBuffer:bufUp     offset:offUp   atIndex:1];
        [enc setBuffer:bufResult offset:offR    atIndex:2];
        [enc setBytes:&length length:sizeof(uint32_t) atIndex:3];

        // Dispatch length/2 + 1 threads — vectorized half2 path + odd-tail guard
        uint32_t dispatch = (length / 2u) + 1u;
        NSUInteger tgw = pick_tgw(pipeline, (NSUInteger)dispatch, 256);

        [enc dispatchThreads:MTLSizeMake(dispatch, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopyR && should_copy_back(&ks)) memcpy(result, bufResult.contents, bytes);
        return 0;
    }
}

extern "C" int dotllm_metal_bias_add_f32(
    dotllm_metal_context* ctx,
    float*          output,
    const uint16_t* bias,       // FP16 bias — matches bias_add_f32.cu
    uint32_t        dim,
    uint32_t        seq_len)
{
    @autoreleasepool {
        if (!ctx || !output || !bias) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "bias_add_f32.metal", "bias_add_f32");
        if (!pipeline) return -3;

        uint32_t total = dim * seq_len;
        NSUInteger outputBytes = (NSUInteger)total * sizeof(float);
        NSUInteger biasBytes   = (NSUInteger)dim   * sizeof(uint16_t);

        // output is in-place: zero-copy on hit, copy-in-and-out fallback on miss.
        NSUInteger offOut = 0, offBias = 0;
        bool needCopy = false;
        // Manual in-place: if there's no shared mapping we need a copy IN+OUT.
        NSUInteger offHit = 0;
        id<MTLBuffer> bufOutput = lookup_shared_buffer(ctx, output, &offHit);
        if (bufOutput) {
            offOut = offHit;
            needCopy = false;
        } else {
            bufOutput = [ctx->device newBufferWithBytes:output length:outputBytes
                                                options:MTLResourceStorageModeShared];
            needCopy = true;
        }
        id<MTLBuffer> bufBias = bind_input(ctx, bias, biasBytes, &offBias);
        if (!bufOutput || !bufBias) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufOutput offset:offOut  atIndex:0];
        [enc setBuffer:bufBias   offset:offBias atIndex:1];
        [enc setBytes:&dim     length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&seq_len length:sizeof(uint32_t) atIndex:3];

        NSUInteger tgw = pick_tgw(pipeline, (NSUInteger)total, 256);

        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopy && should_copy_back(&ks)) memcpy(output, bufOutput.contents, outputBytes);
        return 0;
    }
}

extern "C" int dotllm_metal_bias_add_f16(
    dotllm_metal_context* ctx,
    uint16_t*       output,     // FP16 in-place
    const uint16_t* bias,       // FP16 bias
    uint32_t        dim,
    uint32_t        seq_len)
{
    @autoreleasepool {
        if (!ctx || !output || !bias) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "bias_add.metal", "bias_add_f16");
        if (!pipeline) return -3;

        uint32_t total = dim * seq_len;
        NSUInteger outputBytes = (NSUInteger)total * sizeof(uint16_t);
        NSUInteger biasBytes   = (NSUInteger)dim   * sizeof(uint16_t);

        NSUInteger offOut = 0, offBias = 0;
        bool needCopy = false;
        NSUInteger offHit = 0;
        id<MTLBuffer> bufOutput = lookup_shared_buffer(ctx, output, &offHit);
        if (bufOutput) {
            offOut = offHit;
        } else {
            bufOutput = [ctx->device newBufferWithBytes:output length:outputBytes
                                                options:MTLResourceStorageModeShared];
            needCopy = true;
        }
        id<MTLBuffer> bufBias = bind_input(ctx, bias, biasBytes, &offBias);
        if (!bufOutput || !bufBias) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufOutput offset:offOut  atIndex:0];
        [enc setBuffer:bufBias   offset:offBias atIndex:1];
        [enc setBytes:&dim     length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&seq_len length:sizeof(uint32_t) atIndex:3];

        // Dispatch total/2 + 1 threads — vectorized half2 path + odd-tail guard
        uint32_t dispatch = (total / 2u) + 1u;
        NSUInteger tgw = pick_tgw(pipeline, (NSUInteger)dispatch, 256);

        [enc dispatchThreads:MTLSizeMake(dispatch, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopy && should_copy_back(&ks)) memcpy(output, bufOutput.contents, outputBytes);
        return 0;
    }
}

extern "C" int dotllm_metal_rope_f32(
    dotllm_metal_context* ctx,
    float*         q,
    float*         k,
    const int32_t* positions,
    int32_t        seq_len,
    int32_t        num_heads,
    int32_t        num_kv_heads,
    int32_t        head_dim,
    int32_t        rope_dim,
    float          theta,
    int32_t        rope_type)   // 0 = norm, 1 = neox
{
    @autoreleasepool {
        if (!ctx || !q || !k || !positions) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "rope_f32.metal", "rope_f32");
        if (!pipeline) return -3;

        int32_t half_rope     = rope_dim / 2;
        uint32_t total_q      = (uint32_t)(seq_len * num_heads    * half_rope);
        uint32_t total_k      = (uint32_t)(seq_len * num_kv_heads * half_rope);
        uint32_t dispatch_len = total_q > total_k ? total_q : total_k;

        NSUInteger qBytes   = (NSUInteger)(seq_len * num_heads    * head_dim) * sizeof(float);
        NSUInteger kBytes   = (NSUInteger)(seq_len * num_kv_heads * head_dim) * sizeof(float);
        NSUInteger posBytes = (NSUInteger)seq_len * sizeof(int32_t);

        NSUInteger offQ = 0, offK = 0, offPos = 0;
        id<MTLBuffer> bufQ   = bind_inout(ctx, q, qBytes, &offQ);
        id<MTLBuffer> bufK   = bind_inout(ctx, k, kBytes, &offK);
        id<MTLBuffer> bufPos = bind_input(ctx, positions, posBytes, &offPos);
        if (!bufQ || !bufK || !bufPos) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufQ   offset:offQ   atIndex:0];
        [enc setBuffer:bufK   offset:offK   atIndex:1];
        [enc setBuffer:bufPos offset:offPos atIndex:2];
        [enc setBytes:&seq_len      length:sizeof(int32_t) atIndex:3];
        [enc setBytes:&num_heads    length:sizeof(int32_t) atIndex:4];
        [enc setBytes:&num_kv_heads length:sizeof(int32_t) atIndex:5];
        [enc setBytes:&head_dim     length:sizeof(int32_t) atIndex:6];
        [enc setBytes:&rope_dim     length:sizeof(int32_t) atIndex:7];
        [enc setBytes:&theta        length:sizeof(float)   atIndex:8];
        [enc setBytes:&rope_type    length:sizeof(int32_t) atIndex:9];

        NSUInteger tgw = pick_tgw(pipeline, (NSUInteger)dispatch_len, 256);

        [enc dispatchThreads:MTLSizeMake(dispatch_len, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;
        return 0;
    }
}

extern "C" int dotllm_metal_rope_f16(
    dotllm_metal_context* ctx,
    uint16_t*      q,
    uint16_t*      k,
    const int32_t* positions,
    int32_t        seq_len,
    int32_t        num_heads,
    int32_t        num_kv_heads,
    int32_t        head_dim,
    int32_t        rope_dim,
    float          theta,
    int32_t        rope_type)
{
    @autoreleasepool {
        if (!ctx || !q || !k || !positions) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "rope.metal", "rope_f16");
        if (!pipeline) return -3;

        int32_t half_rope     = rope_dim / 2;
        uint32_t total_q      = (uint32_t)(seq_len * num_heads    * half_rope);
        uint32_t total_k      = (uint32_t)(seq_len * num_kv_heads * half_rope);
        uint32_t dispatch_len = total_q > total_k ? total_q : total_k;

        NSUInteger qBytes   = (NSUInteger)(seq_len * num_heads    * head_dim) * sizeof(uint16_t);
        NSUInteger kBytes   = (NSUInteger)(seq_len * num_kv_heads * head_dim) * sizeof(uint16_t);
        NSUInteger posBytes = (NSUInteger)seq_len * sizeof(int32_t);

        NSUInteger offQ = 0, offK = 0, offPos = 0;
        id<MTLBuffer> bufQ   = bind_inout(ctx, q, qBytes, &offQ);
        id<MTLBuffer> bufK   = bind_inout(ctx, k, kBytes, &offK);
        id<MTLBuffer> bufPos = bind_input(ctx, positions, posBytes, &offPos);
        if (!bufQ || !bufK || !bufPos) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufQ   offset:offQ   atIndex:0];
        [enc setBuffer:bufK   offset:offK   atIndex:1];
        [enc setBuffer:bufPos offset:offPos atIndex:2];
        [enc setBytes:&seq_len      length:sizeof(int32_t) atIndex:3];
        [enc setBytes:&num_heads    length:sizeof(int32_t) atIndex:4];
        [enc setBytes:&num_kv_heads length:sizeof(int32_t) atIndex:5];
        [enc setBytes:&head_dim     length:sizeof(int32_t) atIndex:6];
        [enc setBytes:&rope_dim     length:sizeof(int32_t) atIndex:7];
        [enc setBytes:&theta        length:sizeof(float)   atIndex:8];
        [enc setBytes:&rope_type    length:sizeof(int32_t) atIndex:9];

        NSUInteger tgw = pick_tgw(pipeline, (NSUInteger)dispatch_len, 256);

        [enc dispatchThreads:MTLSizeMake(dispatch_len, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;
        return 0;
    }
}

extern "C" int dotllm_metal_rmsnorm_f32(
    dotllm_metal_context* ctx,
    const float* input,
    const float* weight,
    float*       output,
    int32_t      n,
    int32_t      seq_len,
    float        eps)
{
    @autoreleasepool {
        if (!ctx || !input || !weight || !output) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "rmsnorm_f32.metal", "rmsnorm_f32");
        if (!pipeline) return -3;

        // Threadgroup size: scaled to hidden dim. The kernel is reduction-heavy
        // (mean of squares over n elements) so we prefer 512 threads per group
        // when n is large enough, falling back to the simd width for tiny rows.
        // `pick_tgw` honors the pipeline cap and rounds to a multiple of warp.
        // The kernel's `simd_sums[32]` scratch is sized for up to 1024 threads.
        uint32_t tgSize = (uint32_t)pick_tgw(pipeline, (NSUInteger)n, 512);

        NSUInteger inputBytes  = (NSUInteger)(seq_len * n) * sizeof(float);
        NSUInteger weightBytes = (NSUInteger)n * sizeof(float);

        NSUInteger offIn = 0, offW = 0, offOut = 0;
        bool needCopy = false;
        id<MTLBuffer> bufIn  = bind_input (ctx, input,  inputBytes,  &offIn);
        id<MTLBuffer> bufW   = bind_input (ctx, weight, weightBytes, &offW);
        id<MTLBuffer> bufOut = bind_output(ctx, output, inputBytes,  &offOut, &needCopy);
        if (!bufIn || !bufW || !bufOut) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufIn  offset:offIn  atIndex:0];
        [enc setBuffer:bufW   offset:offW   atIndex:1];
        [enc setBuffer:bufOut offset:offOut atIndex:2];
        [enc setBytes:&n   length:sizeof(int32_t) atIndex:3];
        [enc setBytes:&eps length:sizeof(float)   atIndex:4];

        // dispatchThreadgroups: launches exactly seq_len groups.
        // Each group = one token; tgSize threads collaborate on n elements.
        // Unlike dispatchThreads, this gives explicit control over group count.
        [enc dispatchThreadgroups:MTLSizeMake(seq_len, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopy && should_copy_back(&ks)) memcpy(output, bufOut.contents, inputBytes);
        return 0;
    }
}

extern "C" int dotllm_metal_rmsnorm_f16(
    dotllm_metal_context* ctx,
    const uint16_t* input,
    const uint16_t* weight,
    uint16_t*       output,
    int32_t         n,
    int32_t         seq_len,
    float           eps)
{
    @autoreleasepool {
        if (!ctx || !input || !weight || !output) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "rmsnorm.metal", "rmsnorm_f16");
        if (!pipeline) return -3;

        // See rmsnorm_f32: 512 preferred, capped by the pipeline.
        uint32_t tgSize = (uint32_t)pick_tgw(pipeline, (NSUInteger)n, 512);

        NSUInteger inputBytes  = (NSUInteger)(seq_len * n) * sizeof(uint16_t);
        NSUInteger weightBytes = (NSUInteger)n * sizeof(uint16_t);

        NSUInteger offIn = 0, offW = 0, offOut = 0;
        bool needCopy = false;
        id<MTLBuffer> bufIn  = bind_input (ctx, input,  inputBytes,  &offIn);
        id<MTLBuffer> bufW   = bind_input (ctx, weight, weightBytes, &offW);
        id<MTLBuffer> bufOut = bind_output(ctx, output, inputBytes,  &offOut, &needCopy);
        if (!bufIn || !bufW || !bufOut) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufIn  offset:offIn  atIndex:0];
        [enc setBuffer:bufW   offset:offW   atIndex:1];
        [enc setBuffer:bufOut offset:offOut atIndex:2];
        [enc setBytes:&n   length:sizeof(int32_t) atIndex:3];
        [enc setBytes:&eps length:sizeof(float)   atIndex:4];

        [enc dispatchThreadgroups:MTLSizeMake(seq_len, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopy && should_copy_back(&ks)) memcpy(output, bufOut.contents, inputBytes);
        return 0;
    }
}

// Helper shared by both convert kernels.
// srcElemSize / dstElemSize: sizeof(half)=2, sizeof(float)=4.
static int run_convert_kernel(
    dotllm_metal_context* ctx,
    const char* functionName,
    const void* src,
    void*       dst,
    int32_t     n,
    size_t      srcElemSize,
    size_t      dstElemSize)
{
    @autoreleasepool {
        if (!ctx || !src || !dst || n <= 0) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "convert.metal", functionName);
        if (!pipeline) return -3;

        NSUInteger srcBytes = (NSUInteger)n * srcElemSize;
        NSUInteger dstBytes = (NSUInteger)n * dstElemSize;

        NSUInteger offSrc = 0, offDst = 0;
        bool needCopy = false;
        id<MTLBuffer> bufSrc = bind_input (ctx, src, srcBytes, &offSrc);
        id<MTLBuffer> bufDst = bind_output(ctx, dst, dstBytes, &offDst, &needCopy);
        if (!bufSrc || !bufDst) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufSrc offset:offSrc atIndex:0];
        [enc setBuffer:bufDst offset:offDst atIndex:1];
        [enc setBytes:&n length:sizeof(int32_t) atIndex:2];

        NSUInteger tgw = pick_tgw(pipeline, (NSUInteger)n, 256);

        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopy && should_copy_back(&ks)) memcpy(dst, bufDst.contents, dstBytes);
        return 0;
    }
}

extern "C" int dotllm_metal_per_head_rmsnorm_f32(
    dotllm_metal_context* ctx,
    float*       qk,
    const float* weight,
    int32_t      num_heads,
    int32_t      head_dim,
    int32_t      seq_len,
    float        eps)
{
    @autoreleasepool {
        if (!ctx || !qk || !weight) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "per_head_rmsnorm_f32.metal", "per_head_rmsnorm_f32");
        if (!pipeline) return -3;

        // Per-head reduction: work per group = head_dim (typically 64-128).
        // Cap at 128 so we don't waste threads on tiny rows.
        uint32_t tgSize   = (uint32_t)pick_tgw(pipeline, (NSUInteger)head_dim, 128);
        uint32_t nGroups  = (uint32_t)(seq_len * num_heads); // one threadgroup per (token, head)

        NSUInteger qkBytes     = (NSUInteger)(seq_len * num_heads * head_dim) * sizeof(float);
        NSUInteger weightBytes = (NSUInteger)head_dim * sizeof(float);

        NSUInteger offQK = 0, offW = 0;
        id<MTLBuffer> bufQK     = bind_inout(ctx, qk,     qkBytes,     &offQK);
        id<MTLBuffer> bufWeight = bind_input(ctx, weight, weightBytes, &offW);
        if (!bufQK || !bufWeight) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufQK     offset:offQK atIndex:0];
        [enc setBuffer:bufWeight offset:offW  atIndex:1];
        [enc setBytes:&eps       length:sizeof(float)   atIndex:2];
        [enc setBytes:&num_heads length:sizeof(int32_t) atIndex:3];
        [enc setBytes:&head_dim  length:sizeof(int32_t) atIndex:4];
        [enc setBytes:&seq_len   length:sizeof(int32_t) atIndex:5];

        // One threadgroup per (token, head) pair.
        [enc dispatchThreadgroups:MTLSizeMake(nGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;
        // bufQK wraps host memory directly — no memcpy needed.
        return 0;
    }
}

extern "C" int dotllm_metal_per_head_rmsnorm_f16(
    dotllm_metal_context* ctx,
    uint16_t*       qk,
    const uint16_t* weight,
    int32_t         num_heads,
    int32_t         head_dim,
    int32_t         seq_len,
    float           eps)
{
    @autoreleasepool {
        if (!ctx || !qk || !weight) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "per_head_rmsnorm.metal", "per_head_rmsnorm_f16");
        if (!pipeline) return -3;

        // See per_head_rmsnorm_f32: cap 128 for head_dim-sized reductions.
        uint32_t tgSize  = (uint32_t)pick_tgw(pipeline, (NSUInteger)head_dim, 128);
        uint32_t nGroups = (uint32_t)(seq_len * num_heads); // one threadgroup per (token, head)

        NSUInteger qkBytes     = (NSUInteger)(seq_len * num_heads * head_dim) * sizeof(uint16_t);
        NSUInteger weightBytes = (NSUInteger)head_dim * sizeof(uint16_t);

        NSUInteger offQK = 0, offW = 0;
        id<MTLBuffer> bufQK     = bind_inout(ctx, qk,     qkBytes,     &offQK);
        id<MTLBuffer> bufWeight = bind_input(ctx, weight, weightBytes, &offW);
        if (!bufQK || !bufWeight) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufQK     offset:offQK atIndex:0];
        [enc setBuffer:bufWeight offset:offW  atIndex:1];
        [enc setBytes:&eps       length:sizeof(float)   atIndex:2];
        [enc setBytes:&num_heads length:sizeof(int32_t) atIndex:3];
        [enc setBytes:&head_dim  length:sizeof(int32_t) atIndex:4];
        [enc setBytes:&seq_len   length:sizeof(int32_t) atIndex:5];

        [enc dispatchThreadgroups:MTLSizeMake(nGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;
        return 0;
    }
}

extern "C" int dotllm_metal_rmsnorm_f32in_f16out(
    dotllm_metal_context* ctx,
    const float*    input,
    const float*    weight,
    uint16_t*       output,
    int32_t         n,
    int32_t         seq_len,
    float           eps)
{
    @autoreleasepool {
        if (!ctx || !input || !weight || !output) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "rmsnorm_f32in.metal", "rmsnorm_f32in_f16out");
        if (!pipeline) return -3;

        // Reduction over hidden dim; allow up to 512 threads for better occupancy.
        uint32_t tgSize = (uint32_t)pick_tgw(pipeline, (NSUInteger)n, 512);

        NSUInteger inputBytes  = (NSUInteger)(seq_len * n) * sizeof(float);
        NSUInteger outputBytes = (NSUInteger)(seq_len * n) * sizeof(uint16_t);
        NSUInteger weightBytes = (NSUInteger)n * sizeof(float);

        NSUInteger offIn = 0, offW = 0, offOut = 0;
        bool needCopy = false;
        id<MTLBuffer> bufIn  = bind_input (ctx, input,  inputBytes,  &offIn);
        id<MTLBuffer> bufW   = bind_input (ctx, weight, weightBytes, &offW);
        id<MTLBuffer> bufOut = bind_output(ctx, output, outputBytes, &offOut, &needCopy);
        if (!bufIn || !bufW || !bufOut) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufIn  offset:offIn  atIndex:0];
        [enc setBuffer:bufW   offset:offW   atIndex:1];
        [enc setBuffer:bufOut offset:offOut atIndex:2];
        [enc setBytes:&n   length:sizeof(int32_t) atIndex:3];
        [enc setBytes:&eps length:sizeof(float)   atIndex:4];

        [enc dispatchThreadgroups:MTLSizeMake(seq_len, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        memcpy(output, bufOut.contents, outputBytes);
        return 0;
    }
}

extern "C" int dotllm_metal_fused_add_rmsnorm_f16(
    dotllm_metal_context* ctx,
    uint16_t*       residual,
    const uint16_t* x,
    const uint16_t* weight,
    uint16_t*       output,
    int32_t         n,
    int32_t         seq_len,
    float           eps)
{
    @autoreleasepool {
        if (!ctx || !residual || !x || !weight || !output) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "fused_add_rmsnorm.metal", "fused_add_rmsnorm_f16");
        if (!pipeline) return -3;

        // Reduction over hidden dim; cap at 512.
        uint32_t tgSize = (uint32_t)pick_tgw(pipeline, (NSUInteger)n, 512);

        NSUInteger rowBytes    = (NSUInteger)n * sizeof(uint16_t);
        NSUInteger totalBytes  = (NSUInteger)(seq_len * n) * sizeof(uint16_t);

        // residual is read AND written by the kernel — needs both directions on miss.
        NSUInteger offRes = 0, offX = 0, offW = 0, offOut = 0;
        bool needCopyRes = false, needCopyOut = false;
        NSUInteger offHit = 0;
        id<MTLBuffer> bufRes = lookup_shared_buffer(ctx, residual, &offHit);
        if (bufRes) {
            offRes = offHit;
        } else {
            bufRes = [ctx->device newBufferWithBytes:residual length:totalBytes
                                             options:MTLResourceStorageModeShared];
            needCopyRes = true;
        }
        id<MTLBuffer> bufX   = bind_input (ctx, x,      totalBytes, &offX);
        id<MTLBuffer> bufW   = bind_input (ctx, weight, rowBytes,   &offW);
        id<MTLBuffer> bufOut = bind_output(ctx, output, totalBytes, &offOut, &needCopyOut);

        if (!bufRes || !bufX || !bufW || !bufOut) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufRes  offset:offRes atIndex:0];
        [enc setBuffer:bufX    offset:offX   atIndex:1];
        [enc setBuffer:bufW    offset:offW   atIndex:2];
        [enc setBuffer:bufOut  offset:offOut atIndex:3];
        [enc setBytes:&n   length:sizeof(int32_t) atIndex:4];
        [enc setBytes:&eps length:sizeof(float)   atIndex:5];

        // One threadgroup per token.
        [enc dispatchThreadgroups:MTLSizeMake(seq_len, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopyRes && should_copy_back(&ks)) memcpy(residual, bufRes.contents, totalBytes);
        if (needCopyOut && should_copy_back(&ks)) memcpy(output,   bufOut.contents, totalBytes);
        return 0;
    }
}

// ── Embedding lookup helper ───────────────────────────────────────────────────
// All variants share the same buffer layout and dispatch pattern.
// embed_table_bytes: total byte size of the embed table (varies by dtype).
// out_elem_bytes: size of one output element (4 for FP32, 2 for FP16).
static int run_embedding_kernel(
    dotllm_metal_context* ctx,
    const char*  functionName,
    const void*  embed_table,
    NSUInteger   embed_table_bytes,
    const int32_t* token_ids,
    void*        output,
    NSUInteger   out_elem_bytes,
    int32_t      hidden_size,
    int32_t      seq_len)
{
    @autoreleasepool {
        if (!ctx || !embed_table || !token_ids || !output) return -10;

        // Source file is cosmetic (kernels live in the single metallib), but keep
        // it honest: f32-out kernels are in embedding_f32out.metal, f16-out kernels
        // in embedding.metal (port of embedding.cu).
        const char* shaderFile = strstr(functionName, "_f32out")
                               ? "embedding_f32out.metal"
                               : "embedding.metal";
        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, shaderFile, functionName);
        if (!pipeline) return -3;

        // One threadgroup per token; threads copy/dequantize hidden_size elements.
        uint32_t tgSize     = (uint32_t)pick_tgw(pipeline, (NSUInteger)hidden_size, 256);
        NSUInteger posBytes = (NSUInteger)seq_len * sizeof(int32_t);
        NSUInteger outBytes = (NSUInteger)(seq_len * hidden_size) * out_elem_bytes;

        NSUInteger offTable = 0, offIds = 0, offOut = 0;
        bool needCopy = false;
        id<MTLBuffer> bufTable = bind_input (ctx, embed_table, embed_table_bytes, &offTable);
        id<MTLBuffer> bufIds   = bind_input (ctx, token_ids,   posBytes,          &offIds);
        id<MTLBuffer> bufOut   = bind_output(ctx, output,      outBytes,          &offOut, &needCopy);
        if (!bufTable || !bufIds || !bufOut) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufTable offset:offTable atIndex:0];
        [enc setBuffer:bufIds   offset:offIds   atIndex:1];
        [enc setBuffer:bufOut   offset:offOut   atIndex:2];
        [enc setBytes:&seq_len     length:sizeof(int32_t) atIndex:3];
        [enc setBytes:&hidden_size length:sizeof(int32_t) atIndex:4];

        // One threadgroup per token; threads copy/dequantize hidden_size elements.
        [enc dispatchThreadgroups:MTLSizeMake(seq_len, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopy && should_copy_back(&ks)) memcpy(output, bufOut.contents, outBytes);
        return 0;
    }
}

extern "C" int dotllm_metal_embedding_f32_f32out(
    dotllm_metal_context* ctx,
    const float*   embed_table,
    const int32_t* token_ids,
    float*         output,
    int32_t        vocab_size,
    int32_t        hidden_size,
    int32_t        seq_len)
{
    NSUInteger tableBytes = (NSUInteger)(vocab_size * hidden_size) * sizeof(float);
    return run_embedding_kernel(ctx, "embedding_lookup_f32_f32out",
        embed_table, tableBytes, token_ids, output, sizeof(float), hidden_size, seq_len);
}

extern "C" int dotllm_metal_embedding_f16_f32out(
    dotllm_metal_context* ctx,
    const uint16_t* embed_table,
    const int32_t*  token_ids,
    float*          output,
    int32_t         vocab_size,
    int32_t         hidden_size,
    int32_t         seq_len)
{
    NSUInteger tableBytes = (NSUInteger)(vocab_size * hidden_size) * sizeof(uint16_t);
    return run_embedding_kernel(ctx, "embedding_lookup_f16_f32out",
        embed_table, tableBytes, token_ids, output, sizeof(float), hidden_size, seq_len);
}

extern "C" int dotllm_metal_embedding_q8_0_f32out(
    dotllm_metal_context* ctx,
    const uint8_t* embed_table,
    const int32_t* token_ids,
    float*         output,
    int32_t        vocab_size,
    int32_t        hidden_size,
    int32_t        seq_len)
{
    const int32_t Q8_0_BLOCK_SIZE  = 32;
    const int32_t Q8_0_BLOCK_BYTES = 34;
    int32_t blocks_per_row = hidden_size / Q8_0_BLOCK_SIZE;
    NSUInteger tableBytes = (NSUInteger)(vocab_size * blocks_per_row) * Q8_0_BLOCK_BYTES;
    return run_embedding_kernel(ctx, "embedding_lookup_q8_0_f32out",
        embed_table, tableBytes, token_ids, output, sizeof(float), hidden_size, seq_len);
}

extern "C" int dotllm_metal_embedding_f32_f16out(
    dotllm_metal_context* ctx,
    const float*    embed_table,
    const int32_t*  token_ids,
    uint16_t*       output,
    int32_t         vocab_size,
    int32_t         hidden_size,
    int32_t         seq_len)
{
    NSUInteger tableBytes = (NSUInteger)(vocab_size * hidden_size) * sizeof(float);
    return run_embedding_kernel(ctx, "embedding_lookup_f32_f16out",
        embed_table, tableBytes, token_ids, output, sizeof(uint16_t), hidden_size, seq_len);
}

extern "C" int dotllm_metal_embedding_f16_f16out(
    dotllm_metal_context* ctx,
    const uint16_t* embed_table,
    const int32_t*  token_ids,
    uint16_t*       output,
    int32_t         vocab_size,
    int32_t         hidden_size,
    int32_t         seq_len)
{
    NSUInteger tableBytes = (NSUInteger)(vocab_size * hidden_size) * sizeof(uint16_t);
    return run_embedding_kernel(ctx, "embedding_lookup_f16_f16out",
        embed_table, tableBytes, token_ids, output, sizeof(uint16_t), hidden_size, seq_len);
}

extern "C" int dotllm_metal_embedding_q8_0_f16out(
    dotllm_metal_context* ctx,
    const uint8_t*  embed_table,
    const int32_t*  token_ids,
    uint16_t*       output,
    int32_t         vocab_size,
    int32_t         hidden_size,
    int32_t         seq_len)
{
    const int32_t Q8_0_BLOCK_SIZE  = 32;
    const int32_t Q8_0_BLOCK_BYTES = 34;
    int32_t blocks_per_row = hidden_size / Q8_0_BLOCK_SIZE;
    NSUInteger tableBytes = (NSUInteger)(vocab_size * blocks_per_row) * Q8_0_BLOCK_BYTES;
    return run_embedding_kernel(ctx, "embedding_lookup_q8_0_f16out",
        embed_table, tableBytes, token_ids, output, sizeof(uint16_t), hidden_size, seq_len);
}

extern "C" int dotllm_metal_embedding_q6_k_f16out(
    dotllm_metal_context* ctx,
    const uint8_t*  embed_table,
    const int32_t*  token_ids,
    uint16_t*       output,
    int32_t         vocab_size,
    int32_t         hidden_size,
    int32_t         seq_len)
{
    const int32_t Q6_K_SUPER_BLOCK_SIZE = 256;
    const int32_t Q6_K_BLOCK_BYTES      = 210;
    int32_t blocks_per_row = hidden_size / Q6_K_SUPER_BLOCK_SIZE;
    NSUInteger tableBytes = (NSUInteger)(vocab_size * blocks_per_row) * Q6_K_BLOCK_BYTES;
    return run_embedding_kernel(ctx, "embedding_lookup_q6_k_f16out",
        embed_table, tableBytes, token_ids, output, sizeof(uint16_t), hidden_size, seq_len);
}

// Attention using persistent K/V MTLBuffers — only Q and output need temp alloc.
static int run_attention_f16_with_kvcache(
    dotllm_metal_context* ctx,
    dotllm_metal_kvcache* cache,
    const uint16_t* q,
    uint16_t*       output,
    int32_t         layer,
    int32_t         seq_q,
    int32_t         num_heads,
    int32_t         num_kv_heads,
    int32_t         head_dim,
    int32_t         position_offset,
    int32_t         sliding_window)
{
    @autoreleasepool {
        if (!ctx || !cache || !q || !output) return -10;
        if (layer < 0 || layer >= cache->num_layers) return -10;
        if (seq_q <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) return -10;
        if (num_heads % num_kv_heads != 0) return -10;

        int32_t seq_kv = cache->current_length;
        if (seq_kv <= 0) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "attention.metal", "attention_f16");
        if (!pipeline) return -3;

        NSUInteger qBytes   = (NSUInteger)(seq_q * num_heads * head_dim) * sizeof(uint16_t);
        NSUInteger outBytes = qBytes;

        NSUInteger offQ = 0, offOut = 0;
        bool needCopy = false;
        id<MTLBuffer> bufQ   = bind_input (ctx, q,      qBytes,   &offQ);
        id<MTLBuffer> bufOut = bind_output(ctx, output, outBytes, &offOut, &needCopy);

        // K/V: persistent cache buffers — zero copies on Apple Silicon
        id<MTLBuffer> bufK = [cache->key_buffers   objectAtIndex:(NSUInteger)layer];
        id<MTLBuffer> bufV = [cache->value_buffers objectAtIndex:(NSUInteger)layer];

        if (!bufQ || !bufOut || !bufK || !bufV) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufQ      offset:offQ   atIndex:0];
        [enc setBuffer:bufK      offset:0      atIndex:1];
        [enc setBuffer:bufV      offset:0      atIndex:2];
        [enc setBuffer:bufOut    offset:offOut atIndex:3];
        [enc setBytes:&seq_q           length:sizeof(int32_t) atIndex:4];
        [enc setBytes:&seq_kv          length:sizeof(int32_t) atIndex:5];
        [enc setBytes:&num_heads       length:sizeof(int32_t) atIndex:6];
        [enc setBytes:&num_kv_heads    length:sizeof(int32_t) atIndex:7];
        [enc setBytes:&head_dim        length:sizeof(int32_t) atIndex:8];
        [enc setBytes:&position_offset length:sizeof(int32_t) atIndex:9];
        [enc setBytes:&sliding_window  length:sizeof(int32_t) atIndex:10];

        const int TILE_KV = 256;
        NSUInteger smemBytes = (NSUInteger)(2 * head_dim + TILE_KV + 8) * sizeof(float);
        [enc setThreadgroupMemoryLength:smemBytes atIndex:0];

        // Attention: each threadgroup processes head_dim work for its (token, head)
        // pair, iterating over the KV cache in TILE_KV chunks. Cap 256 keeps smem
        // small and matches the TILE_KV=256 tile size.
        NSUInteger tgw     = pick_tgw(pipeline, (NSUInteger)head_dim, 256);
        NSUInteger nGroups = (NSUInteger)(seq_q * num_heads);
        [enc dispatchThreadgroups:MTLSizeMake(nGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopy && should_copy_back(&ks)) memcpy(output, bufOut.contents, outBytes);
        return 0;
    }
}

extern "C" int dotllm_metal_attention_f16_kvcache(
    dotllm_metal_context* ctx,
    dotllm_metal_kvcache* cache,
    const uint16_t* q,
    uint16_t*       output,
    int32_t         layer,
    int32_t         seq_q,
    int32_t         num_heads,
    int32_t         num_kv_heads,
    int32_t         head_dim,
    int32_t         position_offset,
    int32_t         sliding_window)
{
    return run_attention_f16_with_kvcache(ctx, cache, q, output, layer,
        seq_q, num_heads, num_kv_heads, head_dim, position_offset, sliding_window);
}

// ── Attention ────────────────────────────────────────────────────────────────
//
// Shared dispatch helper: both f16 and f32 variants use the same layout,
// the only difference is element size and shader name.
//
static int run_attention_kernel(
    dotllm_metal_context* ctx,
    const char*  functionName,
    const void*  q,
    const void*  k,
    const void*  v,
    void*        output,
    size_t       elemSize,   // sizeof(float) or sizeof(uint16_t)
    int32_t      seq_q,
    int32_t      seq_kv,
    int32_t      num_heads,
    int32_t      num_kv_heads,
    int32_t      head_dim,
    int32_t      position_offset,
    int32_t      sliding_window)
{
    @autoreleasepool {
        if (!ctx || !q || !k || !v || !output) return -10;
        if (seq_q <= 0 || seq_kv <= 0 || num_heads <= 0 || num_kv_heads <= 0
            || head_dim <= 0 || num_heads % num_kv_heads != 0) return -10;

        // 1:1 with the CUDA sources: attention_f16 lives in attention.metal
        // (port of attention.cu), attention_f32 in attention_f32.metal.
        NSString* shaderFile = (elemSize == sizeof(float))
                             ? @"attention_f32.metal"
                             : @"attention.metal";
        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, shaderFile.UTF8String, functionName);
        if (!pipeline) return -3;

        NSUInteger qBytes      = (NSUInteger)(seq_q  * num_heads    * head_dim) * elemSize;
        NSUInteger kvBytes     = (NSUInteger)(seq_kv * num_kv_heads * head_dim) * elemSize;
        NSUInteger outputBytes = qBytes;

        NSUInteger offQ = 0, offK = 0, offV = 0, offOut = 0;
        bool needCopy = false;
        id<MTLBuffer> bufQ   = bind_input (ctx, q,      qBytes,      &offQ);
        id<MTLBuffer> bufK   = bind_input (ctx, k,      kvBytes,     &offK);
        id<MTLBuffer> bufV   = bind_input (ctx, v,      kvBytes,     &offV);
        id<MTLBuffer> bufOut = bind_output(ctx, output, outputBytes, &offOut, &needCopy);

        if (!bufQ || !bufK || !bufV || !bufOut) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufQ      offset:offQ   atIndex:0];
        [enc setBuffer:bufK      offset:offK   atIndex:1];
        [enc setBuffer:bufV      offset:offV   atIndex:2];
        [enc setBuffer:bufOut    offset:offOut atIndex:3];
        [enc setBytes:&seq_q           length:sizeof(int32_t) atIndex:4];
        [enc setBytes:&seq_kv          length:sizeof(int32_t) atIndex:5];
        [enc setBytes:&num_heads       length:sizeof(int32_t) atIndex:6];
        [enc setBytes:&num_kv_heads    length:sizeof(int32_t) atIndex:7];
        [enc setBytes:&head_dim        length:sizeof(int32_t) atIndex:8];
        [enc setBytes:&position_offset length:sizeof(int32_t) atIndex:9];
        [enc setBytes:&sliding_window  length:sizeof(int32_t) atIndex:10];

        // Dynamic threadgroup memory:
        //   q_shared(head_dim) + score_tile(256) + out_accum(head_dim) + warp_scratch(8)
        const int TILE_KV = 256;
        NSUInteger smemBytes = (NSUInteger)(2 * head_dim + TILE_KV + 8) * sizeof(float);
        [enc setThreadgroupMemoryLength:smemBytes atIndex:0];

        // Attention: each threadgroup processes head_dim work for its (token, head)
        // pair, iterating over the KV cache in TILE_KV chunks. Cap 256 keeps smem
        // small and matches the TILE_KV=256 tile size.
        NSUInteger tgw     = pick_tgw(pipeline, (NSUInteger)head_dim, 256);
        NSUInteger nGroups = (NSUInteger)(seq_q * num_heads);
        [enc dispatchThreadgroups:MTLSizeMake(nGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopy && should_copy_back(&ks)) memcpy(output, bufOut.contents, outputBytes);
        return 0;
    }
}

extern "C" int dotllm_metal_attention_f16(
    dotllm_metal_context* ctx,
    const uint16_t* q,
    const uint16_t* k,
    const uint16_t* v,
    uint16_t*       output,
    int32_t         seq_q,
    int32_t         seq_kv,
    int32_t         num_heads,
    int32_t         num_kv_heads,
    int32_t         head_dim,
    int32_t         position_offset,
    int32_t         sliding_window)
{
    return run_attention_kernel(ctx, "attention_f16",
        q, k, v, output, sizeof(uint16_t),
        seq_q, seq_kv, num_heads, num_kv_heads, head_dim,
        position_offset, sliding_window);
}

extern "C" int dotllm_metal_attention_f32(
    dotllm_metal_context* ctx,
    const float* q,
    const float* k,
    const float* v,
    float*       output,
    int32_t      seq_q,
    int32_t      seq_kv,
    int32_t      num_heads,
    int32_t      num_kv_heads,
    int32_t      head_dim,
    int32_t      position_offset,
    int32_t      sliding_window)
{
    return run_attention_kernel(ctx, "attention_f32",
        q, k, v, output, sizeof(float),
        seq_q, seq_kv, num_heads, num_kv_heads, head_dim,
        position_offset, sliding_window);
}

// ── KV-cache quantization ────────────────────────────────────────────────────
//
// Both quant kernels share the same buffer layout:
//   buffer(0) = src  (FP16 input, read-only, passed as uint16_t*)
//   buffer(1) = dst  (quantized output bytes)
//   buffer(2) = total_blocks (constant int)
//
// Dispatch: one thread per block → dispatchThreads(total_blocks, 1, 1).
//
static int run_quant_kv_kernel(
    dotllm_metal_context* ctx,
    const char*      functionName,
    const uint16_t*  src,
    NSUInteger       src_bytes,
    uint8_t*         dst,
    NSUInteger       dst_bytes,
    int32_t          total_blocks)
{
    @autoreleasepool {
        if (!ctx || !src || !dst || total_blocks <= 0) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "quant_kv.metal", functionName);
        if (!pipeline) return -3;

        NSUInteger offSrc = 0, offDst = 0;
        bool needCopy = false;
        id<MTLBuffer> bufSrc = bind_input (ctx, src, src_bytes, &offSrc);
        id<MTLBuffer> bufDst = bind_output(ctx, dst, dst_bytes, &offDst, &needCopy);
        if (!bufSrc || !bufDst) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufSrc   offset:offSrc atIndex:0];
        [enc setBuffer:bufDst   offset:offDst atIndex:1];
        [enc setBytes:&total_blocks length:sizeof(int32_t) atIndex:2];

        // One thread per quantization block.
        NSUInteger tgw = pick_tgw(pipeline, (NSUInteger)total_blocks, 256);

        [enc dispatchThreads:MTLSizeMake(total_blocks, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopy && should_copy_back(&ks)) memcpy(dst, bufDst.contents, dst_bytes);
        return 0;
    }
}

extern "C" int dotllm_metal_quant_f16_to_q8_0(
    dotllm_metal_context* ctx,
    const uint16_t* src,
    uint8_t*        dst,
    int32_t         total_blocks)
{
    const int32_t BLOCK_SIZE  = 32;
    const int32_t BLOCK_BYTES = 34;
    NSUInteger src_bytes = (NSUInteger)total_blocks * BLOCK_SIZE * sizeof(uint16_t);
    NSUInteger dst_bytes = (NSUInteger)total_blocks * BLOCK_BYTES;
    return run_quant_kv_kernel(ctx, "quant_f16_to_q8_0",
        src, src_bytes, dst, dst_bytes, total_blocks);
}

extern "C" int dotllm_metal_quant_f16_to_q4_0(
    dotllm_metal_context* ctx,
    const uint16_t* src,
    uint8_t*        dst,
    int32_t         total_blocks)
{
    const int32_t BLOCK_SIZE  = 32;
    const int32_t BLOCK_BYTES = 18;
    NSUInteger src_bytes = (NSUInteger)total_blocks * BLOCK_SIZE * sizeof(uint16_t);
    NSUInteger dst_bytes = (NSUInteger)total_blocks * BLOCK_BYTES;
    return run_quant_kv_kernel(ctx, "quant_f16_to_q4_0",
        src, src_bytes, dst, dst_bytes, total_blocks);
}

// ── Quantized GEMV ───────────────────────────────────────────────────────────

extern "C" int dotllm_metal_quantized_gemv_q8_0_f32in(
    dotllm_metal_context* ctx,
    const uint8_t* weight,
    const float*   x,
    float*         y,
    int32_t        n,
    int32_t        k)
{
    @autoreleasepool {
        if (!ctx || !weight || !x || !y || n <= 0 || k <= 0 || k % 32 != 0) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "quantized_gemv_f32in.metal",
                                       "quantized_gemv_q8_0_f32in");
        if (!pipeline) return -3;

        // weight: n rows × (k/32) blocks × 34 bytes
        int32_t bpr = k / 32;
        NSUInteger weightBytes = (NSUInteger)n * bpr * 34;
        NSUInteger xBytes      = (NSUInteger)k * sizeof(float);
        NSUInteger yBytes      = (NSUInteger)n * sizeof(float);

        NSUInteger offW = 0, offX = 0, offY = 0;
        bool needCopy = false;
        id<MTLBuffer> bufW = bind_input (ctx, weight, weightBytes, &offW);
        id<MTLBuffer> bufX = bind_input (ctx, x,      xBytes,      &offX);
        id<MTLBuffer> bufY = bind_output(ctx, y,      yBytes,      &offY, &needCopy);
        if (!bufW || !bufX || !bufY) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufW offset:offW atIndex:0];
        [enc setBuffer:bufX offset:offX atIndex:1];
        [enc setBuffer:bufY offset:offY atIndex:2];
        [enc setBytes:&n length:sizeof(int32_t) atIndex:3];
        [enc setBytes:&k length:sizeof(int32_t) atIndex:4];

        // One threadgroup per output row; each thread strides over Q8_0 blocks
        // (bpr = k/32). pick_tgw shrinks tgw when bpr < 256 (no idle threads).
        NSUInteger tgw = pick_tgw(pipeline, (NSUInteger)bpr, 256);
        [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopy && should_copy_back(&ks)) memcpy(y, bufY.contents, yBytes);
        return 0;
    }
}

// ── Quantized GEMV — FP16 I/O ────────────────────────────────────────────────
//
// Common dispatch helper for all 5 FP16-in/out GEMV kernels.
// weight: raw quantized bytes; x/y: half vectors (uint16_t).
// One threadgroup per output row, 256 threads per group.

static int run_gemv_f16_kernel(
    dotllm_metal_context* ctx,
    const char*    functionName,
    const uint8_t* weight,
    NSUInteger     weightBytes,
    const uint16_t* x,
    uint16_t*       y,
    int32_t         n,
    int32_t         k)
{
    @autoreleasepool {
        if (!ctx || !weight || !x || !y || n <= 0 || k <= 0) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "quantized_gemv.metal", functionName);
        if (!pipeline) return -3;

        NSUInteger xBytes = (NSUInteger)k * sizeof(uint16_t);
        NSUInteger yBytes = (NSUInteger)n * sizeof(uint16_t);

        NSUInteger offW = 0, offX = 0, offY = 0;
        bool needCopyY = false;
        id<MTLBuffer> bufW = bind_input (ctx, weight, weightBytes, &offW);
        id<MTLBuffer> bufX = bind_input (ctx, x,      xBytes,      &offX);
        id<MTLBuffer> bufY = bind_output(ctx, y,      yBytes,      &offY, &needCopyY);
        if (!bufW || !bufX || !bufY) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufW offset:offW atIndex:0];
        [enc setBuffer:bufX offset:offX atIndex:1];
        [enc setBuffer:bufY offset:offY atIndex:2];
        // Small int constants → setBytes (no MTLBuffer allocation).
        [enc setBytes:&n length:sizeof(int32_t) atIndex:3];
        [enc setBytes:&k length:sizeof(int32_t) atIndex:4];

        // Generic FP16 GEMV: 5 quant formats supported. bpr is k/32 for
        // Q8_0/Q4_0/Q5_0 and k/256 for K-quants. We use k/32 as an upper bound
        // (no underutilization for K-quants when k is large; harmless overshoot
        // for tiny k since pick_tgw rounds to warp).
        NSUInteger tgw = pick_tgw(pipeline, (NSUInteger)k / 32, 256);
        [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopyY && should_copy_back(&ks)) memcpy(y, bufY.contents, yBytes);
        return 0;
    }
}

extern "C" int dotllm_metal_quantized_gemv_q8_0(
    dotllm_metal_context* ctx,
    const uint8_t*  weight,
    const uint16_t* x,
    uint16_t*       y,
    int32_t         n,
    int32_t         k)
{
    if (k % 32 != 0) return -10;
    int32_t bpr = k / 32;
    NSUInteger weightBytes = (NSUInteger)n * bpr * 34;
    return run_gemv_f16_kernel(ctx, "quantized_gemv_q8_0",
        weight, weightBytes, x, y, n, k);
}

extern "C" int dotllm_metal_quantized_gemv_q5_0(
    dotllm_metal_context* ctx,
    const uint8_t*  weight,
    const uint16_t* x,
    uint16_t*       y,
    int32_t         n,
    int32_t         k)
{
    if (k % 32 != 0) return -10;
    int32_t bpr = k / 32;
    NSUInteger weightBytes = (NSUInteger)n * bpr * 22;
    return run_gemv_f16_kernel(ctx, "quantized_gemv_q5_0",
        weight, weightBytes, x, y, n, k);
}

extern "C" int dotllm_metal_quantized_gemv_q4_k(
    dotllm_metal_context* ctx,
    const uint8_t*  weight,
    const uint16_t* x,
    uint16_t*       y,
    int32_t         n,
    int32_t         k)
{
    if (k % 256 != 0) return -10;
    int32_t sbpr = k / 256;
    NSUInteger weightBytes = (NSUInteger)n * sbpr * 144;
    return run_gemv_f16_kernel(ctx, "quantized_gemv_q4_k",
        weight, weightBytes, x, y, n, k);
}

extern "C" int dotllm_metal_quantized_gemv_q5_k(
    dotllm_metal_context* ctx,
    const uint8_t*  weight,
    const uint16_t* x,
    uint16_t*       y,
    int32_t         n,
    int32_t         k)
{
    if (k % 256 != 0) return -10;
    int32_t sbpr = k / 256;
    NSUInteger weightBytes = (NSUInteger)n * sbpr * 176;
    return run_gemv_f16_kernel(ctx, "quantized_gemv_q5_k",
        weight, weightBytes, x, y, n, k);
}

extern "C" int dotllm_metal_quantized_gemv_q6_k(
    dotllm_metal_context* ctx,
    const uint8_t*  weight,
    const uint16_t* x,
    uint16_t*       y,
    int32_t         n,
    int32_t         k)
{
    if (k % 256 != 0) return -10;
    int32_t sbpr = k / 256;
    NSUInteger weightBytes = (NSUInteger)n * sbpr * 210;
    return run_gemv_f16_kernel(ctx, "quantized_gemv_q6_k",
        weight, weightBytes, x, y, n, k);
}

// ── Dequantization ───────────────────────────────────────────────────────────
//
// All dequant kernels share the same buffer layout:
//   buffer(0) = src  (raw quantized bytes, read-only)
//   buffer(1) = dst  (output half values)
//   buffer(2) = total_blocks or total_superblocks (constant int)
//
// Two dispatch strategies:
//   "warp-per-block"  (Q8_0 / Q4_0 / Q5_0): tgw=256, 8 warps/group → nGroups = ceil(total/8)
//   "thread-per-elem" (Q4_K / Q5_K / Q6_K): tgw=256,  1 group/superblock → nGroups = total
//
static int run_dequant_kernel(
    dotllm_metal_context* ctx,
    const char*    functionName,
    const uint8_t* src,
    NSUInteger     src_bytes,
    uint16_t*      dst,
    NSUInteger     dst_bytes,
    int32_t        total,       // total_blocks or total_superblocks
    NSUInteger     n_groups,
    NSUInteger     tgw)
{
    @autoreleasepool {
        if (!ctx || !src || !dst || total <= 0) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "dequant.metal", functionName);
        if (!pipeline) return -3;

        NSUInteger offSrc = 0, offDst = 0;
        bool needCopy = false;
        id<MTLBuffer> bufSrc = bind_input (ctx, src, src_bytes, &offSrc);
        id<MTLBuffer> bufDst = bind_output(ctx, dst, dst_bytes, &offDst, &needCopy);
        if (!bufSrc || !bufDst) return -7;

        dotllm_kernel_scope ks = begin_kernel(ctx);
        id<MTLCommandBuffer>         cmd = ks.cmd;
        id<MTLComputeCommandEncoder> enc = ks.enc;
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufSrc offset:offSrc atIndex:0];
        [enc setBuffer:bufDst offset:offDst atIndex:1];
        [enc setBytes:&total length:sizeof(int32_t) atIndex:2];

        [enc dispatchThreadgroups:MTLSizeMake(n_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        int rc = finish_kernel(&ks);
        if (rc) return rc;

        if (needCopy && should_copy_back(&ks)) memcpy(dst, bufDst.contents, dst_bytes);
        return 0;
    }
}

// threads-per-group for all dequant kernels
static const NSUInteger kDequantTgw = 256;
// warps-per-group for the warp-per-block kernels (256 threads / 32 per block)
static const NSUInteger kDequantWarpsPerGroup = 8;

extern "C" int dotllm_metal_dequant_q8_0_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_blocks)
{
    const int32_t BLOCK_BYTES = 34, BLOCK_SIZE = 32;
    NSUInteger src_bytes = (NSUInteger)total_blocks * BLOCK_BYTES;
    NSUInteger dst_bytes = (NSUInteger)total_blocks * BLOCK_SIZE * sizeof(uint16_t);
    NSUInteger n_groups  = ((NSUInteger)total_blocks + kDequantWarpsPerGroup - 1) / kDequantWarpsPerGroup;
    return run_dequant_kernel(ctx, "dequant_q8_0_f16",
        src, src_bytes, dst, dst_bytes, total_blocks, n_groups, kDequantTgw);
}

extern "C" int dotllm_metal_dequant_q4_0_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_blocks)
{
    const int32_t BLOCK_BYTES = 18, BLOCK_SIZE = 32;
    NSUInteger src_bytes = (NSUInteger)total_blocks * BLOCK_BYTES;
    NSUInteger dst_bytes = (NSUInteger)total_blocks * BLOCK_SIZE * sizeof(uint16_t);
    NSUInteger n_groups  = ((NSUInteger)total_blocks + kDequantWarpsPerGroup - 1) / kDequantWarpsPerGroup;
    return run_dequant_kernel(ctx, "dequant_q4_0_f16",
        src, src_bytes, dst, dst_bytes, total_blocks, n_groups, kDequantTgw);
}

extern "C" int dotllm_metal_dequant_q5_0_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_blocks)
{
    const int32_t BLOCK_BYTES = 22, BLOCK_SIZE = 32;
    NSUInteger src_bytes = (NSUInteger)total_blocks * BLOCK_BYTES;
    NSUInteger dst_bytes = (NSUInteger)total_blocks * BLOCK_SIZE * sizeof(uint16_t);
    NSUInteger n_groups  = ((NSUInteger)total_blocks + kDequantWarpsPerGroup - 1) / kDequantWarpsPerGroup;
    return run_dequant_kernel(ctx, "dequant_q5_0_f16",
        src, src_bytes, dst, dst_bytes, total_blocks, n_groups, kDequantTgw);
}

extern "C" int dotllm_metal_dequant_q4_k_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_superblocks)
{
    const int32_t BLOCK_BYTES = 144, SUPER_BLOCK_SIZE = 256;
    NSUInteger src_bytes = (NSUInteger)total_superblocks * BLOCK_BYTES;
    NSUInteger dst_bytes = (NSUInteger)total_superblocks * SUPER_BLOCK_SIZE * sizeof(uint16_t);
    NSUInteger n_groups  = (NSUInteger)total_superblocks; // one group per superblock
    return run_dequant_kernel(ctx, "dequant_q4_k_f16",
        src, src_bytes, dst, dst_bytes, total_superblocks, n_groups, kDequantTgw);
}

extern "C" int dotllm_metal_dequant_q5_k_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_superblocks)
{
    const int32_t BLOCK_BYTES = 176, SUPER_BLOCK_SIZE = 256;
    NSUInteger src_bytes = (NSUInteger)total_superblocks * BLOCK_BYTES;
    NSUInteger dst_bytes = (NSUInteger)total_superblocks * SUPER_BLOCK_SIZE * sizeof(uint16_t);
    NSUInteger n_groups  = (NSUInteger)total_superblocks;
    return run_dequant_kernel(ctx, "dequant_q5_k_f16",
        src, src_bytes, dst, dst_bytes, total_superblocks, n_groups, kDequantTgw);
}

extern "C" int dotllm_metal_dequant_q6_k_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_superblocks)
{
    const int32_t BLOCK_BYTES = 210, SUPER_BLOCK_SIZE = 256;
    NSUInteger src_bytes = (NSUInteger)total_superblocks * BLOCK_BYTES;
    NSUInteger dst_bytes = (NSUInteger)total_superblocks * SUPER_BLOCK_SIZE * sizeof(uint16_t);
    NSUInteger n_groups  = (NSUInteger)total_superblocks;
    return run_dequant_kernel(ctx, "dequant_q6_k_f16",
        src, src_bytes, dst, dst_bytes, total_superblocks, n_groups, kDequantTgw);
}

extern "C" int dotllm_metal_convert_f16_to_f32(
    dotllm_metal_context* ctx,
    const uint16_t* src,
    float*          dst,
    int32_t         n)
{
    return run_convert_kernel(ctx, "convert_f16_to_f32", src, dst, n, sizeof(uint16_t), sizeof(float));
}

extern "C" int dotllm_metal_convert_f32_to_f16(
    dotllm_metal_context* ctx,
    const float* src,
    uint16_t*    dst,
    int32_t      n)
{
    return run_convert_kernel(ctx, "convert_f32_to_f16", src, dst, n, sizeof(float), sizeof(uint16_t));
}

// ── GEMM via Metal Performance Shaders ───────────────────────────────────────
//
// C = alpha * op(A) * op(B) + beta * C
//
// op(X) = X        if transpose_x = 0
// op(X) = X^T      if transpose_x = 1
//
// Matrix descriptors describe the *storage layout* (rows × columns × rowBytes).
// The transpose flag tells MPS to treat the matrix as transposed during
// the multiplication — the storage is unchanged.
//
// Standard LLM projection: Y[seqLen, N] = X[seqLen, K] · W[N, K]^T
//   → m = seqLen, k = K (inputDim), n = N (outputDim)
//   → A = X (no transpose), B = W (transpose B)
//   → A descriptor: rows=m, columns=k, rowBytes=k*sizeof(elem)
//   → B descriptor: rows=n, columns=k, rowBytes=k*sizeof(elem)
//   → C descriptor: rows=m, columns=n, rowBytes=n*sizeof(elem)
//
// Element size: 2 bytes (half) or 4 bytes (float).

static int run_gemm(
    dotllm_metal_context* ctx,
    MPSDataType  dataType,
    size_t       elemSize,
    const void*  a,
    const void*  b,
    void*        c,
    int32_t      m,
    int32_t      n,
    int32_t      k,
    int32_t      transpose_a,
    int32_t      transpose_b,
    float        alpha,
    float        beta)
{
    @autoreleasepool {
        if (!ctx || !a || !b || !c) return -10;
        if (m <= 0 || n <= 0 || k <= 0) return -12;

        // Storage rows/columns for each matrix.
        // op(A) is m×k → if transposed, A is stored as k×m.
        NSUInteger aRows = transpose_a ? (NSUInteger)k : (NSUInteger)m;
        NSUInteger aCols = transpose_a ? (NSUInteger)m : (NSUInteger)k;
        // op(B) is k×n → if transposed, B is stored as n×k.
        NSUInteger bRows = transpose_b ? (NSUInteger)n : (NSUInteger)k;
        NSUInteger bCols = transpose_b ? (NSUInteger)k : (NSUInteger)n;
        NSUInteger cRows = (NSUInteger)m;
        NSUInteger cCols = (NSUInteger)n;

        NSUInteger aBytes = aRows * aCols * elemSize;
        NSUInteger bBytes = bRows * bCols * elemSize;
        NSUInteger cBytes = cRows * cCols * elemSize;

        // Zero-copy bind. For C: if beta != 0 the kernel reads C, so we need
        // copy-in on miss — handled inline.
        NSUInteger offA = 0, offB = 0, offC = 0;
        bool needCopyC = false;
        id<MTLBuffer> bufA = bind_input(ctx, a, aBytes, &offA);
        id<MTLBuffer> bufB = bind_input(ctx, b, bBytes, &offB);
        id<MTLBuffer> bufC;
        if (beta != 0.0f) {
            // C is read AND written. Reuse on hit; copy-in fallback on miss.
            NSUInteger offHit = 0;
            bufC = lookup_shared_buffer(ctx, c, &offHit);
            if (bufC) {
                offC = offHit;
            } else {
                bufC = [ctx->device newBufferWithBytes:c length:cBytes
                            options:MTLResourceStorageModeShared];
                needCopyC = true;
            }
        } else {
            bufC = bind_output(ctx, c, cBytes, &offC, &needCopyC);
        }
        if (!bufA || !bufB || !bufC) return -7;

        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:aRows
                            columns:aCols
                           rowBytes:aCols * elemSize
                           dataType:dataType];
        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:bRows
                            columns:bCols
                           rowBytes:bCols * elemSize
                           dataType:dataType];
        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:cRows
                            columns:cCols
                           rowBytes:cCols * elemSize
                           dataType:dataType];

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA offset:offA descriptor:descA];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB offset:offB descriptor:descB];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC offset:offC descriptor:descC];

        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:ctx->device
             transposeLeft:(transpose_a != 0)
            transposeRight:(transpose_b != 0)
                resultRows:(NSUInteger)m
             resultColumns:(NSUInteger)n
           interiorColumns:(NSUInteger)k
                     alpha:(double)alpha
                      beta:(double)beta];
        if (!mm) return -3;

        // MPS opens its OWN encoder on the command buffer, so if we're inside
        // a batched forward we must close the active compute encoder first,
        // let MPS do its thing, then reopen a fresh compute encoder for the
        // next kernels.
        bool batched = (ctx->active_enc != nil);
        id<MTLCommandBuffer> cmd;
        if (batched) {
            [ctx->active_enc endEncoding];
            ctx->active_enc = nil;
            cmd = ctx->active_cmd;
        } else {
            cmd = [ctx->queue commandBuffer];
            if (!cmd) return -8;
        }

        [mm encodeToCommandBuffer:cmd
                      leftMatrix:matA
                     rightMatrix:matB
                    resultMatrix:matC];

        if (batched) {
            ctx->active_enc = [cmd computeCommandEncoder];
            if (!ctx->active_enc) return -9;
            return 0;
        }

        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error != nil) return -11;
        if (needCopyC) memcpy(c, bufC.contents, cBytes);
        return 0;
    }
}

extern "C" int dotllm_metal_gemm_f16(
    dotllm_metal_context* ctx,
    const uint16_t* a,
    const uint16_t* b,
    uint16_t*       c,
    int32_t         m,
    int32_t         n,
    int32_t         k,
    int32_t         transpose_a,
    int32_t         transpose_b,
    float           alpha,
    float           beta)
{
    return run_gemm(ctx, MPSDataTypeFloat16, sizeof(uint16_t),
                    a, b, c, m, n, k, transpose_a, transpose_b, alpha, beta);
}

extern "C" int dotllm_metal_gemm_f32(
    dotllm_metal_context* ctx,
    const float*    a,
    const float*    b,
    float*          c,
    int32_t         m,
    int32_t         n,
    int32_t         k,
    int32_t         transpose_a,
    int32_t         transpose_b,
    float           alpha,
    float           beta)
{
    return run_gemm(ctx, MPSDataTypeFloat32, sizeof(float),
                    a, b, c, m, n, k, transpose_a, transpose_b, alpha, beta);
}

