#pragma once
#include <stdint.h>
#include <stddef.h>
#include "dotllm_core.h"

#ifdef __cplusplus
extern "C" {
#endif


/// Creates a KV-cache for the given model geometry.
/// Returns NULL on failure (OOM or invalid arguments).
dotllm_metal_kvcache* dotllm_metal_kvcache_create(
    dotllm_metal_context* ctx,
    int32_t num_layers,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t max_seq_len);

/// Destroys the KV-cache and releases all MTLBuffer resources.
void dotllm_metal_kvcache_destroy(dotllm_metal_kvcache* cache);

/// Returns the CPU-writable contents pointer for layer's K buffer (FP16 rows).
/// Layout: [max_seq_len, num_kv_heads * head_dim], row-major, FP16.
void* dotllm_metal_kvcache_key_ptr(dotllm_metal_kvcache* cache, int32_t layer);

/// Returns the CPU-writable contents pointer for layer's V buffer (FP16 rows).
void* dotllm_metal_kvcache_value_ptr(dotllm_metal_kvcache* cache, int32_t layer);

/// Returns the current number of valid cached positions.
int32_t dotllm_metal_kvcache_current_length(dotllm_metal_kvcache* cache);

/// Sets the current valid length (used for rollback and prefix-cache reuse).
void dotllm_metal_kvcache_set_current_length(dotllm_metal_kvcache* cache, int32_t length);

/// FP16 attention using persistent K/V MTLBuffers from the given cache layer.
/// K/V data must have been written to the cache (via the key/value ptr) before
/// calling this function. seq_q is usually 1 during decode.
/// position_offset is the position of the first query token (= cache length before update).
int dotllm_metal_attention_f16_kvcache(
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
    int32_t         sliding_window);

/// Creates a Metal context (device + command queue + pipeline cache).
/// Returns NULL on failure.
dotllm_metal_context* dotllm_metal_create_context(void);

/// Destroys the context and releases all Metal resources.
void dotllm_metal_destroy_context(dotllm_metal_context* ctx);

/// Allocates `bytes` of MTLResourceStorageModeShared memory and returns the
/// `.contents` pointer. The backing MTLBuffer is retained by the context until
/// dotllm_metal_free_shared is called or the context is destroyed.
///
/// On Apple Silicon, the returned pointer is regular host RAM that the GPU can
/// also read/write directly — the same bytes are visible to both sides without
/// any copy or synchronisation.
///
/// Returns NULL on OOM or invalid arguments.
void* dotllm_metal_alloc_shared(dotllm_metal_context* ctx, size_t bytes);

/// Releases a buffer previously returned by dotllm_metal_alloc_shared.
/// Safe to call with a NULL pointer.
void dotllm_metal_free_shared(dotllm_metal_context* ctx, void* ptr);

/// Registers a caller-owned memory region (e.g. a memory-mapped GGUF file) as
/// a zero-copy MTLBuffer backed by MTLResourceStorageModeShared. Kernels can
/// then recover the MTLBuffer (and the offset within it) for any pointer that
/// falls inside [ptr, ptr+bytes), eliminating CPU→GPU copies.
///
/// Requirements:
///   - `ptr` must be page-aligned (`getpagesize()`, 16 KiB on Apple Silicon).
///   - `bytes` is rounded up to the next page multiple internally.
///   - The caller MUST keep the memory alive (and not unmap it) until
///     dotllm_metal_unregister_buffer is called or the context is destroyed.
///
/// Returns 0 on success; negative on failure (alignment/OOM).
int dotllm_metal_register_buffer(
    dotllm_metal_context* ctx, const void* ptr, size_t bytes);

/// Unregisters a region previously passed to dotllm_metal_register_buffer.
/// Releases the backing MTLBuffer. Safe to call with an unknown pointer.
void dotllm_metal_unregister_buffer(dotllm_metal_context* ctx, const void* ptr);

/// Begins a batched forward pass: opens a command buffer + compute encoder.
/// While active, every kernel call encodes its dispatch into this encoder
/// and skips its per-kernel commit/waitUntilCompleted — eliminating the
/// per-kernel driver submission overhead (~100 µs on macOS).
///
/// Pair with dotllm_metal_end_forward. Nesting is not supported.
/// Returns 0 on success, negative on error.
int dotllm_metal_begin_forward(dotllm_metal_context* ctx);

/// Ends a batched forward pass: closes the encoder, commits, and BLOCKS until
/// the GPU finishes. After return, host reads of shared buffers see final data.
int dotllm_metal_end_forward(dotllm_metal_context* ctx);

/// Copies `bytes` from src to dst between GPU-visible buffers. Both pointers
/// must be GPU-resident (alloc_shared or register_buffer); in standalone mode
/// the call falls back to CPU memcpy when not registered.
///
/// In batched mode, the copy is encoded as a blit in the active command buffer
/// — subsequent kernels in the same forward see the result without any CPU sync.
int dotllm_metal_buffer_copy(
    dotllm_metal_context* ctx, void* dst, const void* src, size_t bytes);

/// Element-wise addition: result[i] = a[i] + b[i]  (all FP32)
/// Port of add_f32.cu::add_f32
int dotllm_metal_add_f32(
    dotllm_metal_context* ctx,
    const float*    a,
    const float*    b,
    float*          result,
    uint32_t        length);

/// Element-wise addition: result[i] = a[i] + b[i]  (all FP16, vectorized half2)
/// Port of add.cu::add_f16
int dotllm_metal_add_f16(
    dotllm_metal_context* ctx,
    const uint16_t* a,
    const uint16_t* b,
    uint16_t*       result,
    uint32_t        length);

/// Mixed-precision addition: result_f32[i] = a_f32[i] + b_f16[i]
/// Used when adding an FP16 projection output into the FP32 residual stream.
/// Port of add_f32.cu::add_f32_f16
int dotllm_metal_add_f32_f16(
    dotllm_metal_context* ctx,
    const float*    a,
    const uint16_t* b,
    float*          result,
    uint32_t        length);

/// Numerically stable softmax, one threadgroup per row. FP16 I/O, FP32 accumulation.
/// input/output layout: [rows, cols] — row-major.
/// Port of softmax.cu::softmax_f16
int dotllm_metal_softmax_f16(
    dotllm_metal_context* ctx,
    const uint16_t* input,
    uint16_t*       output,
    int32_t         rows,
    int32_t         cols);

/// SwiGLU (FP32): result[i] = SiLU(gate[i]) * up[i]
/// Port of swiglu_f32.cu::swiglu_f32
int dotllm_metal_swiglu_f32(
    dotllm_metal_context* ctx,
    const float* gate,
    const float* up,
    float*       result,
    uint32_t     length);

/// SwiGLU (FP16): result[i] = SiLU(gate[i]) * up[i]
/// Vectorized: half2 loads/stores, FP32 computation for sigmoid precision.
/// Port of swiglu.cu::swiglu_f16
int dotllm_metal_swiglu_f16(
    dotllm_metal_context* ctx,
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t*       result,
    uint32_t        length);

/// Rotary Position Embedding (in-place). Direct translation of rope_f32.cu.
/// rope_type: 0 = norm/interleaved (Llama/Mistral), 1 = neox/split (Qwen/Phi).
int dotllm_metal_rope_f32(
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
    int32_t        rope_type);

/// Rotary Position Embedding (in-place, FP16). FP32 accumulation internally.
/// Direct translation of rope_f16.cu. Same parameters as dotllm_metal_rope_f32.
/// rope_type: 0 = norm/interleaved (Llama/Mistral), 1 = neox/split (Qwen/Phi).
int dotllm_metal_rope_f16(
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
    int32_t        rope_type);

/// Bias addition (in-place, FP32 output + FP16 bias): output[t, i] += float(bias[i])
/// Port of bias_add_f32.cu::bias_add_f32
int dotllm_metal_bias_add_f32(
    dotllm_metal_context* ctx,
    float*          output,
    const uint16_t* bias,       // FP16
    uint32_t        dim,
    uint32_t        seq_len);

/// Bias addition (in-place, FP16 output + FP16 bias): output[t, i] += bias[i]
/// Vectorized: half2 packed operations process 2 elements per thread.
/// Port of bias_add.cu::bias_add_f16
int dotllm_metal_bias_add_f16(
    dotllm_metal_context* ctx,
    uint16_t*       output,     // FP16, in-place
    const uint16_t* bias,       // FP16
    uint32_t        dim,
    uint32_t        seq_len);

/// RMS Normalization (FP32): output[t, i] = input[t, i] / rms(input[t]) * weight[i]
/// One threadgroup per token — reduction kernel, NOT element-wise.
int dotllm_metal_rmsnorm_f32(
    dotllm_metal_context* ctx,
    const float* input,
    const float* weight,
    float*       output,
    int32_t      n,
    int32_t      seq_len,
    float        eps);

/// RMS Normalization (FP16): same formula, FP16 I/O with FP32 accumulation.
/// Port of rmsnorm_f16.cu. One threadgroup per token.
int dotllm_metal_rmsnorm_f16(
    dotllm_metal_context* ctx,
    const uint16_t* input,
    const uint16_t* weight,
    uint16_t*       output,
    int32_t         n,
    int32_t         seq_len,
    float           eps);

/// Fused residual-add + RMS normalization (FP16).
/// Pass 1: sum = FP32(residual[i]) + FP32(x[i]); residual[i] = FP16(sum); accumulate sum².
/// Pass 2: output[i] = FP16(FP32(residual[i]) * rms_inv * FP32(weight[i])).
/// One threadgroup per token.
int dotllm_metal_fused_add_rmsnorm_f16(
    dotllm_metal_context* ctx,
    uint16_t*       residual,
    const uint16_t* x,
    const uint16_t* weight,
    uint16_t*       output,
    int32_t         n,
    int32_t         seq_len,
    float           eps);

/// Per-head RMS Normalization (in-place): for each (token, head), normalizes head_dim elements.
/// Used by models with QK-norm (Gemma 2, Cohere). One threadgroup per (token × head).
int dotllm_metal_per_head_rmsnorm_f32(
    dotllm_metal_context* ctx,
    float*       qk,
    const float* weight,
    int32_t      num_heads,
    int32_t      head_dim,
    int32_t      seq_len,
    float        eps);

/// Per-head RMS Normalization (FP16, in-place). Port of per_head_rmsnorm_f16.cu.
/// FP16 I/O with FP32 accumulation. One threadgroup per (token × head).
int dotllm_metal_per_head_rmsnorm_f16(
    dotllm_metal_context* ctx,
    uint16_t*       qk,
    const uint16_t* weight,
    int32_t         num_heads,
    int32_t         head_dim,
    int32_t         seq_len,
    float           eps);

/// RMS Normalization — FP32 residual input, FP32 weight, FP16 output.
/// Used when the residual stream is FP32 but downstream GEMM needs FP16 input.
/// Port of rmsnorm_f32in.cu::rmsnorm_f32in_f16out.
int dotllm_metal_rmsnorm_f32in_f16out(
    dotllm_metal_context* ctx,
    const float*    input,
    const float*    weight,
    uint16_t*       output,
    int32_t         n,
    int32_t         seq_len,
    float           eps);

/// Embedding lookup — FP32 table → FP32 output.
/// output[t] = embed_table[token_ids[t]]  (hidden_size floats copied per token).
int dotllm_metal_embedding_f32_f32out(
    dotllm_metal_context* ctx,
    const float* embed_table,
    const int32_t* token_ids,
    float*         output,
    int32_t        vocab_size,
    int32_t        hidden_size,
    int32_t        seq_len);

/// Embedding lookup — FP16 table → FP32 output (cast on the fly).
int dotllm_metal_embedding_f16_f32out(
    dotllm_metal_context* ctx,
    const uint16_t* embed_table,
    const int32_t*  token_ids,
    float*          output,
    int32_t         vocab_size,
    int32_t         hidden_size,
    int32_t         seq_len);

/// Embedding lookup — Q8_0 quantized table → FP32 output (dequantize on the fly).
/// embed_table layout: blocks of Q8_0_BLOCK_BYTES (2-byte half scale + 32 int8 values).
int dotllm_metal_embedding_q8_0_f32out(
    dotllm_metal_context* ctx,
    const uint8_t* embed_table,
    const int32_t* token_ids,
    float*         output,
    int32_t        vocab_size,
    int32_t        hidden_size,
    int32_t        seq_len);

/// Embedding lookup — FP32 table → FP16 output (cast on the fly).
int dotllm_metal_embedding_f32_f16out(
    dotllm_metal_context* ctx,
    const float*    embed_table,
    const int32_t*  token_ids,
    uint16_t*       output,
    int32_t         vocab_size,
    int32_t         hidden_size,
    int32_t         seq_len);

/// Embedding lookup — FP16 table → FP16 output (straight copy of half rows).
int dotllm_metal_embedding_f16_f16out(
    dotllm_metal_context* ctx,
    const uint16_t* embed_table,
    const int32_t*  token_ids,
    uint16_t*       output,
    int32_t         vocab_size,
    int32_t         hidden_size,
    int32_t         seq_len);

/// Embedding lookup — Q8_0 quantized table → FP16 output (dequantize on the fly).
int dotllm_metal_embedding_q8_0_f16out(
    dotllm_metal_context* ctx,
    const uint8_t*  embed_table,
    const int32_t*  token_ids,
    uint16_t*       output,
    int32_t         vocab_size,
    int32_t         hidden_size,
    int32_t         seq_len);

/// Embedding lookup — Q6_K quantized table → FP16 output (dequantize on the fly).
/// embed_table layout: vocab_size × (hidden_size/256) superblocks × 210 bytes each.
int dotllm_metal_embedding_q6_k_f16out(
    dotllm_metal_context* ctx,
    const uint8_t*  embed_table,
    const int32_t*  token_ids,
    uint16_t*       output,
    int32_t         vocab_size,
    int32_t         hidden_size,
    int32_t         seq_len);

// ── Attention ────────────────────────────────────────────────────────────────

/// Tiled scaled dot-product attention: FP16 Q/K/V/output, FP32 accumulation.
/// Same algorithm and layout as dotllm_metal_attention_f32 — only the dtype differs.
/// Output is written as FP16 (float → half truncation on store).
int dotllm_metal_attention_f16(
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
    int32_t         sliding_window);

/// Tiled scaled dot-product attention with online softmax and GQA support.
/// Q layout: [seq_q,  num_heads    * head_dim] — FP32.
/// K layout: [seq_kv, num_kv_heads * head_dim] — FP32.
/// V layout: [seq_kv, num_kv_heads * head_dim] — FP32.
/// Output:   [seq_q,  num_heads    * head_dim] — FP32.
/// position_offset: 0 for prefill; cached token count for decode.
/// sliding_window:  0 means full causal attention (no window limit).
int dotllm_metal_attention_f32(
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
    int32_t      sliding_window);

// ── KV-cache quantization ────────────────────────────────────────────────────

/// FP16 → Q8_0 quantization (KV-cache eviction).
/// src: total_blocks × 32 half values.
/// dst: total_blocks × 34 bytes (2-byte half scale + 32 int8 values).
int dotllm_metal_quant_f16_to_q8_0(
    dotllm_metal_context* ctx,
    const uint16_t* src,
    uint8_t*        dst,
    int32_t         total_blocks);

/// FP16 → Q4_0 quantization (KV-cache eviction).
/// src: total_blocks × 32 half values.
/// dst: total_blocks × 18 bytes (2-byte half scale + 16 packed nibble bytes).
int dotllm_metal_quant_f16_to_q4_0(
    dotllm_metal_context* ctx,
    const uint16_t* src,
    uint8_t*        dst,
    int32_t         total_blocks);

// ── Quantized GEMV ───────────────────────────────────────────────────────────

/// Quantized GEMV: y[i] = dot(dequant(W_q8_0[i,:]), x)  for i in [0, n).
/// weight layout: n rows × (k/32) blocks × 34 bytes per block.
/// x: float32 input vector, length k. y: float32 output vector, length n.
/// k must be a multiple of 32.
int dotllm_metal_quantized_gemv_q8_0_f32in(
    dotllm_metal_context* ctx,
    const uint8_t* weight,
    const float*   x,
    float*         y,
    int32_t        n,
    int32_t        k);

/// Quantized GEMV (FP16 I/O): y[i] = dot(dequant(W_q8_0[i,:]), x)  for i in [0, n).
/// weight: n × (k/32) blocks × 34 bytes. x/y: FP16 vectors. k must be a multiple of 32.
int dotllm_metal_quantized_gemv_q8_0(
    dotllm_metal_context* ctx,
    const uint8_t*  weight,
    const uint16_t* x,
    uint16_t*       y,
    int32_t         n,
    int32_t         k);

/// Quantized GEMV (FP16 I/O): y[i] = dot(dequant(W_q5_0[i,:]), x)  for i in [0, n).
/// weight: n × (k/32) blocks × 22 bytes. x/y: FP16 vectors. k must be a multiple of 32.
int dotllm_metal_quantized_gemv_q5_0(
    dotllm_metal_context* ctx,
    const uint8_t*  weight,
    const uint16_t* x,
    uint16_t*       y,
    int32_t         n,
    int32_t         k);

/// Quantized GEMV (FP16 I/O): y[i] = dot(dequant(W_q4_k[i,:]), x)  for i in [0, n).
/// weight: n × (k/256) superblocks × 144 bytes. x/y: FP16 vectors. k must be a multiple of 256.
int dotllm_metal_quantized_gemv_q4_k(
    dotllm_metal_context* ctx,
    const uint8_t*  weight,
    const uint16_t* x,
    uint16_t*       y,
    int32_t         n,
    int32_t         k);

/// Quantized GEMV (FP16 I/O): y[i] = dot(dequant(W_q5_k[i,:]), x)  for i in [0, n).
/// weight: n × (k/256) superblocks × 176 bytes. x/y: FP16 vectors. k must be a multiple of 256.
int dotllm_metal_quantized_gemv_q5_k(
    dotllm_metal_context* ctx,
    const uint8_t*  weight,
    const uint16_t* x,
    uint16_t*       y,
    int32_t         n,
    int32_t         k);

/// Quantized GEMV (FP16 I/O): y[i] = dot(dequant(W_q6_k[i,:]), x)  for i in [0, n).
/// weight: n × (k/256) superblocks × 210 bytes. x/y: FP16 vectors. k must be a multiple of 256.
int dotllm_metal_quantized_gemv_q6_k(
    dotllm_metal_context* ctx,
    const uint8_t*  weight,
    const uint16_t* x,
    uint16_t*       y,
    int32_t         n,
    int32_t         k);

// ── Dequantization ───────────────────────────────────────────────────────────

/// Dequantize Q8_0 → FP16.
/// src layout: total_blocks × 34 bytes (2-byte half scale + 32 int8 weights).
/// dst layout: total_blocks × 32 halves.
int dotllm_metal_dequant_q8_0_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_blocks);

/// Dequantize Q4_0 → FP16.
/// src layout: total_blocks × 18 bytes (2-byte half scale + 16 packed nibble bytes).
/// dst layout: total_blocks × 32 halves.
int dotllm_metal_dequant_q4_0_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_blocks);

/// Dequantize Q5_0 → FP16.
/// src layout: total_blocks × 22 bytes (2-byte half scale + 4-byte qh mask + 16 packed nibble bytes).
/// dst layout: total_blocks × 32 halves.
int dotllm_metal_dequant_q5_0_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_blocks);

/// Dequantize Q4_K → FP16.
/// src layout: total_superblocks × 144 bytes.
/// dst layout: total_superblocks × 256 halves.
int dotllm_metal_dequant_q4_k_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_superblocks);

/// Dequantize Q5_K → FP16.
/// src layout: total_superblocks × 176 bytes.
/// dst layout: total_superblocks × 256 halves.
int dotllm_metal_dequant_q5_k_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_superblocks);

/// Dequantize Q6_K → FP16.
/// src layout: total_superblocks × 210 bytes.
/// dst layout: total_superblocks × 256 halves.
int dotllm_metal_dequant_q6_k_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_superblocks);

/// Type conversion: float16 → float32, element-wise.
/// src and dst must each hold n elements (src: n×2 bytes, dst: n×4 bytes).
/// uint16_t* is used for half because C has no standard half type.
int dotllm_metal_convert_f16_to_f32(
    dotllm_metal_context* ctx,
    const uint16_t* src,
    float*          dst,
    int32_t         n);

/// Type conversion: float32 → float16, element-wise.
int dotllm_metal_convert_f32_to_f16(
    dotllm_metal_context* ctx,
    const float* src,
    uint16_t*    dst,
    int32_t      n);

// ── GEMM (Metal Performance Shaders) ─────────────────────────────────────────
//
// C[m, n] = alpha * op(A) * op(B) + beta * C
//   op(X) = X^T if transpose_x != 0, else X.
//
// Storage layouts (row-major, no transpose):
//   A : [m, k] (or [k, m] if transpose_a)
//   B : [k, n] (or [n, k] if transpose_b)
//   C : [m, n]
//
// Standard LLM projection Y = X · W^T with W stored as [N, K]:
//   m = seqLen, n = outputDim, k = inputDim
//   transpose_a = 0, transpose_b = 1, alpha = 1, beta = 0.

/// FP16 GEMM via MPSMatrixMultiplication. Buffers passed as uint16_t* (half bit layout).
int dotllm_metal_gemm_f16(
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
    float           beta);

/// FP32 GEMM via MPSMatrixMultiplication.
int dotllm_metal_gemm_f32(
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
    float           beta);

#ifdef __cplusplus
}
#endif
