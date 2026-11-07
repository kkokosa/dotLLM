#ifndef DOTLLM_ASCEND_H
#define DOTLLM_ASCEND_H

#include <stdint.h>

#define DOTLLM_ASCEND_API __attribute__((visibility("default")))
#ifdef __cplusplus
extern "C" {
#endif

/* ABI 3: Linux, 64-bit. Caller owns AscendCL initialization and current context. */
DOTLLM_ASCEND_API uint32_t dotllm_ascend_abi_version(void);

/* Thread-local UTF-8 diagnostic; valid until the next bridge operation on this thread. */
DOTLLM_ASCEND_API const char* dotllm_ascend_last_error(void);

/*
 * Synchronous FP16 Linear: Y[M,N] = X[M,K] * W[N,K]^T.
 * All buffers are contiguous device allocations on the stream's context.
 * Y must not overlap X or W. The caller validates allocation sizes.
 * Returns 0 on success, native status on failure, or a negative bridge error.
 * No C++ exception escapes. A failed device execution invalidates the calling session.
 */
DOTLLM_ASCEND_API int32_t dotllm_ascend_linear_f16(
    const void* x, const void* weight, void* output,
    int64_t m, int64_t n, int64_t k, void* stream);

/* All operations are synchronous, use FP16 device storage, and forbid output aliasing. */
DOTLLM_ASCEND_API int32_t dotllm_ascend_rms_norm_f16(
    const void* x, const void* weight, void* output,
    int64_t rows, int64_t width, double epsilon, void* stream);
/* right_rows is either rows (residual) or 1 (broadcast bias). */
DOTLLM_ASCEND_API int32_t dotllm_ascend_add_f16(
    const void* left, const void* right, void* output,
    int64_t rows, int64_t width, int64_t right_rows, void* stream);
DOTLLM_ASCEND_API int32_t dotllm_ascend_swiglu_f16(
    const void* gate, const void* up, void* output, int64_t count, void* stream);
/* x/output [rows, heads*head_dim]; cos/sin [rows, rotary_dim].
 * neox=0 pairs (2i,2i+1); neox=1 pairs (i,i+rotary_dim/2).
 * Dimensions past rotary_dim are copied unchanged. */
DOTLLM_ASCEND_API int32_t dotllm_ascend_rope_f16(
    const void* x, const void* cos, const void* sin, void* output,
    int64_t rows, int64_t heads, int64_t head_dim, int64_t rotary_dim,
    int32_t neox, void* stream);
/* Baseline device attention: Q [rows,heads*D], K/V [kv_rows,kv_heads*D].
 * Causal visibility ends at query_offset + row. KV allocation tail is excluded.
 * FP32 scores/softmax/value accumulation; output is FP16.
 * workspace is internal and scoped to this synchronous call. */
DOTLLM_ASCEND_API int32_t dotllm_ascend_attention_f16(
    const void* query, const void* keys, const void* values, void* output,
    int64_t rows, int64_t kv_rows, int64_t heads, int64_t kv_heads,
    int64_t head_dim, int64_t query_offset, void* stream);

/* FP32 intermediate path. Matrix/norm weights are FP32; RoPE tables remain FP16.
 * linear uses KEEP_DTYPE FP32 computation.
 * kv_f16 chooses contiguous cache FP16 or uncached FP32 K/V. */
DOTLLM_ASCEND_API int32_t dotllm_ascend_cast(const void*, void*, int64_t, int32_t, void*);
DOTLLM_ASCEND_API int32_t dotllm_ascend_linear_f32(const void*, const void*, void*, int64_t, int64_t, int64_t, void*);
DOTLLM_ASCEND_API int32_t dotllm_ascend_rms_norm_f32(const void*, const void*, void*, int64_t, int64_t, double, void*);
DOTLLM_ASCEND_API int32_t dotllm_ascend_add_f32(const void*, const void*, void*, int64_t, int64_t, int64_t, int32_t, void*);
DOTLLM_ASCEND_API int32_t dotllm_ascend_swiglu_f32(const void*, const void*, void*, int64_t, void*);
DOTLLM_ASCEND_API int32_t dotllm_ascend_rope_f32(const void*, const void*, const void*, void*, int64_t, int64_t, int64_t, int64_t, int32_t, void*);
DOTLLM_ASCEND_API int32_t dotllm_ascend_attention_f32(const void*, const void*, const void*, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int32_t, void*);

#ifdef __cplusplus
}
#endif
#endif
