/* Test-only ABI fixtures. Never use this library for inference. */
#include <stdint.h>
#ifndef TEST_ABI
#define TEST_ABI 3
#endif
uint32_t dotllm_ascend_abi_version(void) { return TEST_ABI; }
#ifndef TEST_MISSING_SYMBOLS
const char *dotllm_ascend_last_error(void) { return "injected bridge failure (test only)"; }
int dotllm_ascend_linear_f16(void *x, void *w, void *y, int64_t m, int64_t n, int64_t k, void *stream) {
    (void)x; (void)w; (void)y; (void)m; (void)n; (void)k; (void)stream;
    return 100999;
}
int dotllm_ascend_rms_norm_f16(void) { return 100999; }
int dotllm_ascend_add_f16(void) { return 100999; }
int dotllm_ascend_swiglu_f16(void) { return 100999; }
int dotllm_ascend_rope_f16(void) { return 100999; }
int dotllm_ascend_attention_f16(void) { return 100999; }
int dotllm_ascend_cast(void) { return 100999; }
int dotllm_ascend_linear_f32(void) { return 100999; }
int dotllm_ascend_rms_norm_f32(void) { return 100999; }
int dotllm_ascend_add_f32(void) { return 100999; }
int dotllm_ascend_swiglu_f32(void) { return 100999; }
int dotllm_ascend_rope_f32(void) { return 100999; }
int dotllm_ascend_attention_f32(void) { return 100999; }
#endif
