// Fused SwiGLU activation kernel for dotLLM.
// out[i] = SiLU(gate[i]) * up[i] = gate[i] * sigmoid(gate[i]) * up[i]
// Vectorized: half2 loads/stores, FP32 computation for sigmoid precision.

#include <cuda_fp16.h>
#include <stdint.h>
#ifndef NDEBUG
#include <assert.h>
#endif

__device__ __forceinline__ bool is_aligned_4(const void* ptr)
{
    return (reinterpret_cast<uintptr_t>(ptr) & 0x3) == 0;
}

extern "C" __global__ void __launch_bounds__(256) swiglu_f16(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    const int n,
    const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * seq_len;
    int total2 = total / 2;
    bool can_vectorize = is_aligned_4(gate) && is_aligned_4(up) && is_aligned_4(output);

    if (can_vectorize && idx < total2)
    {
        half2 g2 = reinterpret_cast<const half2*>(gate)[idx];
        half2 u2 = reinterpret_cast<const half2*>(up)[idx];
#ifndef NDEBUG
        assert(is_aligned_4(&reinterpret_cast<const half2*>(gate)[idx]));
        assert(is_aligned_4(&reinterpret_cast<const half2*>(up)[idx]));
        assert(is_aligned_4(&reinterpret_cast<half2*>(output)[idx]));
#endif

        float g0 = __low2float(g2), g1 = __high2float(g2);
        float u0 = __low2float(u2), u1 = __high2float(u2);

        // SiLU(g) = g / (1 + exp(-g))
        float s0 = g0 / (1.0f + expf(-g0)) * u0;
        float s1 = g1 / (1.0f + expf(-g1)) * u1;

        reinterpret_cast<half2*>(output)[idx] = __floats2half2_rn(s0, s1);
    }
    else if (!can_vectorize && idx < total)
    {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        output[idx] = __float2half(g / (1.0f + expf(-g)) * u);
    }

    if (can_vectorize && (total & 1) && idx == total2)
    {
        int last = total - 1;
        float g = __half2float(gate[last]);
        float u = __half2float(up[last]);
        output[last] = __float2half(g / (1.0f + expf(-g)) * u);
    }
}
