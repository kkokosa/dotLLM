// Bias addition kernel for dotLLM.
// output[t, i] += bias[i]  for t in [0, seqLen)
// Vectorized: half2 packed operations when dim is even.

#include <cuda_fp16.h>
#include <stdint.h>
#ifndef NDEBUG
#include <assert.h>
#endif

__device__ __forceinline__ bool is_aligned_4(const void* ptr)
{
    return (reinterpret_cast<uintptr_t>(ptr) & 0x3) == 0;
}

extern "C" __global__ void __launch_bounds__(256) bias_add_f16(
    half* __restrict__ output,
    const half* __restrict__ bias,
    const int dim,
    const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dim * seq_len;
    int dim2 = dim / 2;
    int total2 = total / 2;
    bool can_vectorize = is_aligned_4(output) && is_aligned_4(bias);

    if (can_vectorize && idx < total2)
    {
        half2* out2 = reinterpret_cast<half2*>(output);
        const half2* bias2 = reinterpret_cast<const half2*>(bias);

        // Map half2 index back to row/col pair index
        // Each row has dim elements = dim/2 half2 elements
        int col2 = idx % dim2;
#ifndef NDEBUG
        assert(is_aligned_4(&out2[idx]));
        assert(is_aligned_4(&bias2[col2]));
#endif
        out2[idx] = __hadd2(out2[idx], bias2[col2]);
    }
    else if (!can_vectorize && idx < total)
    {
        int col = idx % dim;
        output[idx] = __float2half(__half2float(output[idx]) + __half2float(bias[col]));
    }

    if (can_vectorize && (total & 1) && idx == total2)
    {
        int last = total - 1;
        int col = last % dim;
        output[last] = __float2half(__half2float(output[last]) + __half2float(bias[col]));
    }
}
