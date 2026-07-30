// Element-wise addition kernel for dotLLM.
// output[i] = a[i] + b[i]  (FP16, in-place safe: output may alias a or b)
// Vectorized: half2 packed operations process 2 elements per thread.

#include <cuda_fp16.h>
#include <stdint.h>
#ifndef NDEBUG
#include <assert.h>
#endif

__device__ __forceinline__ bool is_aligned_4(const void* ptr)
{
    return (reinterpret_cast<uintptr_t>(ptr) & 0x3) == 0;
}

extern "C" __global__ void __launch_bounds__(256) add_f16(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ output,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n / 2;
    bool can_vectorize = is_aligned_4(a) && is_aligned_4(b) && is_aligned_4(output);

    if (can_vectorize && idx < n2)
    {
        const half2* a2 = reinterpret_cast<const half2*>(a);
        const half2* b2 = reinterpret_cast<const half2*>(b);
        half2* out2 = reinterpret_cast<half2*>(output);
#ifndef NDEBUG
        assert(is_aligned_4(&a2[idx]));
        assert(is_aligned_4(&b2[idx]));
        assert(is_aligned_4(&out2[idx]));
#endif
        out2[idx] = __hadd2(a2[idx], b2[idx]);
    }
    else if (!can_vectorize && idx < n)
    {
        output[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
    }

    if (can_vectorize && (n & 1) && idx == n2)
    {
        int last = n - 1;
        output[last] = __float2half(__half2float(a[last]) + __half2float(b[last]));
    }
}
