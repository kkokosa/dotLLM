// Element-wise addition kernel for dotLLM.
// output[i] = a[i] + b[i]  (FP16, in-place safe: output may alias a or b)
// Vectorized: half2 packed operations process 2 elements per thread.

#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256) add_f16(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ output,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n / 2;

    if (idx < n2)
    {
        const half2* a2 = reinterpret_cast<const half2*>(a);
        const half2* b2 = reinterpret_cast<const half2*>(b);
        half2* out2 = reinterpret_cast<half2*>(output);
        out2[idx] = __hadd2(a2[idx], b2[idx]);
    }

    // Handle odd trailing element (single thread)
    if ((n & 1) && idx == 0)
    {
        int last = n - 1;
        output[last] = __float2half(__half2float(a[last]) + __half2float(b[last]));
    }
}
