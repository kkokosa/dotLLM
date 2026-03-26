// Element-wise addition kernel for dotLLM.
// output[i] = a[i] + b[i]  (FP16, in-place safe: output may alias a or b)

#include <cuda_fp16.h>

extern "C" __global__ void add_f16(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ output,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        output[idx] = __float2half(va + vb);
    }
}
