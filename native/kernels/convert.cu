// Type conversion kernels for dotLLM.
// Vectorized: half2/float2 for 2x throughput.

#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256) convert_f16_to_f32(
    const half* __restrict__ src,
    float* __restrict__ dst,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n / 2;

    if (idx < n2)
    {
        half2 in = reinterpret_cast<const half2*>(src)[idx];
        float2 out = __half22float2(in);
        reinterpret_cast<float2*>(dst)[idx] = out;
    }

    // Handle odd trailing element
    if ((n & 1) && idx == 0)
        dst[n - 1] = __half2float(src[n - 1]);
}

extern "C" __global__ void __launch_bounds__(256) convert_f32_to_f16(
    const float* __restrict__ src,
    half* __restrict__ dst,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n / 2;

    if (idx < n2)
    {
        float2 in = reinterpret_cast<const float2*>(src)[idx];
        half2 out = __float22half2_rn(in);
        reinterpret_cast<half2*>(dst)[idx] = out;
    }

    // Handle odd trailing element
    if ((n & 1) && idx == 0)
        dst[n - 1] = __float2half(src[n - 1]);
}
