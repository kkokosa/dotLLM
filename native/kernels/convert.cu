// Type conversion kernels for dotLLM.

#include <cuda_fp16.h>

extern "C" __global__ void convert_f16_to_f32(
    const half* __restrict__ src,
    float* __restrict__ dst,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        dst[idx] = __half2float(src[idx]);
}

extern "C" __global__ void convert_f32_to_f16(
    const float* __restrict__ src,
    half* __restrict__ dst,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        dst[idx] = __float2half(src[idx]);
}
