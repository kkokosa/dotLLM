// Type conversion kernels for dotLLM.
// Vectorized: half2/float2 for 2x throughput.

#include <cuda_fp16.h>
#include <stdint.h>
#ifndef NDEBUG
#include <assert.h>
#endif

__device__ __forceinline__ bool is_aligned_4(const void* ptr)
{
    return (reinterpret_cast<uintptr_t>(ptr) & 0x3) == 0;
}

__device__ __forceinline__ bool is_aligned_8(const void* ptr)
{
    return (reinterpret_cast<uintptr_t>(ptr) & 0x7) == 0;
}

extern "C" __global__ void __launch_bounds__(256) convert_f16_to_f32(
    const half* __restrict__ src,
    float* __restrict__ dst,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n / 2;
    bool can_vectorize = is_aligned_4(src) && is_aligned_8(dst);

    if (can_vectorize && idx < n2)
    {
        half2 in = reinterpret_cast<const half2*>(src)[idx];
        float2 out = __half22float2(in);
#ifndef NDEBUG
        assert(is_aligned_4(&reinterpret_cast<const half2*>(src)[idx]));
        assert(is_aligned_8(&reinterpret_cast<float2*>(dst)[idx]));
#endif
        reinterpret_cast<float2*>(dst)[idx] = out;
    }
    else if (!can_vectorize && idx < n)
    {
        dst[idx] = __half2float(src[idx]);
    }

    if (can_vectorize && (n & 1) && idx == n2)
        dst[n - 1] = __half2float(src[n - 1]);
}

extern "C" __global__ void __launch_bounds__(256) convert_f32_to_f16(
    const float* __restrict__ src,
    half* __restrict__ dst,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n / 2;
    bool can_vectorize = is_aligned_8(src) && is_aligned_4(dst);

    if (can_vectorize && idx < n2)
    {
        float2 in = reinterpret_cast<const float2*>(src)[idx];
        half2 out = __float22half2_rn(in);
#ifndef NDEBUG
        assert(is_aligned_8(&reinterpret_cast<const float2*>(src)[idx]));
        assert(is_aligned_4(&reinterpret_cast<half2*>(dst)[idx]));
#endif
        reinterpret_cast<half2*>(dst)[idx] = out;
    }
    else if (!can_vectorize && idx < n)
    {
        dst[idx] = __float2half(src[idx]);
    }

    if (can_vectorize && (n & 1) && idx == n2)
        dst[n - 1] = __float2half(src[n - 1]);
}
