// RMS Normalization kernel for dotLLM.
// out[i] = (input[i] / rms) * weight[i], rms = sqrt(mean(input^2) + eps)
// FP16 in/out, FP32 accumulation for numerical stability.
// One block per row, warp shuffle reduction.

#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256) rmsnorm_f16(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const int n,
    const float eps)
{
    // Each block processes one row of length n
    const int row = blockIdx.x;
    const half* x = input + (size_t)row * n;
    half* y = output + (size_t)row * n;

    // Step 1: compute sum of squares using FP32 accumulation
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float v = __half2float(x[i]);
        sum_sq += v * v;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);

    // Cross-warp reduction via shared memory
    __shared__ float warp_sums[32]; // max 32 warps per block
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0)
        warp_sums[warp_id] = sum_sq;
    __syncthreads();

    // First warp reduces all warp sums
    if (warp_id == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        sum_sq = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // Broadcast result
    __shared__ float rms_inv;
    if (threadIdx.x == 0)
        rms_inv = rsqrtf(sum_sq / (float)n + eps);
    __syncthreads();

    // Step 2: normalize and scale
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float v = __half2float(x[i]);
        float w = __half2float(weight[i]);
        y[i] = __float2half(v * rms_inv * w);
    }
}
