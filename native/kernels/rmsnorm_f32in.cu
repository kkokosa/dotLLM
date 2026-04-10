// RMS Normalization with FP32 input, FP16 weight, FP16 output.
// Used when the residual stream is FP32 but downstream GEMM needs FP16 input.
// out[i] = FP16((FP32_input[i] / rms) * FP32_weight[i])

#include <cuda_fp16.h>  // needed for half output type

extern "C" __global__ void __launch_bounds__(256) rmsnorm_f32in_f16out(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    half* __restrict__ output,
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    const float* x = input + (size_t)row * n;
    half* y = output + (size_t)row * n;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float v = x[i];
        sum_sq += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);

    __shared__ float warp_sums[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        sum_sq = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    __shared__ float rms_inv;
    if (threadIdx.x == 0)
        rms_inv = rsqrtf(sum_sq / (float)n + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float v = x[i];
        float w = weight[i];
        y[i] = __float2half(v * rms_inv * w);
    }
}
