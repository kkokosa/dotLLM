// Softmax kernel for dotLLM.
// Numerically stable: subtract max, exp, normalize.
// One block per row, warp reduction. FP16 in/out, FP32 accumulation.

#include <cuda_fp16.h>
#include <float.h>

extern "C" __global__ void softmax_f16(
    const half* __restrict__ input,
    half* __restrict__ output,
    const int rows,
    const int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const half* x = input + (size_t)row * cols;
    half* y = output + (size_t)row * cols;

    // Step 1: find max
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float v = __half2float(x[i]);
        if (v > max_val) max_val = v;
    }

    // Warp reduction for max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        float other = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        if (other > max_val) max_val = other;
    }

    __shared__ float warp_vals[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_vals[warp_id] = max_val;
    __syncthreads();

    if (warp_id == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        max_val = (lane < num_warps) ? warp_vals[lane] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            float other = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
            if (other > max_val) max_val = other;
        }
    }

    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = max_val;
    __syncthreads();
    max_val = shared_max;

    // Step 2: sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float v = __half2float(x[i]);
        sum_exp += expf(v - max_val);
    }

    // Warp reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);

    if (lane == 0) warp_vals[warp_id] = sum_exp;
    __syncthreads();

    if (warp_id == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        sum_exp = (lane < num_warps) ? warp_vals[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    }

    __shared__ float shared_sum_inv;
    if (threadIdx.x == 0) shared_sum_inv = 1.0f / sum_exp;
    __syncthreads();

    // Step 3: normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float v = __half2float(x[i]);
        y[i] = __float2half(expf(v - max_val) * shared_sum_inv);
    }
}
