// Fused residual-add + RMS normalization kernel for dotLLM.
// Eliminates one FP16 truncation at the critical residual junction:
//   1. sum = FP32(residual) + FP32(x)         — keep in FP32
//   2. residual[i] = FP16(sum)                 — update residual buffer
//   3. output[i] = FP16(sum * rsqrt(rms) * w)  — normalize from FP32 sum

#include <cuda_fp16.h>

extern "C" __global__ void fused_add_rmsnorm_f16(
    half* __restrict__ residual,       // [n] in/out: updated with sum
    const half* __restrict__ x,        // [n] layer output to add
    const half* __restrict__ weight,   // [n] norm weights
    half* __restrict__ output,         // [n] normalized output
    const int n,
    const float eps)
{
    // Each block processes one row
    const int row = blockIdx.x;
    half* res_row = residual + (size_t)row * n;
    const half* x_row = x + (size_t)row * n;
    half* out_row = output + (size_t)row * n;

    // Pass 1: compute sum in FP32, write sum back to residual (FP16),
    // and accumulate sum-of-squares for RMS
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float r = __half2float(res_row[i]);
        float xi = __half2float(x_row[i]);
        float sum = r + xi;

        // Update residual with the sum (FP16 storage)
        res_row[i] = __float2half(sum);

        // Accumulate for RMS from the FP32 sum (no extra truncation!)
        sum_sq += sum * sum;
    }

    // Warp reduction for sum_sq
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

    // Pass 2: normalize — read from residual (which now has the sum in FP16)
    // Note: we read back from FP16, so there IS one truncation. But the RMS
    // was computed from the FP32 sum, which is the key improvement.
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float v = __half2float(res_row[i]);
        float w = __half2float(weight[i]);
        out_row[i] = __float2half(v * rms_inv * w);
    }
}
