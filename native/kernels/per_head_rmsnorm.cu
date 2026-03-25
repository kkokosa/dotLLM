// Per-head RMS normalization kernel (QK-norm, Qwen3-style) for dotLLM.
// Normalizes each head vector independently: qk[t, h, :headDim]

#include <cuda_fp16.h>

extern "C" __global__ void per_head_rmsnorm_f16(
    half* __restrict__ qk,
    const half* __restrict__ weight,
    const float eps,
    const int num_heads,
    const int head_dim,
    const int seq_len)
{
    // One block per (token, head) pair
    int block_id = blockIdx.x;
    int t = block_id / num_heads;
    int h = block_id % num_heads;

    if (t >= seq_len) return;

    int stride = num_heads * head_dim;
    half* vec = qk + (size_t)t * stride + h * head_dim;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    {
        float v = __half2float(vec[i]);
        sum_sq += v * v;
    }

    // Warp reduction
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
        rms_inv = rsqrtf(sum_sq / (float)head_dim + eps);
    __syncthreads();

    // Normalize and scale
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    {
        float v = __half2float(vec[i]);
        float w = __half2float(weight[i]);
        vec[i] = __float2half(v * rms_inv * w);
    }
}
