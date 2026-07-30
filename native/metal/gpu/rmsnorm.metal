// ISO port of rmsnorm.cu.

#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_f16(
    device const half*  input   [[ buffer(0) ]],
    device const half*  weight  [[ buffer(1) ]],
    device       half*  output  [[ buffer(2) ]],
    constant     int&   n       [[ buffer(3) ]],
    constant     float& eps     [[ buffer(4) ]],
    uint row         [[ threadgroup_position_in_grid ]],
    uint threadIdx_x [[ thread_position_in_threadgroup ]],
    uint blockDim_x  [[ threads_per_threadgroup ]])
{
    // Each block processes one row of length n
    const int row_i = (int)row;
    device const half* x = input + (size_t)row_i * n;
    device       half* y = output + (size_t)row_i * n;

    // Step 1: compute sum of squares using FP32 accumulation
    float sum_sq = 0.0f;
    for (int i = (int)threadIdx_x; i < n; i += (int)blockDim_x)
    {
        float v = float(x[i]);
        sum_sq += v * v;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += simd_shuffle_down(sum_sq, offset);

    // Cross-warp reduction via threadgroup memory
    threadgroup float warp_sums[32]; // max 32 warps per block
    int lane = threadIdx_x % 32;
    int warp_id = threadIdx_x / 32;

    if (lane == 0)
        warp_sums[warp_id] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First warp reduces all warp sums
    if (warp_id == 0)
    {
        int num_warps = (blockDim_x + 31) / 32;
        sum_sq = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += simd_shuffle_down(sum_sq, offset);
    }

    // Broadcast result
    threadgroup float rms_inv;
    if (threadIdx_x == 0)
        rms_inv = rsqrt(sum_sq / (float)n + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: normalize and scale
    for (int i = (int)threadIdx_x; i < n; i += (int)blockDim_x)
    {
        float v = float(x[i]);
        float w = float(weight[i]);
        y[i] = half(v * rms_inv * w);
    }
}
