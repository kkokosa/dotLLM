// ISO port of per_head_rmsnorm.cu.
//
// Per-head RMS Normalization (FP16, in-place). FP16 I/O with FP32 accumulation.
// One threadgroup per (token, head) pair.

#include <metal_stdlib>
using namespace metal;

kernel void per_head_rmsnorm_f16(
    device       half*  qk        [[ buffer(0) ]],   // in-place [seqLen, num_heads, head_dim]
    device const half*  weight    [[ buffer(1) ]],   // [head_dim]
    constant     float& eps       [[ buffer(2) ]],
    constant     int&   num_heads [[ buffer(3) ]],
    constant     int&   head_dim  [[ buffer(4) ]],
    constant     int&   seq_len   [[ buffer(5) ]],
    uint block_id    [[ threadgroup_position_in_grid ]],   // blockIdx.x
    uint threadIdx_x [[ thread_position_in_threadgroup ]], // threadIdx.x
    uint blockDim_x  [[ threads_per_threadgroup ]])        // blockDim.x
{
    // One block per (token, head) pair — mirrors per_head_rmsnorm.cu.
    int t = block_id / num_heads;
    int h = block_id % num_heads;
    if (t >= seq_len) return;

    int stride = num_heads * head_dim;
    device half* vec = qk + (size_t)t * stride + h * head_dim;

    // Compute sum of squares (half → float on load).
    float sum_sq = 0.0f;
    for (int i = threadIdx_x; i < head_dim; i += blockDim_x) {
        float v = float(vec[i]);
        sum_sq += v * v;
    }

    // Warp (simd-group) reduction. simd_sum replaces the CUDA __shfl_down_sync
    // tree-reduce; unlike __shfl_down_sync it returns the sum to ALL lanes.
    sum_sq = simd_sum(sum_sq);

    threadgroup float warp_sums[32];
    int lane    = threadIdx_x % 32;
    int warp_id = threadIdx_x / 32;
    if (lane == 0) warp_sums[warp_id] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_id == 0) {
        int num_warps = (blockDim_x + 31) / 32;
        sum_sq = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        sum_sq = simd_sum(sum_sq);
    }

    threadgroup float rms_inv;
    if (threadIdx_x == 0)
        rms_inv = rsqrt(sum_sq / (float)head_dim + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize and scale.
    for (int i = threadIdx_x; i < head_dim; i += blockDim_x) {
        float v = float(vec[i]);
        float w = float(weight[i]);
        vec[i] = half(v * rms_inv * w);
    }
}
