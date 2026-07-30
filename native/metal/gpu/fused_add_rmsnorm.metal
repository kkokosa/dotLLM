// ISO port of fused_add_rmsnorm.cu.

#include <metal_stdlib>
using namespace metal;

kernel void fused_add_rmsnorm_f16(
    device half* residual                   [[ buffer(0) ]],
    device const half* x                    [[ buffer(1) ]],
    device const half* weight               [[ buffer(2) ]],
    device half* output                     [[ buffer(3) ]],
    constant int& n                         [[ buffer(4) ]],
    constant float& eps                     [[ buffer(5) ]],
    
    uint block_id_x                         [[ threadgroup_position_in_grid ]],   // blockIdx.x
    uint threadIdx_x                        [[ thread_position_in_threadgroup ]], // threadIdx.x
    uint blockDim_x                         [[ threads_per_threadgroup ]],        // blockDim.x
    uint simd_group_width                   [[ threads_per_simdgroup]])
{
    const int row = block_id_x;
    device half* res_row = residual + (size_t)row * n;
    device const half* x_row = x + (size_t)row * n;
    device half* out_row = output + (size_t)row * n;
    
    // Pass 1: compute sum in FP32, write sum back to residual (FP16),
    // and accumulate sum-of-squares for RMS.
    // `(n & 3) == 0` ⟺ n is a multiple of 4: take the vectorized half4 path
    // (4 elements per load, n/4 iterations; the half4 cast also needs the
    // 8-byte-aligned rows that a multiple-of-4 n guarantees). Otherwise n/4
    // iterations would skip the trailing 1-3 elements and a half4 read could
    // spill past the row → fall back to the scalar loop. LLM hidden sizes are
    // always multiples of 4, so the fast path runs in practice; the scalar
    // branch is defensive correctness.
    float sum_sq = 0.0f;
    if ((n & 3) == 0) {
        float4 acc4 = float4(0.0f);
        uint nv = (uint)n >> 2;
        for (uint i = (uint)threadIdx_x; i < nv; i += (uint)blockDim_x) {
            float4 r = float4(*(device const half4*)(res_row + i * 4));
            float4 xv = float4(*(device const half4*)(x_row   + i * 4));
            float4 s = r + xv;
            *(device half4*)(res_row + i * 4) = half4(s);
            acc4 = fma(s, s, acc4);
        }
        sum_sq = acc4.x + acc4.y + acc4.z + acc4.w;
    } else {
        for (int i = threadIdx_x; i < n; i += blockDim_x)
        {
            float r = float(res_row[i]);
            float xi = float(x_row[i]);
            float sum = r + xi;
            res_row[i] = half(sum);
            sum_sq += sum * sum;
        }
    }

    // Warp reduction for sum_sq
    for (int offset = simd_group_width / 2; offset > 0; offset >>= 1)
        sum_sq += simd_shuffle_down(sum_sq, offset);

    threadgroup float warp_sums[32];
    int lane = threadIdx_x % simd_group_width;
    int warp_id = threadIdx_x / simd_group_width;

    if (lane == 0) warp_sums[warp_id] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_id == 0)
    {
        int num_warps = (blockDim_x + simd_group_width - 1) / simd_group_width;
        sum_sq = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = simd_group_width / 2; offset > 0; offset >>= 1)
            sum_sq += simd_shuffle_down(sum_sq, offset);
    }

    threadgroup float rms_inv = 0;
    if (threadIdx_x == 0)
        rms_inv = rsqrt(sum_sq / (float)n + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: normalize — read from residual (which now has the sum in FP16).
    // Vectorized when n % 4 == 0.
    if ((n & 3) == 0) {
        uint nv = (uint)n >> 2;
        for (uint i = (uint)threadIdx_x; i < nv; i += (uint)blockDim_x) {
            float4 v = float4(*(device const half4*)(res_row + i * 4));
            float4 w = float4(*(device const half4*)(weight  + i * 4));
            *(device half4*)(out_row + i * 4) = half4(v * rms_inv * w);
        }
    } else {
        for (int i = threadIdx_x; i < n; i += blockDim_x)
        {
            float v = float(res_row[i]);
            float w = float(weight[i]);
            out_row[i] = half(v * rms_inv * w);
        }
    }
}

