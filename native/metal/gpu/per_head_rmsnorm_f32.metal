// ISO port of per_head_rmsnorm_f32.cu.

#include <metal_stdlib>
using namespace metal;

kernel void per_head_rmsnorm_f32(
    device float* qk            [[ buffer(0) ]],
    device const float* weight  [[ buffer(1) ]],
    constant float& eps         [[ buffer(2) ]],
    constant int& num_heads     [[ buffer(3) ]],
    constant int& head_dim      [[ buffer(4) ]],
    constant int& seq_len       [[ buffer(5) ]],
    
    uint block_id               [[ threadgroup_position_in_grid ]],   // blockIdx.x
    uint threadIdx_x            [[ thread_position_in_threadgroup ]], // threadIdx.x
    uint blockDim_x             [[ threads_per_threadgroup ]],        // blockDim.x
    uint simd_group_width       [[threads_per_simdgroup]])
{
    uint t = block_id / (uint)num_heads;
    uint h = block_id % (uint)num_heads;
    if (t >= (uint)seq_len) return;

    device float* vec = qk + (size_t)t * (size_t)num_heads * head_dim + h * head_dim;
    float sum_sq = 0.0f;
    
    for (int i = threadIdx_x; i < head_dim; i += blockDim_x) {
        float v = vec[i];
        sum_sq += v * v;
    }

    for (int off = simd_group_width/2; off > 0; off >>= 1)
        sum_sq += simd_shuffle_down(sum_sq, off);
    
    threadgroup float ws[32];
    int lane = threadIdx_x % simd_group_width;
    int wid = threadIdx_x / simd_group_width;
    
    if (lane == 0)
        ws[wid] = sum_sq;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (wid == 0) {
        int nw = (blockDim_x + simd_group_width-1)/simd_group_width;
        sum_sq = (lane<nw)?ws[lane]:0.0f;
        
        for (int off=simd_group_width/2; off>0; off>>=1)
            sum_sq += simd_shuffle_down(sum_sq, off);
    }
    
    threadgroup float ri = 0;
    if (threadIdx_x==0)
        ri = rsqrt(sum_sq/(float)head_dim+eps);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (int i = threadIdx_x; i < head_dim; i += blockDim_x)
        vec[i] = vec[i] * ri * weight[i];
}

