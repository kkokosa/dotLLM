// ISO port of rmsnorm_f32.cu.

#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_f32(
    device const float* input   [[ buffer(0) ]],
    device const float* weight  [[ buffer(1) ]],
    device       float* output  [[ buffer(2) ]],
    constant     int&   n       [[ buffer(3) ]],
    constant     float& eps     [[ buffer(4) ]],
    uint row         [[ threadgroup_position_in_grid ]],
    uint threadIdx_x [[ thread_position_in_threadgroup ]],
    uint blockDim_x  [[ threads_per_threadgroup ]])
{
    const int row_i = (int)row;
    device const float* x = input + (size_t)row_i * n;
    device       float* y = output + (size_t)row_i * n;

    float sum_sq = 0.0f;
    for (int i = (int)threadIdx_x; i < n; i += (int)blockDim_x)
    { float v = x[i]; sum_sq += v * v; }

    for (int off = 32/2; off > 0; off >>= 1)
        sum_sq += simd_shuffle_down(sum_sq, off);
    threadgroup float ws[32];
    int lane = threadIdx_x % 32, wid = threadIdx_x / 32;
    if (lane == 0) ws[wid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (wid == 0) {
        int nw = (blockDim_x + 31) / 32;
        sum_sq = (lane < nw) ? ws[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            sum_sq += simd_shuffle_down(sum_sq, off);
    }
    threadgroup float ri;
    if (threadIdx_x == 0) ri = rsqrt(sum_sq / (float)n + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = (int)threadIdx_x; i < n; i += (int)blockDim_x)
        y[i] = x[i] * ri * weight[i];
}
