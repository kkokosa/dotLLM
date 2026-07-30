// ISO port of quantized_gemv_f32in.cu

#include <metal_stdlib>
using namespace metal;

kernel void quantized_gemv_q8_0_f32in(
    device const uchar* weight               [[ buffer(0) ]],
    device const float* x                    [[ buffer(1) ]],
    device float*       y                    [[ buffer(2) ]],
    constant int&       n                    [[ buffer(3) ]],
    constant int&       k                    [[ buffer(4) ]],

    uint row                                 [[ threadgroup_position_in_grid ]],   // blockIdx.x
    uint thread_idx_x                        [[ thread_position_in_threadgroup ]], // threadIdx.x
    uint block_dim_x                         [[ threads_per_threadgroup ]]         // blockDim.x
    )
{
    if ((int)row >= n) return;

    const int bpr = k / 32; // Q8_0 blocks per row
    device const uchar* w_row = weight + (size_t)row * bpr * 34;
    float acc = 0.0f;

    for (int b = (int)thread_idx_x; b < bpr; b += (int)block_dim_x)
    {
        device const uchar* block = w_row + b * 34;
        float d = float(*(device const half*)block);
        device const char* qs = (device const char*)(block + 2);
        float s = 0.0f;
        for (int j = 0; j < 32; j++) s += (float)qs[j] * x[b * 32 + j];
        acc += d * s;
    }

    acc = simd_sum(acc);

    threadgroup float ws[32]; // one slot per simd-group (max 32 simd-groups for 1024 threads)
    uint lane = thread_idx_x % 32;
    uint wid  = thread_idx_x / 32;
    if (lane == 0) ws[wid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (wid == 0) {
        uint nw = (block_dim_x + 31) / 32; // number of active simd-groups
        acc = (lane < nw) ? ws[lane] : 0.0f;
        acc = simd_sum(acc);
    }

    if (thread_idx_x == 0) y[row] = acc;
}
