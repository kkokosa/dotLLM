// ISO port of softmax.cu.

#include <metal_stdlib>
using namespace metal;

kernel void softmax_f16(
    device const half* input   [[buffer(0)]],
    device       half* output  [[buffer(1)]],
    constant int&      rows    [[buffer(2)]],
    constant int&      cols    [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgw [[threads_per_threadgroup]])
{
    if (row >= (uint)rows) return;

    device const half* x = input  + (ulong)row * cols;
    device       half* y = output + (ulong)row * cols;

    constexpr uint WARP_SIZE = 32u;
    uint lane      = tid % WARP_SIZE;
    uint warp_id   = tid / WARP_SIZE;
    uint num_warps = (tgw + WARP_SIZE - 1u) / WARP_SIZE;

    // Scratch for cross-warp reduction — max 8 simd groups for 256 threads
    threadgroup float warp_scratch[8];
    threadgroup float shared_max = 0;
    threadgroup float shared_sum_inv = 0;

    // ── Pass 1: find max ──────────────────────────────────────────────────────
    float max_val = -MAXFLOAT;
    for (uint i = tid; i < (uint)cols; i += tgw)
        max_val = max(max_val, float(x[i]));

    // Intra-warp reduce — simd_max broadcasts to ALL lanes
    max_val = simd_max(max_val);
    if (lane == 0) warp_scratch[warp_id] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-warp reduce in warp 0
    if (warp_id == 0)
    {
        float v = (lane < num_warps) ? warp_scratch[lane] : -MAXFLOAT;
        max_val = simd_max(v);
    }
    if (tid == 0) shared_max = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_val = shared_max;

    // ── Pass 2: compute exp(x - max), store to output, accumulate sum ─────────
    float sum_exp = 0.0f;
    for (uint i = tid; i < (uint)cols; i += tgw)
    {
        float e = exp(float(x[i]) - max_val);
        sum_exp += e;
        y[i] = half(e);
    }

    // Intra-warp reduce
    sum_exp = simd_sum(sum_exp);
    if (lane == 0) warp_scratch[warp_id] = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-warp reduce in warp 0
    if (warp_id == 0)
    {
        float v = (lane < num_warps) ? warp_scratch[lane] : 0.0f;
        sum_exp = simd_sum(v);
    }
    if (tid == 0) shared_sum_inv = 1.0f / sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_inv = shared_sum_inv;

    // ── Pass 3: normalize in-place from stored exp values ────────────────────
    for (uint i = tid; i < (uint)cols; i += tgw)
        y[i] = half(float(y[i]) * sum_inv);
}
