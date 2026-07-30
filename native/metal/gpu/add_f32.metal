// ISO port of add_f32.cu.

#include <metal_stdlib>
using namespace metal;

// ── add_f32 ──────────────────────────────────────────────────────────────────
// output[i] = a[i] + b[i]  (all FP32, in-place safe)
// Port of add_f32.cu::add_f32
kernel void add_f32(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant uint&      n      [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < n)
        output[idx] = a[idx] + b[idx];
}

// ── add_f32_f16 ──────────────────────────────────────────────────────────────
// output_f32[i] = a_f32[i] + b_f16[i]
// Used when adding FP16 projection output into FP32 residual stream.
// Port of add_f32.cu::add_f32_f16
kernel void add_f32_f16(
    device const float* a      [[buffer(0)]],
    device const half*  b      [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant uint&      n      [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < n)
        output[idx] = a[idx] + float(b[idx]);
}
