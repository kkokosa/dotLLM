// ISO port of convert.cu.

#include <metal_stdlib>
using namespace metal;

kernel void convert_f16_to_f32(
    device const half*  src [[ buffer(0) ]],
    device       float* dst [[ buffer(1) ]],
    constant     int&   n   [[ buffer(2) ]],
    uint idx [[ thread_position_in_grid ]])
{
    if ((int)idx >= n) return;
    dst[idx] = float(src[idx]);
}

kernel void convert_f32_to_f16(
    device const float* src [[ buffer(0) ]],
    device       half*  dst [[ buffer(1) ]],
    constant     int&   n   [[ buffer(2) ]],
    uint idx [[ thread_position_in_grid ]])
{
    if ((int)idx >= n) return;
    dst[idx] = half(src[idx]);
}
