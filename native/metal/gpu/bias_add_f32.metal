// ISO port of bias_add_f32.cu.

#include <metal_stdlib>
using namespace metal;

kernel void bias_add_f32(
    device float*       output  [[buffer(0)]],
    device const half*  bias    [[buffer(1)]],
    constant uint&      dim     [[buffer(2)]],
    constant uint&      seq_len [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < dim * seq_len)
        output[idx] += float(bias[idx % dim]);
}
