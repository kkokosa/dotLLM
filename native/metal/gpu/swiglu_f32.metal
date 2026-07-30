// ISO port of swiglu_f32.cu.
// out[i] = SiLU(gate[i]) * up[i]  where SiLU(x) = x / (1 + exp(-x))

#include <metal_stdlib>
using namespace metal;

kernel void swiglu_f32(
    device const float* gate   [[buffer(0)]],
    device const float* up     [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant uint&      length [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= length) return;

    float g = gate[idx];
    output[idx] = (g / (1.0f + exp(-g))) * up[idx];
}
