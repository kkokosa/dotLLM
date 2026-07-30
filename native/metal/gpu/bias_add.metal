// ISO port of bias_add.cu.

#include <metal_stdlib>
using namespace metal;

kernel void bias_add_f16(
    device half*       output  [[buffer(0)]],
    device const half* bias    [[buffer(1)]],
    constant uint&     dim     [[buffer(2)]],
    constant uint&     seq_len [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    uint total  = dim * seq_len;
    uint dim2   = dim / 2u;
    uint total2 = total / 2u;

    // dim is always even for transformer hidden sizes, so half2 is safe
    if (idx < total2)
    {
        device       half2* out2  = (device half2*)output;
        device const half2* bias2 = (device const half2*)bias;

        // Map half2 index back to bias column pair
        uint col2 = idx % dim2;
        out2[idx] = out2[idx] + bias2[col2];
    }

    // Handle odd dim (shouldn't happen for transformers, but be safe)
    if ((total & 1u) && idx == 0u)
    {
        uint last = total - 1u;
        uint col  = last % dim;
        output[last] = output[last] + bias[col];
    }
}
