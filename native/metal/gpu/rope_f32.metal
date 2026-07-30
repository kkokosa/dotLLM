// ISO port of rope_f32.cu.

#include <metal_stdlib>
using namespace metal;

kernel void rope_f32(
    device float*       q           [[ buffer(0) ]],
    device float*       k           [[ buffer(1) ]],
    device const int*   positions   [[ buffer(2) ]],
    constant int&       seq_len     [[ buffer(3) ]],
    constant int&       num_heads   [[ buffer(4) ]],
    constant int&       num_kv_heads[[ buffer(5) ]],
    constant int&       head_dim    [[ buffer(6) ]],
    constant int&       rope_dim    [[ buffer(7) ]],
    constant float&     theta       [[ buffer(8) ]],
    constant int&       rope_type   [[ buffer(9) ]],
    uint idx                        [[ thread_position_in_grid ]])
{
    int half_rope      = rope_dim / 2;
    int total_q_pairs  = seq_len * num_heads    * half_rope;
    int total_k_pairs  = seq_len * num_kv_heads * half_rope;

    if ((int)idx < total_q_pairs)
    {
        int pair      = (int)idx % half_rope;
        int remainder = (int)idx / half_rope;
        int head      = remainder % num_heads;
        int t         = remainder / num_heads;

        float freq    = 1.0f / powr(theta, (float)(2 * pair) / (float)rope_dim);
        float angle   = (float)positions[t] * freq;
        float cos_val = cos(angle);
        float sin_val = sin(angle);

        int base = t * num_heads * head_dim + head * head_dim;
        int i0   = (rope_type == 1) ? base + pair            : base + 2 * pair;
        int i1   = (rope_type == 1) ? base + pair + half_rope : base + 2 * pair + 1;

        float v0 = q[i0], v1 = q[i1];
        q[i0] = v0 * cos_val - v1 * sin_val;
        q[i1] = v0 * sin_val + v1 * cos_val;
    }

    if ((int)idx < total_k_pairs)
    {
        int pair      = (int)idx % half_rope;
        int remainder = (int)idx / half_rope;
        int head      = remainder % num_kv_heads;
        int t         = remainder / num_kv_heads;

        float freq    = 1.0f / powr(theta, (float)(2 * pair) / (float)rope_dim);
        float angle   = (float)positions[t] * freq;
        float cos_val = cos(angle);
        float sin_val = sin(angle);

        int base = t * num_kv_heads * head_dim + head * head_dim;
        int i0   = (rope_type == 1) ? base + pair            : base + 2 * pair;
        int i1   = (rope_type == 1) ? base + pair + half_rope : base + 2 * pair + 1;

        float v0 = k[i0], v1 = k[i1];
        k[i0] = v0 * cos_val - v1 * sin_val;
        k[i1] = v0 * sin_val + v1 * cos_val;
    }
}
