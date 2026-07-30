// ISO port of rope.cu.

#include <metal_stdlib>
using namespace metal;

kernel void rope_f16(
    device half*       q           [[ buffer(0) ]],
    device half*       k           [[ buffer(1) ]],
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
    int half_rope = rope_dim / 2;
    int total_q_pairs = seq_len * num_heads * half_rope;
    int total_k_pairs = seq_len * num_kv_heads * half_rope;

    // Process Q
    if ((int)idx < total_q_pairs)
    {
        int pair      = (int)idx % half_rope;
        int remainder = (int)idx / half_rope;
        int head      = remainder % num_heads;
        int t         = remainder / num_heads;

        int pos = positions[t];
        float freq = 1.0f / powr(theta, (float)(2 * pair) / (float)rope_dim);
        float angle = (float)pos * freq;
        float cos_val = cos(angle);
        float sin_val = sin(angle);

        int base_idx = t * num_heads * head_dim + head * head_dim;

        int i0, i1;
        if (rope_type == 1) // neox: [0..half, half..dim]
        {
            i0 = base_idx + pair;
            i1 = base_idx + pair + half_rope;
        }
        else // standard: [0,1], [2,3], ...
        {
            i0 = base_idx + 2 * pair;
            i1 = base_idx + 2 * pair + 1;
        }

        float v0 = float(q[i0]);
        float v1 = float(q[i1]);
        q[i0] = half(v0 * cos_val - v1 * sin_val);
        q[i1] = half(v0 * sin_val + v1 * cos_val);
    }

    // Process K (same logic, different head count and stride)
    if ((int)idx < total_k_pairs)
    {
        int pair      = (int)idx % half_rope;
        int remainder = (int)idx / half_rope;
        int head      = remainder % num_kv_heads;
        int t         = remainder / num_kv_heads;

        int pos = positions[t];
        float freq = 1.0f / powr(theta, (float)(2 * pair) / (float)rope_dim);
        float angle = (float)pos * freq;
        float cos_val = cos(angle);
        float sin_val = sin(angle);

        int base_idx = t * num_kv_heads * head_dim + head * head_dim;

        int i0, i1;
        if (rope_type == 1)
        {
            i0 = base_idx + pair;
            i1 = base_idx + pair + half_rope;
        }
        else
        {
            i0 = base_idx + 2 * pair;
            i1 = base_idx + 2 * pair + 1;
        }

        float v0 = float(k[i0]);
        float v1 = float(k[i1]);
        k[i0] = half(v0 * cos_val - v1 * sin_val);
        k[i1] = half(v0 * sin_val + v1 * cos_val);
    }
}
