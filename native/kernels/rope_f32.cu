// RoPE kernel with FP32 Q/K data.
// Optimized: freq/cos/sin computed once and reused for both Q and K.

#include <math.h>

extern "C" __global__ void __launch_bounds__(256) rope_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ positions,
    const int seq_len, const int num_heads, const int num_kv_heads,
    const int head_dim, const int rope_dim, const float theta, const int rope_type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_rope = rope_dim / 2;
    int total_q_pairs = seq_len * num_heads * half_rope;
    int total_k_pairs = seq_len * num_kv_heads * half_rope;

    if (idx >= total_q_pairs && idx >= total_k_pairs) return;

    // Compute freq/cos/sin once
    int pair, t, pos;
    float cos_val, sin_val;

    if (idx < total_q_pairs)
    {
        pair = idx % half_rope;
        int remainder = idx / half_rope;
        t = remainder / num_heads;
        pos = positions[t];
    }
    else
    {
        pair = idx % half_rope;
        int remainder = idx / half_rope;
        t = remainder / num_kv_heads;
        pos = positions[t];
    }

    float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
    float angle = (float)pos * freq;
    cos_val = cosf(angle);
    sin_val = sinf(angle);

    // Apply to Q
    if (idx < total_q_pairs)
    {
        int remainder = idx / half_rope;
        int head = remainder % num_heads;

        int base_idx = t * num_heads * head_dim + head * head_dim;
        int i0 = (rope_type == 1) ? base_idx + pair : base_idx + 2 * pair;
        int i1 = (rope_type == 1) ? base_idx + pair + half_rope : base_idx + 2 * pair + 1;

        float v0 = q[i0], v1 = q[i1];
        q[i0] = v0 * cos_val - v1 * sin_val;
        q[i1] = v0 * sin_val + v1 * cos_val;
    }

    // Apply to K (reusing cos/sin when position matches)
    if (idx < total_k_pairs)
    {
        int remainder = idx / half_rope;
        int head = remainder % num_kv_heads;
        int k_t = remainder / num_kv_heads;
        int k_pos = positions[k_t];

        float k_cos = cos_val, k_sin = sin_val;
        if (k_pos != pos || (idx >= total_q_pairs))
        {
            float k_freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
            float k_angle = (float)k_pos * k_freq;
            k_cos = cosf(k_angle);
            k_sin = sinf(k_angle);
        }

        int base_idx = k_t * num_kv_heads * head_dim + head * head_dim;
        int i0 = (rope_type == 1) ? base_idx + pair : base_idx + 2 * pair;
        int i1 = (rope_type == 1) ? base_idx + pair + half_rope : base_idx + 2 * pair + 1;

        float v0 = k[i0], v1 = k[i1];
        k[i0] = v0 * k_cos - v1 * k_sin;
        k[i1] = v0 * k_sin + v1 * k_cos;
    }
}
