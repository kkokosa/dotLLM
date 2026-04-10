// RoPE kernel with FP32 Q/K data.
// Q and K are processed independently — reuse attempts hurt GQA models.

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

    if (idx < total_q_pairs)
    {
        int pair = idx % half_rope;
        int remainder = idx / half_rope;
        int head = remainder % num_heads;
        int t = remainder / num_heads;

        float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
        float angle = (float)positions[t] * freq;
        float cos_val = cosf(angle), sin_val = sinf(angle);

        int base_idx = t * num_heads * head_dim + head * head_dim;
        int i0 = (rope_type == 1) ? base_idx + pair : base_idx + 2 * pair;
        int i1 = (rope_type == 1) ? base_idx + pair + half_rope : base_idx + 2 * pair + 1;

        float v0 = q[i0], v1 = q[i1];
        q[i0] = v0 * cos_val - v1 * sin_val;
        q[i1] = v0 * sin_val + v1 * cos_val;
    }

    if (idx < total_k_pairs)
    {
        int pair = idx % half_rope;
        int remainder = idx / half_rope;
        int head = remainder % num_kv_heads;
        int t = remainder / num_kv_heads;

        float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
        float angle = (float)positions[t] * freq;
        float cos_val = cosf(angle), sin_val = sinf(angle);

        int base_idx = t * num_kv_heads * head_dim + head * head_dim;
        int i0 = (rope_type == 1) ? base_idx + pair : base_idx + 2 * pair;
        int i1 = (rope_type == 1) ? base_idx + pair + half_rope : base_idx + 2 * pair + 1;

        float v0 = k[i0], v1 = k[i1];
        k[i0] = v0 * cos_val - v1 * sin_val;
        k[i1] = v0 * sin_val + v1 * cos_val;
    }
}
