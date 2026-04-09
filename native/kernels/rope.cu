// Rotary Position Embedding (RoPE) kernel for dotLLM.
// In-place rotation on Q[seqLen, numHeads * headDim] and K[seqLen, numKvHeads * headDim].
// Computes cos/sin from theta in-kernel (no precomputed table upload needed).
// Optimized: freq/cos/sin computed once per thread and reused for both Q and K.

#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256) rope_f16(
    half* __restrict__ q,
    half* __restrict__ k,
    const int* __restrict__ positions,  // [seqLen] on device
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int rope_dim,
    const float theta,
    const int rope_type)  // 0 = standard, 1 = neox interleaved
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_rope = rope_dim / 2;
    int total_q_pairs = seq_len * num_heads * half_rope;
    int total_k_pairs = seq_len * num_kv_heads * half_rope;

    if (idx >= total_q_pairs && idx >= total_k_pairs) return;

    // Compute freq/cos/sin once — shared between Q and K when thread covers both
    // For Q: decompose idx into (t, head, pair)
    // Both Q and K use the same pair index for the frequency computation
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
        // Only K — decompose differently
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

        int q_stride = num_heads * head_dim;
        int base_idx = t * q_stride + head * head_dim;

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

        float v0 = __half2float(q[i0]);
        float v1 = __half2float(q[i1]);
        q[i0] = __float2half(v0 * cos_val - v1 * sin_val);
        q[i1] = __float2half(v0 * sin_val + v1 * cos_val);
    }

    // Apply to K (reusing cos_val, sin_val — same pair and position)
    if (idx < total_k_pairs)
    {
        int remainder = idx / half_rope;
        int head = remainder % num_kv_heads;
        // For K, t may differ from Q's t when num_kv_heads != num_heads
        int k_t = remainder / num_kv_heads;
        int k_pos = positions[k_t];

        // If K's position differs from what we computed cos/sin for, recompute
        // This happens when num_kv_heads < num_heads and the same idx maps to
        // a different (t, head) decomposition for K vs Q
        float k_cos = cos_val, k_sin = sin_val;
        if (k_pos != pos || (idx >= total_q_pairs))
        {
            float k_freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
            float k_angle = (float)k_pos * k_freq;
            k_cos = cosf(k_angle);
            k_sin = sinf(k_angle);
        }

        int k_stride = num_kv_heads * head_dim;
        int base_idx = k_t * k_stride + head * head_dim;

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

        float v0 = __half2float(k[i0]);
        float v1 = __half2float(k[i1]);
        k[i0] = __float2half(v0 * k_cos - v1 * k_sin);
        k[i1] = __float2half(v0 * k_sin + v1 * k_cos);
    }
}
