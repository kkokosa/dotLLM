// Rotary Position Embedding (RoPE) kernel for dotLLM.
// In-place rotation on Q[seqLen, numHeads * headDim] and K[seqLen, numKvHeads * headDim].
// Computes cos/sin from theta in-kernel (no precomputed table upload needed).
// Q and K are processed independently — reuse attempts hurt GQA models due to
// num_heads != num_kv_heads causing different (t, head) decompositions.

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
    // Grid: (seq_len * max(num_heads, num_kv_heads)), one thread per dimension pair
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_rope = rope_dim / 2;
    int total_q_pairs = seq_len * num_heads * half_rope;
    int total_k_pairs = seq_len * num_kv_heads * half_rope;

    // Process Q
    if (idx < total_q_pairs)
    {
        int pair = idx % half_rope;
        int remainder = idx / half_rope;
        int head = remainder % num_heads;
        int t = remainder / num_heads;

        int pos = positions[t];
        float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
        float angle = (float)pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

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

    // Process K (same logic, different head count and stride)
    if (idx < total_k_pairs)
    {
        int pair = idx % half_rope;
        int remainder = idx / half_rope;
        int head = remainder % num_kv_heads;
        int t = remainder / num_kv_heads;

        int pos = positions[t];
        float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
        float angle = (float)pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        int k_stride = num_kv_heads * head_dim;
        int base_idx = t * k_stride + head * head_dim;

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
        k[i0] = __float2half(v0 * cos_val - v1 * sin_val);
        k[i1] = __float2half(v0 * sin_val + v1 * cos_val);
    }
}
