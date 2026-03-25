// Naive attention with FP32 Q/K/V/output.

#include <float.h>
#include <math.h>

extern "C" __global__ void attention_f32(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ output,
    const int seq_q, const int seq_kv,
    const int num_heads, const int num_kv_heads, const int head_dim,
    const int position_offset, const int sliding_window)
{
    int block_id = blockIdx.x;
    if (block_id >= seq_q * num_heads) return;

    int tq = block_id / num_heads;
    int hq = block_id % num_heads;
    int hkv = hq / (num_heads / num_kv_heads);
    float scale = rsqrtf((float)head_dim);

    const float* q_vec = q + (size_t)tq * num_heads * head_dim + hq * head_dim;
    int pos_q = position_offset + tq;

    extern __shared__ float smem[];
    float* out_accum = smem + seq_kv;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) out_accum[d] = 0.0f;

    for (int tkv = threadIdx.x; tkv < seq_kv; tkv += blockDim.x)
    {
        if (tkv > pos_q || (sliding_window > 0 && pos_q - tkv > sliding_window))
        { smem[tkv] = -FLT_MAX; continue; }

        const float* k_vec = k + (size_t)tkv * num_kv_heads * head_dim + hkv * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) score += q_vec[d] * k_vec[d];
        smem[tkv] = score * scale;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mx = -FLT_MAX;
        for (int t = 0; t < seq_kv; t++) if (smem[t] > mx) mx = smem[t];
        smem[seq_kv + head_dim] = mx; // stash max after out_accum
    }
    __syncthreads();
    float max_score = smem[seq_kv + head_dim];

    for (int tkv = threadIdx.x; tkv < seq_kv; tkv += blockDim.x)
        smem[tkv] = (smem[tkv] > -FLT_MAX + 1.0f) ? expf(smem[tkv] - max_score) : 0.0f;
    __syncthreads();

    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int t = 0; t < seq_kv; t++) s += smem[t];
        if (s < 1e-10f) s = 1e-10f;
        smem[seq_kv + head_dim] = 1.0f / s;
    }
    __syncthreads();
    float sum_inv = smem[seq_kv + head_dim];

    for (int tkv = threadIdx.x; tkv < seq_kv; tkv += blockDim.x) smem[tkv] *= sum_inv;
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int tkv = 0; tkv < seq_kv; tkv++)
            if (smem[tkv] > 0.0f)
                acc += smem[tkv] * (v + (size_t)tkv * num_kv_heads * head_dim + hkv * head_dim)[d];
        out_accum[d] = acc;
    }
    __syncthreads();

    float* out_vec = output + (size_t)tq * num_heads * head_dim + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) out_vec[d] = out_accum[d];
}
