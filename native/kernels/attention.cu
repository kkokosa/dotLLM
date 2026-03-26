// Naive scaled dot-product attention kernel for dotLLM.
// Q[seqQ, numHeads * headDim], K[seqKv, numKvHeads * headDim], V same as K.
// All FP16. Output[seqQ, numHeads * headDim].
// GQA: KV head broadcast via group_size = num_heads / num_kv_heads.
// Causal masking + optional sliding window.
// One block per (query_token, query_head) pair.
// FP16 data, FP32 accumulation for numerical stability.

#include <cuda_fp16.h>
#include <float.h>

extern "C" __global__ void attention_f16(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ output,
    const int seq_q,
    const int seq_kv,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int position_offset,
    const int sliding_window)  // 0 = no sliding window
{
    // Block (query_token, query_head) = blockIdx.x
    int block_id = blockIdx.x;
    int total_blocks = seq_q * num_heads;
    if (block_id >= total_blocks) return;

    int tq = block_id / num_heads;
    int hq = block_id % num_heads;

    int group_size = num_heads / num_kv_heads;
    int hkv = hq / group_size;

    float scale = rsqrtf((float)head_dim);

    int q_stride = num_heads * head_dim;
    int kv_stride = num_kv_heads * head_dim;

    const half* q_vec = q + (size_t)tq * q_stride + hq * head_dim;

    // Absolute position for causal masking
    int pos_q = position_offset + tq;

    // Pass 1: compute scores + online softmax (find max and sum)
    // Using shared memory for score accumulation
    extern __shared__ float smem[];
    // smem layout: [0..seq_kv) = scores, then [seq_kv..seq_kv+head_dim) = output accumulator

    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;

    // Initialize output accumulator to zero
    float* out_accum = smem + seq_kv;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_accum[d] = 0.0f;

    // Compute attention scores and accumulate
    for (int tkv = threadIdx.x; tkv < seq_kv; tkv += blockDim.x)
    {
        // Causal mask: only attend to positions <= current query position
        if (tkv > pos_q)
        {
            smem[tkv] = -FLT_MAX;
            continue;
        }

        // Sliding window: skip if outside window
        if (sliding_window > 0 && pos_q - tkv > sliding_window)
        {
            smem[tkv] = -FLT_MAX;
            continue;
        }

        const half* k_vec = k + (size_t)tkv * kv_stride + hkv * head_dim;

        // Dot product Q . K
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += __half2float(q_vec[d]) * __half2float(k_vec[d]);

        smem[tkv] = score * scale;
    }
    __syncthreads();

    // Find max (serial for simplicity in naive implementation)
    if (threadIdx.x == 0)
    {
        for (int tkv = 0; tkv < seq_kv; tkv++)
        {
            if (smem[tkv] > max_score)
                max_score = smem[tkv];
        }
    }
    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = max_score;
    __syncthreads();
    max_score = shared_max;

    // Compute exp(score - max) and sum
    for (int tkv = threadIdx.x; tkv < seq_kv; tkv += blockDim.x)
    {
        if (smem[tkv] > -FLT_MAX + 1.0f)
            smem[tkv] = expf(smem[tkv] - max_score);
        else
            smem[tkv] = 0.0f;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        sum_exp = 0.0f;
        for (int tkv = 0; tkv < seq_kv; tkv++)
            sum_exp += smem[tkv];
        if (sum_exp < 1e-10f) sum_exp = 1e-10f;
    }
    __shared__ float shared_sum_inv;
    if (threadIdx.x == 0) shared_sum_inv = 1.0f / sum_exp;
    __syncthreads();

    // Normalize weights
    for (int tkv = threadIdx.x; tkv < seq_kv; tkv += blockDim.x)
        smem[tkv] *= shared_sum_inv;
    __syncthreads();

    // Weighted sum of V
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int tkv = 0; tkv < seq_kv; tkv++)
        {
            if (smem[tkv] > 0.0f)
            {
                const half* v_vec = v + (size_t)tkv * kv_stride + hkv * head_dim;
                acc += smem[tkv] * __half2float(v_vec[d]);
            }
        }
        out_accum[d] = acc;
    }
    __syncthreads();

    // Write output
    half* out_vec = output + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_vec[d] = __float2half(out_accum[d]);
}
