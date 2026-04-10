// Tiled attention kernel for dotLLM with online softmax.
// Q[seqQ, numHeads * headDim], K[seqKv, numKvHeads * headDim], V same as K.
// All FP16 data, FP32 accumulation for numerical stability.
// GQA: KV head broadcast via group_size = num_heads / num_kv_heads.
// Causal masking + optional sliding window.
// One block per (query_token, query_head) pair.
//
// Key optimizations over naive implementation:
// 1. Q loaded into shared memory once (not re-read per KV position)
// 2. Parallel warp-shuffle reductions (not serial thread-0 scan)
// 3. Tiled online softmax — bounded shared memory O(TILE_KV + headDim)
//    regardless of sequence length (no crash at long contexts)

#include <cuda_fp16.h>
#include <float.h>

#define TILE_KV 256

extern "C" __global__ void __launch_bounds__(256) attention_f16(
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

    // Absolute position for causal masking
    int pos_q = position_offset + tq;

    // Shared memory layout (fixed size, independent of seq_kv):
    //   q_shared[head_dim]     — Q vector cached for reuse
    //   score_tile[TILE_KV]    — attention scores for current tile
    //   out_accum[head_dim]    — weighted V accumulator
    //   warp_scratch[32]       — reduction workspace
    extern __shared__ float smem[];
    float* q_shared    = smem;
    float* score_tile  = smem + head_dim;
    float* out_accum   = score_tile + TILE_KV;
    float* warp_scratch = out_accum + head_dim;

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // Step 1: Load Q vector into shared memory (FP16 → FP32)
    const half* q_vec = q + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        q_shared[d] = __half2float(q_vec[d]);

    // Initialize output accumulator
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_accum[d] = 0.0f;
    __syncthreads();

    // Step 2: Process KV in tiles with online softmax
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    for (int t_start = 0; t_start < seq_kv; t_start += TILE_KV)
    {
        int t_end = t_start + TILE_KV;
        if (t_end > seq_kv) t_end = seq_kv;
        int tile_len = t_end - t_start;

        // 2a. Compute Q·K scores for this tile
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            int tkv = t_start + t;

            // Causal mask: only attend to positions <= current query position
            if (tkv > pos_q)
            {
                score_tile[t] = -FLT_MAX;
                continue;
            }

            // Sliding window: skip if outside window
            if (sliding_window > 0 && pos_q - tkv > sliding_window)
            {
                score_tile[t] = -FLT_MAX;
                continue;
            }

            const half* k_vec = k + (size_t)tkv * kv_stride + hkv * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
                score += q_shared[d] * __half2float(k_vec[d]);

            score_tile[t] = score * scale;
        }
        __syncthreads();

        // 2b. Find tile max via parallel warp-shuffle reduction
        float tile_max = -FLT_MAX;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
            tile_max = fmaxf(tile_max, score_tile[t]);

        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));

        if (lane == 0) warp_scratch[warp_id] = tile_max;
        __syncthreads();

        if (warp_id == 0)
        {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_max = (lane < nw) ? warp_scratch[lane] : -FLT_MAX;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));
        }
        if (threadIdx.x == 0) warp_scratch[0] = tile_max;
        __syncthreads();
        tile_max = warp_scratch[0];

        // 2c. Online softmax: rescale running accumulators
        float new_max = fmaxf(running_max, tile_max);
        float correction = (running_max > -FLT_MAX + 1.0f)
                           ? expf(running_max - new_max) : 0.0f;

        running_sum *= correction;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out_accum[d] *= correction;

        running_max = new_max;
        __syncthreads();

        // 2d. Compute attention weights: exp(score - max)
        float tile_sum = 0.0f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            float w = (score_tile[t] > -FLT_MAX + 1.0f)
                      ? expf(score_tile[t] - running_max) : 0.0f;
            score_tile[t] = w;
            tile_sum += w;
        }

        // Reduce tile_sum
        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
        if (lane == 0) warp_scratch[warp_id] = tile_sum;
        __syncthreads();

        if (warp_id == 0)
        {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_sum = (lane < nw) ? warp_scratch[lane] : 0.0f;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
            if (lane == 0) warp_scratch[0] = tile_sum;
        }
        __syncthreads();
        running_sum += warp_scratch[0];

        // 2e. Accumulate weighted V
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        {
            float v_acc = 0.0f;
            for (int t = 0; t < tile_len; t++)
            {
                if (score_tile[t] > 0.0f)
                {
                    int tkv = t_start + t;
                    const half* v_vec = v + (size_t)tkv * kv_stride + hkv * head_dim;
                    v_acc += score_tile[t] * __half2float(v_vec[d]);
                }
            }
            out_accum[d] += v_acc;
        }
        __syncthreads();
    }

    // Step 3: Final normalize and write output
    float sum_inv = (running_sum > 1e-10f) ? (1.0f / running_sum) : 0.0f;

    half* out_vec = output + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_vec[d] = __float2half(out_accum[d] * sum_inv);
}
