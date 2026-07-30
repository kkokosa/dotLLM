// ISO port of attention.cu.

#include <metal_stdlib>
using namespace metal;

#define TILE_KV 256

[[max_total_threads_per_threadgroup(256)]]
kernel void attention_f16(
    device const half*  q              [[buffer(0)]],
    device const half*  k              [[buffer(1)]],
    device const half*  v              [[buffer(2)]],
    device half*        output         [[buffer(3)]],
    constant int&       seq_q          [[buffer(4)]],
    constant int&       seq_kv         [[buffer(5)]],
    constant int&       num_heads      [[buffer(6)]],
    constant int&       num_kv_heads   [[buffer(7)]],
    constant int&       head_dim       [[buffer(8)]],
    constant int&       position_offset [[buffer(9)]],
    constant int&       sliding_window  [[buffer(10)]],
    threadgroup float*  smem           [[threadgroup(0)]],
    uint block_id   [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_position_in_threadgroup]],
    uint block_dim  [[threads_per_threadgroup]])
{
    int total_blocks = seq_q * num_heads;
    if ((int)block_id >= total_blocks) return;

    int tq  = (int)block_id / num_heads;
    int hq  = (int)block_id % num_heads;

    int group_size = num_heads / num_kv_heads;
    int hkv = hq / group_size;

    float scale = rsqrt((float)head_dim);

    int q_stride  = num_heads    * head_dim;
    int kv_stride = num_kv_heads * head_dim;

    int pos_q = position_offset + tq;

    // Shared memory layout — identical to attention_f32.metal.
    threadgroup float* q_shared     = smem;
    threadgroup float* score_tile   = smem + head_dim;
    threadgroup float* out_accum    = score_tile + TILE_KV;
    threadgroup float* warp_scratch = out_accum + head_dim;

    uint lane    = thread_idx % 32;
    uint warp_id = thread_idx / 32;

    // Step 1: Load Q into shared memory (half → float).
    device const half* q_vec = q + (ulong)tq * q_stride + hq * head_dim;
    for (int d = (int)thread_idx; d < head_dim; d += (int)block_dim)
        q_shared[d] = float(q_vec[d]);

    for (int d = (int)thread_idx; d < head_dim; d += (int)block_dim)
        out_accum[d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Process KV in tiles with online softmax.
    float running_max = -MAXFLOAT;
    float running_sum = 0.0f;

    for (int t_start = 0; t_start < seq_kv; t_start += TILE_KV)
    {
        int t_end  = t_start + TILE_KV;
        if (t_end > seq_kv) t_end = seq_kv;
        int tile_len = t_end - t_start;

        // 2a. Q·K scores for this tile (K: half → float on the fly).
        for (int t = (int)thread_idx; t < tile_len; t += (int)block_dim)
        {
            int tkv = t_start + t;

            if (tkv > pos_q)
            { score_tile[t] = -MAXFLOAT; continue; }

            if (sliding_window > 0 && pos_q - tkv > sliding_window)
            { score_tile[t] = -MAXFLOAT; continue; }

            device const half* k_vec = k + (ulong)tkv * kv_stride + hkv * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
                score += q_shared[d] * float(k_vec[d]);

            score_tile[t] = score * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2b. Tile max reduction (two-pass: intra-warp → cross-warp).
        float tile_max = -MAXFLOAT;
        for (int t = (int)thread_idx; t < tile_len; t += (int)block_dim)
            tile_max = max(tile_max, score_tile[t]);

        tile_max = simd_max(tile_max);
        if (lane == 0) warp_scratch[warp_id] = tile_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (warp_id == 0) {
            uint nw = (block_dim + 31) / 32;
            tile_max = (lane < nw) ? warp_scratch[lane] : -MAXFLOAT;
            tile_max = simd_max(tile_max);
        }
        if (thread_idx == 0) warp_scratch[0] = tile_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_max = warp_scratch[0];

        // 2c. Online softmax rescale.
        float new_max = max(running_max, tile_max);
        float correction = (running_max > -MAXFLOAT + 1.0f)
                           ? exp(running_max - new_max) : 0.0f;

        running_sum *= correction;
        for (int d = (int)thread_idx; d < head_dim; d += (int)block_dim)
            out_accum[d] *= correction;

        running_max = new_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2d. Attention weights: exp(score - max).
        float tile_sum = 0.0f;
        for (int t = (int)thread_idx; t < tile_len; t += (int)block_dim)
        {
            float w = (score_tile[t] > -MAXFLOAT + 1.0f)
                      ? exp(score_tile[t] - running_max) : 0.0f;
            score_tile[t] = w;
            tile_sum += w;
        }

        tile_sum = simd_sum(tile_sum);
        if (lane == 0) warp_scratch[warp_id] = tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (warp_id == 0) {
            uint nw = (block_dim + 31) / 32;
            tile_sum = (lane < nw) ? warp_scratch[lane] : 0.0f;
            tile_sum = simd_sum(tile_sum);
            if (lane == 0) warp_scratch[0] = tile_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        running_sum += warp_scratch[0];

        // 2e. Accumulate weighted V (V: half → float on the fly).
        for (int d = (int)thread_idx; d < head_dim; d += (int)block_dim)
        {
            float v_acc = 0.0f;
            for (int t = 0; t < tile_len; t++)
            {
                if (score_tile[t] > 0.0f)
                {
                    int tkv = t_start + t;
                    device const half* v_vec = v + (ulong)tkv * kv_stride + hkv * head_dim;
                    v_acc += score_tile[t] * float(v_vec[d]);
                }
            }
            out_accum[d] += v_acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 3: Normalize and write output (float → half).
    float sum_inv = (running_sum > 1e-10f) ? (1.0f / running_sum) : 0.0f;

    device half* out_vec = output + (ulong)tq * q_stride + hq * head_dim;
    for (int d = (int)thread_idx; d < head_dim; d += (int)block_dim)
        out_vec[d] = half(out_accum[d] * sum_inv);
}
