// ISO port of embedding_f32out.cu

#include <metal_stdlib>
using namespace metal;

#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BLOCK_BYTES 34

kernel void embedding_lookup_f32_f32out(
    device const float* embed_table         [[ buffer(0) ]],
    device const int* token_ids             [[ buffer(1) ]],
    device float* output                    [[ buffer(2) ]],
    constant int& seq_len                   [[ buffer(3) ]],
    constant int& hidden_size               [[ buffer(4) ]],
    
    uint block_id_x                         [[ threadgroup_position_in_grid ]],   // blockIdx.x
    uint threadIdx_x                        [[ thread_position_in_threadgroup ]], // threadIdx.x
    uint blockDim_x                         [[ threads_per_threadgroup ]],        // blockDim.x
    uint simd_group_width                   [[ threads_per_simdgroup]])
{
    int t = block_id_x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    device const float* row = embed_table + (size_t)token_id * hidden_size;
    device float* out_row = output + (size_t)t * hidden_size;

    for (int i = threadIdx_x; i < hidden_size; i += blockDim_x)
        out_row[i] = row[i];
}

kernel void embedding_lookup_f16_f32out(
    device const half* embed_table          [[ buffer(0) ]],
    device const int* token_ids             [[ buffer(1) ]],
    device float* output                    [[ buffer(2) ]],
    constant int& seq_len                   [[ buffer(3) ]],
    constant int& hidden_size               [[ buffer(4) ]],
    
    uint block_id_x                         [[ threadgroup_position_in_grid ]],   // blockIdx.x
    uint threadIdx_x                        [[ thread_position_in_threadgroup ]], // threadIdx.x
    uint blockDim_x                         [[ threads_per_threadgroup ]],        // blockDim.x
    uint simd_group_width                   [[ threads_per_simdgroup]])
{
    int t = block_id_x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    device const half* row = embed_table + (size_t)token_id * hidden_size;
    device float* out_row = output + (size_t)t * hidden_size;

    for (int i = threadIdx_x; i < hidden_size; i += blockDim_x)
        out_row[i] = float(row[i]);
}

kernel void embedding_lookup_q8_0_f32out(
    device const uchar* embed_table         [[ buffer(0) ]],
    device const int* token_ids             [[ buffer(1) ]],
    device float* output                    [[ buffer(2) ]],
    constant int& seq_len                   [[ buffer(3) ]],
    constant int& hidden_size               [[ buffer(4) ]],
    
    uint block_id_x                         [[ threadgroup_position_in_grid ]],   // blockIdx.x
    uint threadIdx_x                        [[ thread_position_in_threadgroup ]], // threadIdx.x
    uint blockDim_x                         [[ threads_per_threadgroup ]],        // blockDim.x
    uint simd_group_width                   [[ threads_per_simdgroup]])
{
    int t = block_id_x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    
    int blocks_per_row = hidden_size / Q8_0_BLOCK_SIZE;
    
    ulong row_offset = (ulong)token_id * blocks_per_row * Q8_0_BLOCK_BYTES;
    device const uchar* row = embed_table + row_offset;
    device float* out_row = output + (size_t)t * hidden_size;

    for (int b = threadIdx_x; b < blocks_per_row; b += blockDim_x)
    {
        device const uchar* block = row + b * Q8_0_BLOCK_BYTES;
        float d = (float)(*(device const half*)block);
        device const char* qs = (device const char*)(block + 2);

        for (int j = 0; j < Q8_0_BLOCK_SIZE; j++)
            out_row[b * Q8_0_BLOCK_SIZE + j] = d * (float)qs[j];
    }
}
