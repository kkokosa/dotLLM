// Embedding lookup kernel for dotLLM.
// output[t] = embedTable[tokenIds[t]]
// Supports F32, F16, Q8_0 source embedding tables → FP16 output.

#include <cuda_fp16.h>
#include <stdint.h>

// Q8_0 block: 2 bytes (half scale) + 32 bytes (int8 values) = 34 bytes, 32 elements
#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BLOCK_BYTES 34

extern "C" __global__ void embedding_lookup_f32(
    const float* __restrict__ embed_table,
    const int* __restrict__ token_ids,
    half* __restrict__ output,
    const int seq_len,
    const int hidden_size)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    const float* row = embed_table + (size_t)token_id * hidden_size;
    half* out_row = output + (size_t)t * hidden_size;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        out_row[i] = __float2half(row[i]);
}

extern "C" __global__ void embedding_lookup_f16(
    const half* __restrict__ embed_table,
    const int* __restrict__ token_ids,
    half* __restrict__ output,
    const int seq_len,
    const int hidden_size)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    const half* row = embed_table + (size_t)token_id * hidden_size;
    half* out_row = output + (size_t)t * hidden_size;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        out_row[i] = row[i];
}

extern "C" __global__ void embedding_lookup_q8_0(
    const uint8_t* __restrict__ embed_table,
    const int* __restrict__ token_ids,
    half* __restrict__ output,
    const int seq_len,
    const int hidden_size)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    int blocks_per_row = hidden_size / Q8_0_BLOCK_SIZE;
    const uint8_t* row = embed_table + (size_t)token_id * blocks_per_row * Q8_0_BLOCK_BYTES;
    half* out_row = output + (size_t)t * hidden_size;

    for (int b = threadIdx.x; b < blocks_per_row; b += blockDim.x)
    {
        const uint8_t* block = row + b * Q8_0_BLOCK_BYTES;
        // First 2 bytes: half scale
        half scale = *reinterpret_cast<const half*>(block);
        float d = __half2float(scale);
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);

        for (int j = 0; j < Q8_0_BLOCK_SIZE; j++)
            out_row[b * Q8_0_BLOCK_SIZE + j] = __float2half(d * (float)qs[j]);
    }
}
