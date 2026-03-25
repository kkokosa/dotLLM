// Embedding lookup with FP32 output for the residual stream.
// output[t] = dequant(embedTable[tokenIds[t]]) as FP32

#include <cuda_fp16.h>
#include <stdint.h>

#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BLOCK_BYTES 34

extern "C" __global__ void embedding_lookup_f32_f32out(
    const float* __restrict__ embed_table,
    const int* __restrict__ token_ids,
    float* __restrict__ output,
    const int seq_len,
    const int hidden_size)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    const float* row = embed_table + (size_t)token_id * hidden_size;
    float* out_row = output + (size_t)t * hidden_size;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        out_row[i] = row[i];
}

extern "C" __global__ void embedding_lookup_f16_f32out(
    const half* __restrict__ embed_table,
    const int* __restrict__ token_ids,
    float* __restrict__ output,
    const int seq_len,
    const int hidden_size)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    const half* row = embed_table + (size_t)token_id * hidden_size;
    float* out_row = output + (size_t)t * hidden_size;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        out_row[i] = __half2float(row[i]);
}

extern "C" __global__ void embedding_lookup_q8_0_f32out(
    const uint8_t* __restrict__ embed_table,
    const int* __restrict__ token_ids,
    float* __restrict__ output,
    const int seq_len,
    const int hidden_size)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    int blocks_per_row = hidden_size / Q8_0_BLOCK_SIZE;
    const uint8_t* row = embed_table + (size_t)token_id * blocks_per_row * Q8_0_BLOCK_BYTES;
    float* out_row = output + (size_t)t * hidden_size;

    for (int b = threadIdx.x; b < blocks_per_row; b += blockDim.x)
    {
        const uint8_t* block = row + b * Q8_0_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);

        for (int j = 0; j < Q8_0_BLOCK_SIZE; j++)
            out_row[b * Q8_0_BLOCK_SIZE + j] = d * (float)qs[j];
    }
}
