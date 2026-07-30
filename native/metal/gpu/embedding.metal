// ISO port of embedding.cu.
//
// FP16-output embedding lookups: F32 / F16 / Q8_0 source tables → FP16 output.
// (Metal extension: a Q6_K source variant with no CUDA counterpart is also
//  provided here, since it produces FP16 output like the others.)

#include <metal_stdlib>
using namespace metal;

#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BLOCK_BYTES 34

kernel void embedding_lookup_f32_f16out(
    device const float* embed_table         [[ buffer(0) ]],
    device const int* token_ids             [[ buffer(1) ]],
    device half* output                     [[ buffer(2) ]],
    constant int& seq_len                   [[ buffer(3) ]],
    constant int& hidden_size               [[ buffer(4) ]],

    uint block_id_x                         [[ threadgroup_position_in_grid ]],
    uint threadIdx_x                        [[ thread_position_in_threadgroup ]],
    uint blockDim_x                         [[ threads_per_threadgroup ]],
    uint simd_group_width                   [[ threads_per_simdgroup]])
{
    int t = block_id_x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    device const float* row = embed_table + (size_t)token_id * hidden_size;
    device half* out_row = output + (size_t)t * hidden_size;

    for (int i = threadIdx_x; i < hidden_size; i += blockDim_x)
        out_row[i] = (half)row[i];
}

kernel void embedding_lookup_f16_f16out(
    device const half* embed_table          [[ buffer(0) ]],
    device const int* token_ids             [[ buffer(1) ]],
    device half* output                     [[ buffer(2) ]],
    constant int& seq_len                   [[ buffer(3) ]],
    constant int& hidden_size               [[ buffer(4) ]],

    uint block_id_x                         [[ threadgroup_position_in_grid ]],
    uint threadIdx_x                        [[ thread_position_in_threadgroup ]],
    uint blockDim_x                         [[ threads_per_threadgroup ]],
    uint simd_group_width                   [[ threads_per_simdgroup]])
{
    int t = block_id_x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];
    device const half* row = embed_table + (size_t)token_id * hidden_size;
    device half* out_row = output + (size_t)t * hidden_size;

    for (int i = threadIdx_x; i < hidden_size; i += blockDim_x)
        out_row[i] = row[i];
}

// Q6_K layout per superblock (256 elements, 210 bytes):
//   ql[128]     lower 4 bits of each quant
//   qh[64]      upper 2 bits of each quant
//   scales[16]  int8 scale per group of 16 elements
//   d           fp16 super-block scale
#define Q6_K_SUPER_BLOCK_SIZE 256
#define Q6_K_BLOCK_BYTES      210

kernel void embedding_lookup_q6_k_f16out(
    device const uchar* embed_table [[ buffer(0) ]],
    device const int*   token_ids   [[ buffer(1) ]],
    device half*        output      [[ buffer(2) ]],
    constant int&       seq_len     [[ buffer(3) ]],
    constant int&       hidden_size [[ buffer(4) ]],

    uint block_id_x  [[ threadgroup_position_in_grid ]],
    uint threadIdx_x [[ thread_position_in_threadgroup ]],
    uint blockDim_x  [[ threads_per_threadgroup ]],
    uint simd_group_width [[ threads_per_simdgroup ]])
{
    int t = (int)block_id_x;
    if (t >= seq_len) return;

    int token_id      = token_ids[t];
    int blocks_per_row = hidden_size / Q6_K_SUPER_BLOCK_SIZE;

    ulong row_offset = (ulong)token_id * blocks_per_row * Q6_K_BLOCK_BYTES;
    device const uchar* row     = embed_table + row_offset;
    device half*        out_row = output + (size_t)t * hidden_size;

    // One thread per element within a superblock (threadIdx_x = 0..255).
    // Iterate over superblocks sequentially; threads stride if blockDim_x < 256.
    for (int elem = (int)threadIdx_x; elem < Q6_K_SUPER_BLOCK_SIZE; elem += (int)blockDim_x)
    {
        for (int b = 0; b < blocks_per_row; b++)
        {
            device const uchar* block   = row + b * Q6_K_BLOCK_BYTES;
            device const uchar* ql      = block;           // 128 bytes
            device const uchar* qh_base = block + 128;     // 64 bytes
            device const char*  sc      = (device const char*)(block + 192); // 16 bytes
            float d = float(*(device const half*)(block + 208));

            int half_idx    = elem / 128;
            int pos_in_half = elem % 128;

            device const uchar* ql_half = ql      + half_idx * 64;
            device const uchar* qh_half = qh_base + half_idx * 32;
            device const char*  sc_half = sc       + half_idx * 8;

            int group = pos_in_half / 32;
            int l     = pos_in_half % 32;
            int isc   = l / 16;

            int q_val;
            switch (group) {
                case 0: q_val = ((ql_half[l]      & 0x0F) | (((qh_half[l] >> 0) & 3) << 4)) - 32; break;
                case 1: q_val = ((ql_half[l + 32] & 0x0F) | (((qh_half[l] >> 2) & 3) << 4)) - 32; break;
                case 2: q_val = ((ql_half[l]      >> 4)   | (((qh_half[l] >> 4) & 3) << 4)) - 32; break;
                default:q_val = ((ql_half[l + 32] >> 4)   | (((qh_half[l] >> 6) & 3) << 4)) - 32; break;
            }

            out_row[b * Q6_K_SUPER_BLOCK_SIZE + elem] = half(d * float(sc_half[isc + group * 2]) * float(q_val));
        }
    }
}

kernel void embedding_lookup_q8_0_f16out(
    device const uchar* embed_table         [[ buffer(0) ]],
    device const int* token_ids             [[ buffer(1) ]],
    device half* output                     [[ buffer(2) ]],
    constant int& seq_len                   [[ buffer(3) ]],
    constant int& hidden_size               [[ buffer(4) ]],

    uint block_id_x                         [[ threadgroup_position_in_grid ]],
    uint threadIdx_x                        [[ thread_position_in_threadgroup ]],
    uint blockDim_x                         [[ threads_per_threadgroup ]],
    uint simd_group_width                   [[ threads_per_simdgroup]])
{
    int t = block_id_x;
    if (t >= seq_len) return;

    int token_id = token_ids[t];

    int blocks_per_row = hidden_size / Q8_0_BLOCK_SIZE;

    ulong row_offset = (ulong)token_id * blocks_per_row * Q8_0_BLOCK_BYTES;
    device const uchar* row = embed_table + row_offset;
    device half* out_row = output + (size_t)t * hidden_size;

    for (int b = threadIdx_x; b < blocks_per_row; b += blockDim_x)
    {
        device const uchar* block = row + b * Q8_0_BLOCK_BYTES;
        float d = (float)(*(device const half*)block);
        device const char* qs = (device const char*)(block + 2);

        for (int j = 0; j < Q8_0_BLOCK_SIZE; j++)
            out_row[b * Q8_0_BLOCK_SIZE + j] = (half)(d * (float)qs[j]);
    }
}
