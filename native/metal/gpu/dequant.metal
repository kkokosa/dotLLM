// ISO port of dequant.cu.

#include <metal_stdlib>
using namespace metal;

#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BLOCK_BYTES 34

kernel void dequant_q8_0_f16(
    device const uchar* src                 [[ buffer(0) ]],
    device half* dst                        [[ buffer(1) ]],
    constant int& total_blocks              [[ buffer(2) ]],
    
    uint block_id_x                         [[ threadgroup_position_in_grid ]],   // blockIdx.x
    uint thread_idx_x                       [[ thread_position_in_threadgroup ]], // threadIdx.x
    uint block_dim_x                        [[ threads_per_threadgroup ]],        // blockDim.x
    uint simd_group_width                   [[ threads_per_simdgroup]],
    uint grid_dim_x                         [[ threadgroups_per_grid]]
    )
{
    int lane = thread_idx_x % Q8_0_BLOCK_SIZE;
    int warp_in_block = thread_idx_x / Q8_0_BLOCK_SIZE;
    int warps_per_grid = (grid_dim_x * block_dim_x) / Q8_0_BLOCK_SIZE;
    int start_block = block_id_x * (block_dim_x / Q8_0_BLOCK_SIZE) + warp_in_block;

    for (int block_idx = start_block; block_idx < total_blocks; block_idx += warps_per_grid)
    {
        device const uchar* block = src + (size_t)block_idx * Q8_0_BLOCK_BYTES;
        float d = float(*(device const half*)block);
        device const char* qs = (device const char*)(block + 2);
        char q = qs[lane];

        dst[(size_t)block_idx * Q8_0_BLOCK_SIZE + lane] = half(d * (float(q)));
    }
}

#define Q4_0_BLOCK_SIZE 32
#define Q4_0_BLOCK_BYTES 18

kernel void dequant_q4_0_f16(
    device const uchar* src                 [[ buffer(0) ]],
    device half* dst                        [[ buffer(1) ]],
    constant int& total_blocks              [[ buffer(2) ]],
    
    uint block_id_x                         [[ threadgroup_position_in_grid ]],   // blockIdx.x
    uint thread_idx_x                       [[ thread_position_in_threadgroup ]], // threadIdx.x
    uint block_dim_x                        [[ threads_per_threadgroup ]],        // blockDim.x
    uint simd_group_width                   [[ threads_per_simdgroup]],
    uint grid_dim_x                         [[ threadgroups_per_grid]]
    )
{
    int lane = thread_idx_x % Q4_0_BLOCK_SIZE;
    int warp_in_block = thread_idx_x / Q4_0_BLOCK_SIZE;
    int warps_per_grid = (grid_dim_x * block_dim_x) / Q4_0_BLOCK_SIZE;
    int start_block = block_id_x * (block_dim_x / Q4_0_BLOCK_SIZE) + warp_in_block;

    for (int block_idx = start_block; block_idx < total_blocks; block_idx += warps_per_grid)
    {
        device const uchar* block = src + (size_t)block_idx * Q4_0_BLOCK_BYTES;
        float d = float(*(device const half*)block);
        device const char* qs = (device const char*)(block + 2);
        
        // Elements interleave: out[2j]=lo(qs[j]), out[2j+1]=hi(qs[j])
        int byte_idx = lane / 2;
        uchar packed = qs[byte_idx];
        int val = (lane & 1) ? ((int)(packed >> 4) - 8) : ((int)(packed & 0x0F) - 8);

        dst[(size_t)block_idx * Q4_0_BLOCK_SIZE + lane] = half(d * (float)val);
    }
}

#define Q5_0_BLOCK_SIZE 32
#define Q5_0_BLOCK_BYTES 22

kernel void dequant_q5_0_f16(
    device const uchar* src                 [[ buffer(0) ]],
    device half* dst                        [[ buffer(1) ]],
    constant int& total_blocks              [[ buffer(2) ]],
    
    uint block_id_x                         [[ threadgroup_position_in_grid ]],   // blockIdx.x
    uint thread_idx_x                       [[ thread_position_in_threadgroup ]], // threadIdx.x
    uint block_dim_x                        [[ threads_per_threadgroup ]],        // blockDim.x
    uint simd_group_width                   [[ threads_per_simdgroup]],
    uint grid_dim_x                         [[ threadgroups_per_grid]]
    )
{
    int lane = thread_idx_x % Q5_0_BLOCK_SIZE;
    int warp_in_block = thread_idx_x / Q5_0_BLOCK_SIZE;
    int warps_per_grid = (grid_dim_x * block_dim_x) / Q5_0_BLOCK_SIZE;
    int start_block = block_id_x * (block_dim_x / Q5_0_BLOCK_SIZE) + warp_in_block;

    for (int block_idx = start_block; block_idx < total_blocks; block_idx += warps_per_grid)
    {
        device const uchar* block = src + (size_t)block_idx * Q5_0_BLOCK_BYTES;
        float d = float(*(device const half*)block);
        // Read qh as 4 bytes (may be unaligned)
        unsigned int qh = (unsigned int)block[2] | ((unsigned int)block[3] << 8) |
                          ((unsigned int)block[4] << 16) | ((unsigned int)block[5] << 24);
        device const uchar* qs = block + 6;

        // lane 0..15 → low nibbles (elements 0..15)
        // lane 16..31 → high nibbles (elements 16..31)
        int j = lane < 16 ? lane : lane - 16;
        uint8_t packed = qs[j];
        int nibble = (lane < 16) ? (packed & 0x0F) : (packed >> 4);
        int high_bit = (qh >> lane) & 1;
        int val = (nibble | (high_bit << 4)) - 16;

        dst[(size_t)block_idx * Q5_0_BLOCK_SIZE + lane] = half(d * (float)val);
    }
}

#define Q4_K_SUPER_BLOCK_SIZE 256
#define Q4_K_BLOCK_BYTES 144

kernel void dequant_q4_k_f16(
    device const uchar* src                 [[ buffer(0) ]],
    device half* dst                        [[ buffer(1) ]],
    constant int& total_superblocks         [[ buffer(2) ]],
    
    uint block_id_x                         [[ threadgroup_position_in_grid ]],   // blockIdx.x
    uint thread_idx_x                       [[ thread_position_in_threadgroup ]], // threadIdx.x
    uint block_dim_x                        [[ threads_per_threadgroup ]],        // blockDim.x
    uint simd_group_width                   [[ threads_per_simdgroup]],
    uint grid_dim_x                         [[ threadgroups_per_grid]]
    )
{
    // One block per superblock, 256 threads = one thread per output element.
    // Grid-stride loop over superblocks.
    int t = thread_idx_x; // 0..255

    for (int sb_idx = block_id_x; sb_idx < total_superblocks; sb_idx += grid_dim_x)
    {
        device const uchar* block = src + (size_t)sb_idx * Q4_K_BLOCK_BYTES;
        float d = float(*(device const half*)block);
        float dmin = float(*(device const half*)(block + 2));
        device const uchar* scales_raw = block + 4;
        device const uchar* qs = block + 16;

        // Determine which pair (0..3) and position within pair (0..63)
        int pair = t / 64;
        int pos_in_pair = t % 64;
        // Within pair: 0..31 → even sub-block (low nibble), 32..63 → odd sub-block (high nibble)
        int is_odd = pos_in_pair / 32;
        int j = pos_in_pair % 32;

        int sb_even = pair * 2;
        int sb_odd = pair * 2 + 1;
        int sb_cur = is_odd ? sb_odd : sb_even;

        // Decode 6-bit scale and min for this sub-block
        int sc, m;
        if (sb_cur < 4)
        {
            sc = scales_raw[sb_cur] & 0x3F;
            m = scales_raw[sb_cur + 4] & 0x3F;
        }
        else
        {
            sc = (scales_raw[sb_cur + 4] & 0x0F) | ((scales_raw[sb_cur - 4] >> 6) << 4);
            m = (scales_raw[sb_cur + 4] >> 4) | ((scales_raw[sb_cur] >> 6) << 4);
        }

        uint8_t byte_val = qs[pair * 32 + j];
        int nibble = is_odd ? (byte_val >> 4) : (byte_val & 0x0F);

        float result = d * (float)sc * (float)nibble - dmin * (float)m;
        dst[(size_t)sb_idx * Q4_K_SUPER_BLOCK_SIZE + t] = half(result);
    }
}

#define Q5_K_SUPER_BLOCK_SIZE 256
#define Q5_K_BLOCK_BYTES 176

kernel void dequant_q5_k_f16(
    device const uchar* src                 [[ buffer(0) ]],
    device half* dst                        [[ buffer(1) ]],
    constant int& total_superblocks         [[ buffer(2) ]],
    
    uint block_id_x                         [[ threadgroup_position_in_grid ]],   // blockIdx.x
    uint thread_idx_x                       [[ thread_position_in_threadgroup ]], // threadIdx.x
    uint block_dim_x                        [[ threads_per_threadgroup ]],        // blockDim.x
    uint simd_group_width                   [[ threads_per_simdgroup]],
    uint grid_dim_x                         [[ threadgroups_per_grid]]
    )
{
    int t = thread_idx_x; // 0..255

    for (int sb_idx = block_id_x; sb_idx < total_superblocks; sb_idx += grid_dim_x)
    {
        device const uchar* block = src + (size_t)sb_idx * Q5_K_BLOCK_BYTES;
        float d = float(*(device const half*)block);
        float dmin = float(*(device const half*)(block + 2));
        device const uchar* scales_raw = block + 4;
        device const uchar* qh = block + 16;   // 32 bytes
        device const uchar* qs = block + 48;   // 128 bytes

        // Which sub-block (0..7) and position within sub-block (0..31)
        int sub = t / 32;
        int pos = t % 32;

        // Decode 6-bit scale and min
        int sc, m;
        if (sub < 4)
        {
            sc = scales_raw[sub] & 0x3F;
            m = scales_raw[sub + 4] & 0x3F;
        }
        else
        {
            sc = (scales_raw[sub + 4] & 0x0F) | ((scales_raw[sub - 4] >> 6) << 4);
            m = (scales_raw[sub + 4] >> 4) | ((scales_raw[sub] >> 6) << 4);
        }

        float scale = d * (float)sc;
        float min_val = dmin * (float)m;

        // NOTE: this qs/qh access pattern is a direct port of the CUDA kernel.
        // It does NOT match the CPU scalar reference (DequantizeQ5_KScalar).
        // A potential bug exists in the original CUDA implementation and will be fixed
        // for both backends (CUDA + Metal) in a dedicated follow-up PR.
        device const uchar* sub_qs = qs + sub * 16;
        device const uchar* sub_qh = qh + sub * 4;

        int j = pos / 2;
        uint8_t packed = sub_qs[j];
        int nibble = (pos & 1) ? (packed >> 4) : (packed & 0x0F);

        int bit = (sub_qh[j / 4] >> ((j % 4) * 2 + (pos & 1))) & 1;
        int val = nibble | (bit << 4);

        dst[(size_t)sb_idx * Q5_K_SUPER_BLOCK_SIZE + t] = half(scale * (float)val - min_val);
    }
}

#define Q6_K_SUPER_BLOCK_SIZE 256
#define Q6_K_BLOCK_BYTES 210


kernel void dequant_q6_k_f16(
    device const uchar* src                 [[ buffer(0) ]],
    device half* dst                        [[ buffer(1) ]],
    constant int& total_superblocks         [[ buffer(2) ]],
    
    uint block_id_x                         [[ threadgroup_position_in_grid ]],   // blockIdx.x
    uint thread_idx_x                       [[ thread_position_in_threadgroup ]], // threadIdx.x
    uint block_dim_x                        [[ threads_per_threadgroup ]],        // blockDim.x
    uint simd_group_width                   [[ threads_per_simdgroup]],
    uint grid_dim_x                         [[ threadgroups_per_grid]]
    )
{
    int t = thread_idx_x; // 0..255

    for (int sb_idx = block_id_x; sb_idx < total_superblocks; sb_idx += grid_dim_x)
    {
        device const uchar* block = src + (size_t)sb_idx * Q6_K_BLOCK_BYTES;
        device const uchar* ql = block;           // 128 bytes
        device const uchar* qh_base = block + 128;     // 64 bytes
        
        device const char* scales = (device const char*)(block + 192); // 16 bytes
        float d = float(*(device const half*)(block + 208));
        
        // Two 128-element halves (t<128 → first half, t>=128 → second half)
        int half_idx = t / 128;
        int pos_in_half = t % 128;

        device const uchar* ql_half = ql + half_idx * 64;
        device const uchar* qh_half = qh_base + half_idx * 32;
        device const char* sc_half = (device const char*)scales + half_idx * 8;

        // Within each half (128 elements): 4 groups of 32
        int group = pos_in_half / 32;
        int l = pos_in_half % 32;
        int isc = l / 16;

        int q_val;
        switch (group)
        {
            case 0:
                q_val = ((ql_half[l] & 0x0F) | (((qh_half[l] >> 0) & 3) << 4)) - 32;
                break;
            case 1:
                q_val = ((ql_half[l + 32] & 0x0F) | (((qh_half[l] >> 2) & 3) << 4)) - 32;
                break;
            case 2:
                q_val = ((ql_half[l] >> 4) | (((qh_half[l] >> 4) & 3) << 4)) - 32;
                break;
            default: // case 3
                q_val = ((ql_half[l + 32] >> 4) | (((qh_half[l] >> 6) & 3) << 4)) - 32;
                break;
        }

        float sc = d * (float)sc_half[isc + group * 2];
        dst[(size_t)sb_idx * Q6_K_SUPER_BLOCK_SIZE + t] = half(sc * (float)q_val);
    }
}
