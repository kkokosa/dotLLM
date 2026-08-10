// Dequantization kernels for dotLLM.
// Convert quantized weight blocks to FP16.
// Block layouts match docs/QUANTIZATION.md.
//
// Coalesced write pattern:
// - Q8_0/Q4_0/Q5_0: one warp (32 threads) per quant block, each thread writes one element.
// - Q4_K/Q5_K/Q6_K: one CUDA block (256 threads) per superblock, each thread writes one element.
//
// Grid-stride loops: host caps grid size to O(num_SMs), each block processes
// multiple quant blocks / superblocks. Amortizes launch overhead on GPUs with
// many SMs (e.g. RTX 3050) where launching hundreds of thousands of blocks
// overwhelms the hardware block scheduler.

#include <cuda_fp16.h>
#include <stdint.h>

// ── Q8_0: 34 bytes per 32 values ────────────────────────────────────
// struct block_q8_0 { half d; int8_t qs[32]; };
#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BLOCK_BYTES 34

extern "C" __global__ void __launch_bounds__(256) dequant_q8_0_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_blocks)
{
    // One warp per quant block: thread lane writes one element.
    // Grid-stride loop over blocks.
    int lane = threadIdx.x % Q8_0_BLOCK_SIZE;
    int warp_in_block = threadIdx.x / Q8_0_BLOCK_SIZE;
    int warps_per_grid = (gridDim.x * blockDim.x) / Q8_0_BLOCK_SIZE;
    int start_block = blockIdx.x * (blockDim.x / Q8_0_BLOCK_SIZE) + warp_in_block;

    for (int block_idx = start_block; block_idx < total_blocks; block_idx += warps_per_grid)
    {
        const uint8_t* block = src + (size_t)block_idx * Q8_0_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        int8_t q = reinterpret_cast<const int8_t*>(block + 2)[lane];

        dst[(size_t)block_idx * Q8_0_BLOCK_SIZE + lane] = __float2half(d * (float)q);
    }
}

// ── Q4_0: 18 bytes per 32 values ────────────────────────────────────
// struct block_q4_0 { half d; uint8_t qs[16]; };
#define Q4_0_BLOCK_SIZE 32
#define Q4_0_BLOCK_BYTES 18

extern "C" __global__ void __launch_bounds__(256) dequant_q4_0_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_blocks)
{
    int lane = threadIdx.x % Q4_0_BLOCK_SIZE;
    int warp_in_block = threadIdx.x / Q4_0_BLOCK_SIZE;
    int warps_per_grid = (gridDim.x * blockDim.x) / Q4_0_BLOCK_SIZE;
    int start_block = blockIdx.x * (blockDim.x / Q4_0_BLOCK_SIZE) + warp_in_block;

    for (int block_idx = start_block; block_idx < total_blocks; block_idx += warps_per_grid)
    {
        const uint8_t* block = src + (size_t)block_idx * Q4_0_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        const uint8_t* qs = block + 2;

        // Elements interleave: out[2j]=lo(qs[j]), out[2j+1]=hi(qs[j])
        int byte_idx = lane / 2;
        uint8_t packed = qs[byte_idx];
        int val = (lane & 1) ? ((int)(packed >> 4) - 8) : ((int)(packed & 0x0F) - 8);

        dst[(size_t)block_idx * Q4_0_BLOCK_SIZE + lane] = __float2half(d * (float)val);
    }
}

// ── Q5_0: 22 bytes per 32 values ────────────────────────────────────
// struct block_q5_0 { half d; uint32_t qh; uint8_t qs[16]; };
#define Q5_0_BLOCK_SIZE 32
#define Q5_0_BLOCK_BYTES 22

extern "C" __global__ void __launch_bounds__(256) dequant_q5_0_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_blocks)
{
    int lane = threadIdx.x % Q5_0_BLOCK_SIZE;
    int warp_in_block = threadIdx.x / Q5_0_BLOCK_SIZE;
    int warps_per_grid = (gridDim.x * blockDim.x) / Q5_0_BLOCK_SIZE;
    int start_block = blockIdx.x * (blockDim.x / Q5_0_BLOCK_SIZE) + warp_in_block;

    for (int block_idx = start_block; block_idx < total_blocks; block_idx += warps_per_grid)
    {
        const uint8_t* block = src + (size_t)block_idx * Q5_0_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        // Read qh as 4 bytes (may be unaligned)
        unsigned int qh = (unsigned int)block[2] | ((unsigned int)block[3] << 8) |
                          ((unsigned int)block[4] << 16) | ((unsigned int)block[5] << 24);
        const uint8_t* qs = block + 6;

        // lane 0..15 → low nibbles (elements 0..15)
        // lane 16..31 → high nibbles (elements 16..31)
        int j = lane < 16 ? lane : lane - 16;
        uint8_t packed = qs[j];
        int nibble = (lane < 16) ? (packed & 0x0F) : (packed >> 4);
        int high_bit = (qh >> lane) & 1;
        int val = (nibble | (high_bit << 4)) - 16;

        dst[(size_t)block_idx * Q5_0_BLOCK_SIZE + lane] = __float2half(d * (float)val);
    }
}

// ── Q4_1: 20 bytes per 32 values ────────────────────────────────────
// struct block_q4_1 { half d; half m; uint8_t qs[16]; };
// value = d * nibble + m  (unsigned + min, vs Q4_0's signed-after-bias-of-8)
#define Q4_1_BLOCK_SIZE 32
#define Q4_1_BLOCK_BYTES 20

extern "C" __global__ void __launch_bounds__(256) dequant_q4_1_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_blocks)
{
    int lane = threadIdx.x % Q4_1_BLOCK_SIZE;
    int warp_in_block = threadIdx.x / Q4_1_BLOCK_SIZE;
    int warps_per_grid = (gridDim.x * blockDim.x) / Q4_1_BLOCK_SIZE;
    int start_block = blockIdx.x * (blockDim.x / Q4_1_BLOCK_SIZE) + warp_in_block;

    for (int block_idx = start_block; block_idx < total_blocks; block_idx += warps_per_grid)
    {
        const uint8_t* block = src + (size_t)block_idx * Q4_1_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        float m = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* qs = block + 4;

        int byte_idx = lane / 2;
        uint8_t packed = qs[byte_idx];
        int val = (lane & 1) ? (int)(packed >> 4) : (int)(packed & 0x0F);

        dst[(size_t)block_idx * Q4_1_BLOCK_SIZE + lane] = __float2half(d * (float)val + m);
    }
}

// ── Q5_1: 24 bytes per 32 values ────────────────────────────────────
// struct block_q5_1 { half d; half m; uint32_t qh; uint8_t qs[16]; };
// value = d * ((qh_bit << 4) | nibble) + m  (5-bit unsigned + min)
#define Q5_1_BLOCK_SIZE 32
#define Q5_1_BLOCK_BYTES 24

extern "C" __global__ void __launch_bounds__(256) dequant_q5_1_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_blocks)
{
    int lane = threadIdx.x % Q5_1_BLOCK_SIZE;
    int warp_in_block = threadIdx.x / Q5_1_BLOCK_SIZE;
    int warps_per_grid = (gridDim.x * blockDim.x) / Q5_1_BLOCK_SIZE;
    int start_block = blockIdx.x * (blockDim.x / Q5_1_BLOCK_SIZE) + warp_in_block;

    for (int block_idx = start_block; block_idx < total_blocks; block_idx += warps_per_grid)
    {
        const uint8_t* block = src + (size_t)block_idx * Q5_1_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        float m = __half2float(*reinterpret_cast<const half*>(block + 2));
        // Read qh (4 bytes, may be unaligned).
        unsigned int qh = (unsigned int)block[4] | ((unsigned int)block[5] << 8) |
                          ((unsigned int)block[6] << 16) | ((unsigned int)block[7] << 24);
        const uint8_t* qs = block + 8;

        int j = lane < 16 ? lane : lane - 16;
        uint8_t packed = qs[j];
        int nibble = (lane < 16) ? (packed & 0x0F) : (packed >> 4);
        int high_bit = (qh >> lane) & 1;
        int val = nibble | (high_bit << 4);

        dst[(size_t)block_idx * Q5_1_BLOCK_SIZE + lane] = __float2half(d * (float)val + m);
    }
}

// ── Q3_K: 110 bytes per 256 values (super-block with 16 sub-blocks of 16) ──
// struct block_q3_K { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; half d; };
//   hmask: 1 high bit per element (32 × 8 = 256 bits)
//   qs:    2 low bits per element (64 × 4 = 256)
//   scales: 16 × 6-bit signed-after-bias-of-32 packed into 12 bytes
//   d: FP16 super-block delta
// Per-element value: d × (signed_scale[sub]) × (((hmask_bit << 2) | qs_bits) - 4)
#define Q3_K_SUPER_BLOCK_SIZE 256
#define Q3_K_BLOCK_BYTES 110

extern "C" __global__ void __launch_bounds__(256) dequant_q3_k_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_superblocks)
{
    int t = threadIdx.x; // 0..255

    for (int sb_idx = blockIdx.x; sb_idx < total_superblocks; sb_idx += gridDim.x)
    {
        const uint8_t* block = src + (size_t)sb_idx * Q3_K_BLOCK_BYTES;
        const uint8_t* hmask = block;                // 32 bytes
        const uint8_t* qs = block + 32;              // 64 bytes
        const uint8_t* scales12 = block + 32 + 64;   // 12 bytes
        float d = __half2float(*reinterpret_cast<const half*>(block + 32 + 64 + 12));

        // Sub-block index (0..15) for this thread; 16 threads share a sub-block
        // (= 16 elements per sub-block).
        int sub = t / 16;

        // Unpack 6-bit scale for this sub-block from the 12 packed bytes.
        // Per llama.cpp ggml-quants.c dequantize_row_q3_K:
        //   sub 0..7  low nibble = scales12[sub] low nibble
        //   sub 8..15 low nibble = scales12[sub-8] high nibble (NOT sub-4 — that
        //   collides with the high-2-bits packing in scales12[8..11]).
        //   high 2 bits = scales12[8 + (sub % 4)] >> ((sub / 4) * 2)
        // The byte/shift pair is TRANSPOSED relative to the obvious-looking
        // 8 + sub/4 @ (sub%4)*2 — the wrong way round scrambles 12 of 16 scales.
        int lowSrcByte = sub < 8 ? sub : sub - 8;
        int lowNibble = sub < 8
            ? (scales12[lowSrcByte] & 0x0F)
            : ((scales12[lowSrcByte] >> 4) & 0x0F);
        int hiBits = (scales12[8 + (sub & 3)] >> ((sub >> 2) * 2)) & 0x03;
        int signedScale = (lowNibble | (hiBits << 4)) - 32;  // [-32, 31]

        // Per-element 3-bit unpacking. The 2-bit quants are NOT stored four
        // consecutive elements per byte: element t reads bit-pair (t/32)%4 of
        // qs byte (t%32) + 32*(t/128), and hmask bit t/32 of byte t%32
        // (llama.cpp's shift/m loop over 128-element halves).
        int qBits = (qs[(t & 31) + 32 * (t >> 7)] >> (((t >> 5) & 3) * 2)) & 0x03;
        int hBit = (hmask[t & 31] >> (t >> 5)) & 0x01;
        int signed3 = ((hBit << 2) | qBits) - 4;  // [-4, 3]

        dst[(size_t)sb_idx * Q3_K_SUPER_BLOCK_SIZE + t] =
            __float2half(d * (float)signedScale * (float)signed3);
    }
}

// ── Q4_K: 144 bytes per 256 values (super-block with 8 sub-blocks) ──
// struct block_q4_K { half d; half dmin; uint8_t scales[12]; uint8_t qs[128]; };
#define Q4_K_SUPER_BLOCK_SIZE 256
#define Q4_K_BLOCK_BYTES 144

extern "C" __global__ void __launch_bounds__(256) dequant_q4_k_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_superblocks)
{
    // One CUDA block per superblock, 256 threads = one thread per output element.
    // Grid-stride loop over superblocks.
    int t = threadIdx.x; // 0..255

    for (int sb_idx = blockIdx.x; sb_idx < total_superblocks; sb_idx += gridDim.x)
    {
        const uint8_t* block = src + (size_t)sb_idx * Q4_K_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qs = block + 16;

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
        dst[(size_t)sb_idx * Q4_K_SUPER_BLOCK_SIZE + t] = __float2half(result);
    }
}

// ── Q5_K: 176 bytes per 256 values ──────────────────────────────────
// struct block_q5_K { half d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
#define Q5_K_SUPER_BLOCK_SIZE 256
#define Q5_K_BLOCK_BYTES 176

extern "C" __global__ void __launch_bounds__(256) dequant_q5_k_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_superblocks)
{
    int t = threadIdx.x; // 0..255

    for (int sb_idx = blockIdx.x; sb_idx < total_superblocks; sb_idx += gridDim.x)
    {
        const uint8_t* block = src + (size_t)sb_idx * Q5_K_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qh = block + 16;   // 32 bytes
        const uint8_t* qs = block + 48;   // 128 bytes

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

        const uint8_t* sub_qs = qs + sub * 16;
        const uint8_t* sub_qh = qh + sub * 4;

        // pos 0..31: interleaved low/high nibbles
        // sub_out[2*j+0]=lo, sub_out[2*j+1]=hi for j=0..15
        int j = pos / 2;
        uint8_t packed = sub_qs[j];
        int nibble = (pos & 1) ? (packed >> 4) : (packed & 0x0F);

        // Extract 5th bit from qh
        int bit = (sub_qh[j / 4] >> ((j % 4) * 2 + (pos & 1))) & 1;
        int val = nibble | (bit << 4);

        dst[(size_t)sb_idx * Q5_K_SUPER_BLOCK_SIZE + t] = __float2half(scale * (float)val - min_val);
    }
}

// ── Q6_K: 210 bytes per 256 values ──────────────────────────────────
// struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; };
#define Q6_K_SUPER_BLOCK_SIZE 256
#define Q6_K_BLOCK_BYTES 210

extern "C" __global__ void __launch_bounds__(256) dequant_q6_k_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_superblocks)
{
    int t = threadIdx.x; // 0..255

    for (int sb_idx = blockIdx.x; sb_idx < total_superblocks; sb_idx += gridDim.x)
    {
        const uint8_t* block = src + (size_t)sb_idx * Q6_K_BLOCK_BYTES;
        const uint8_t* ql = block;           // 128 bytes
        const uint8_t* qh_base = block + 128;     // 64 bytes
        const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192); // 16 bytes
        float d = __half2float(*reinterpret_cast<const half*>(block + 208));

        // Two 128-element halves (t<128 → first half, t>=128 → second half)
        int half_idx = t / 128;
        int pos_in_half = t % 128;

        const uint8_t* ql_half = ql + half_idx * 64;
        const uint8_t* qh_half = qh_base + half_idx * 32;
        const int8_t* sc_half = scales + half_idx * 8;

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
        dst[(size_t)sb_idx * Q6_K_SUPER_BLOCK_SIZE + t] = __float2half(sc * (float)q_val);
    }
}
