// Dequantization kernels for dotLLM.
// Convert quantized weight blocks to FP16.
// Block layouts match docs/QUANTIZATION.md.

#include <cuda_fp16.h>
#include <stdint.h>

// ── Q8_0: 34 bytes per 32 values ────────────────────────────────────
// struct block_q8_0 { half d; int8_t qs[32]; };
#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BLOCK_BYTES 34

extern "C" __global__ void dequant_q8_0_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_blocks)
{
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    const uint8_t* block = src + (size_t)block_idx * Q8_0_BLOCK_BYTES;
    half scale = *reinterpret_cast<const half*>(block);
    float d = __half2float(scale);
    const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);

    half* out = dst + (size_t)block_idx * Q8_0_BLOCK_SIZE;
    for (int j = 0; j < Q8_0_BLOCK_SIZE; j++)
        out[j] = __float2half(d * (float)qs[j]);
}

// ── Q4_0: 18 bytes per 32 values ────────────────────────────────────
// struct block_q4_0 { half d; uint8_t qs[16]; };
#define Q4_0_BLOCK_SIZE 32
#define Q4_0_BLOCK_BYTES 18

extern "C" __global__ void dequant_q4_0_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_blocks)
{
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    const uint8_t* block = src + (size_t)block_idx * Q4_0_BLOCK_BYTES;
    half scale = *reinterpret_cast<const half*>(block);
    float d = __half2float(scale);
    const uint8_t* qs = block + 2;

    half* out = dst + (size_t)block_idx * Q4_0_BLOCK_SIZE;
    for (int j = 0; j < 16; j++)
    {
        uint8_t packed = qs[j];
        int lo = (int)(packed & 0x0F) - 8;
        int hi = (int)(packed >> 4) - 8;
        out[2 * j + 0] = __float2half(d * (float)lo);
        out[2 * j + 1] = __float2half(d * (float)hi);
    }
}

// ── Q5_0: 22 bytes per 32 values ────────────────────────────────────
// struct block_q5_0 { half d; uint32_t qh; uint8_t qs[16]; };
#define Q5_0_BLOCK_SIZE 32
#define Q5_0_BLOCK_BYTES 22

extern "C" __global__ void dequant_q5_0_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_blocks)
{
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    const uint8_t* block = src + (size_t)block_idx * Q5_0_BLOCK_BYTES;
    float d = __half2float(*reinterpret_cast<const half*>(block));
    // Read qh as 4 bytes — block+2 is not 4-byte aligned (Q5_0 blocks are 22 bytes)
    unsigned int qh = (unsigned int)block[2] | ((unsigned int)block[3] << 8) |
                      ((unsigned int)block[4] << 16) | ((unsigned int)block[5] << 24);
    const uint8_t* qs = block + 6;

    half* out = dst + (size_t)block_idx * Q5_0_BLOCK_SIZE;
    for (int j = 0; j < 16; j++)
    {
        uint8_t packed = qs[j];
        int lo = (packed & 0x0F) | (((qh >> j) & 1) << 4);
        int hi = (packed >> 4) | (((qh >> (j + 16)) & 1) << 4);
        // Low nibbles → elements 0..15, high nibbles → elements 16..31
        out[j]      = __float2half(d * (float)(lo - 16));
        out[j + 16] = __float2half(d * (float)(hi - 16));
    }
}

// ── Q4_K: 144 bytes per 256 values (super-block with 8 sub-blocks) ──
// struct block_q4_K { half d; half dmin; uint8_t scales[12]; uint8_t qs[128]; };
#define Q4_K_SUPER_BLOCK_SIZE 256
#define Q4_K_BLOCK_BYTES 144

extern "C" __global__ void dequant_q4_k_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_superblocks)
{
    int sb_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sb_idx >= total_superblocks) return;

    const uint8_t* block = src + (size_t)sb_idx * Q4_K_BLOCK_BYTES;
    float d = __half2float(*reinterpret_cast<const half*>(block));
    float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
    const uint8_t* scales_raw = block + 4;
    const uint8_t* qs = block + 16; // 4 + 12 = 16

    half* out = dst + (size_t)sb_idx * Q4_K_SUPER_BLOCK_SIZE;

    // Decode 6-bit scales and mins from 12 bytes.
    // Q4_K layout: 4 pairs of sub-blocks, each pair shares 32 qs bytes.
    // Within each pair: lower nibbles → even sub-block, upper nibbles → odd sub-block.
    for (int pair = 0; pair < 4; pair++)
    {
        int sb_even = pair * 2;
        int sb_odd = pair * 2 + 1;

        int sc0, m0, sc1, m1;
        if (sb_even < 4)
        {
            sc0 = scales_raw[sb_even] & 0x3F;
            m0 = scales_raw[sb_even + 4] & 0x3F;
            sc1 = scales_raw[sb_odd] & 0x3F;
            m1 = scales_raw[sb_odd + 4] & 0x3F;
        }
        else
        {
            sc0 = (scales_raw[sb_even + 4] & 0x0F) | ((scales_raw[sb_even - 4] >> 6) << 4);
            m0 = (scales_raw[sb_even + 4] >> 4) | ((scales_raw[sb_even] >> 6) << 4);
            sc1 = (scales_raw[sb_odd + 4] & 0x0F) | ((scales_raw[sb_odd - 4] >> 6) << 4);
            m1 = (scales_raw[sb_odd + 4] >> 4) | ((scales_raw[sb_odd] >> 6) << 4);
        }

        float scale0 = d * (float)sc0;
        float min0 = dmin * (float)m0;
        float scale1 = d * (float)sc1;
        float min1 = dmin * (float)m1;

        const uint8_t* pair_qs = qs + pair * 32;
        half* pair_out = out + pair * 64;

        for (int j = 0; j < 32; j++)
        {
            uint8_t byte_val = pair_qs[j];
            pair_out[j]      = __float2half(scale0 * (float)(byte_val & 0x0F) - min0);
            pair_out[j + 32] = __float2half(scale1 * (float)(byte_val >> 4) - min1);
        }
    }
}

// ── Q5_K: 176 bytes per 256 values ──────────────────────────────────
// struct block_q5_K { half d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
#define Q5_K_SUPER_BLOCK_SIZE 256
#define Q5_K_BLOCK_BYTES 176

extern "C" __global__ void dequant_q5_k_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_superblocks)
{
    int sb_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sb_idx >= total_superblocks) return;

    const uint8_t* block = src + (size_t)sb_idx * Q5_K_BLOCK_BYTES;
    float d = __half2float(*reinterpret_cast<const half*>(block));
    float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
    const uint8_t* scales_raw = block + 4;
    const uint8_t* qh = block + 16; // 4 + 12 = 16
    const uint8_t* qs = block + 48; // 16 + 32 = 48

    half* out = dst + (size_t)sb_idx * Q5_K_SUPER_BLOCK_SIZE;

    for (int sb = 0; sb < 8; sb++)
    {
        int sc, m;
        if (sb < 4)
        {
            sc = scales_raw[sb] & 0x3F;
            m = scales_raw[sb + 4] & 0x3F;
        }
        else
        {
            sc = (scales_raw[sb + 4] & 0x0F) | ((scales_raw[sb - 4] >> 6) << 4);
            m = (scales_raw[sb + 4] >> 4) | ((scales_raw[sb] >> 6) << 4);
        }

        float scale = d * (float)sc;
        float min = dmin * (float)m;

        const uint8_t* sub_qs = qs + sb * 16;
        const uint8_t* sub_qh = qh + sb * 4;
        half* sub_out = out + sb * 32;

        for (int j = 0; j < 16; j++)
        {
            uint8_t packed = sub_qs[j];
            int lo = packed & 0x0F;
            int hi = packed >> 4;

            // Extract 5th bit from qh
            int bit_lo = (sub_qh[j / 4] >> ((j % 4) * 2)) & 1;
            int bit_hi = (sub_qh[j / 4] >> ((j % 4) * 2 + 1)) & 1;

            lo |= (bit_lo << 4);
            hi |= (bit_hi << 4);

            sub_out[2 * j + 0] = __float2half(scale * (float)lo - min);
            sub_out[2 * j + 1] = __float2half(scale * (float)hi - min);
        }
    }
}

// ── Q6_K: 210 bytes per 256 values ──────────────────────────────────
// struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; };
#define Q6_K_SUPER_BLOCK_SIZE 256
#define Q6_K_BLOCK_BYTES 210

extern "C" __global__ void dequant_q6_k_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_superblocks)
{
    int sb_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sb_idx >= total_superblocks) return;

    const uint8_t* block = src + (size_t)sb_idx * Q6_K_BLOCK_BYTES;
    const uint8_t* ql = block;           // 128 bytes
    const uint8_t* qh = block + 128;     // 64 bytes
    const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192); // 16 bytes
    float d = __half2float(*reinterpret_cast<const half*>(block + 208));

    half* out = dst + (size_t)sb_idx * Q6_K_SUPER_BLOCK_SIZE;

    // Two 128-element halves. Within each half:
    // ql lower nibbles (2×32 bytes) → first 64 values, upper nibbles → next 64 values.
    // qh packs 4×2-bit hi-bits per byte for groups of 32 elements.
    for (int half_idx = 0; half_idx < 2; half_idx++)
    {
        const uint8_t* ql_half = ql + half_idx * 64;
        const uint8_t* qh_half = qh + half_idx * 32;
        const int8_t* sc_half = scales + half_idx * 8;
        half* out_half = out + half_idx * 128;

        for (int l = 0; l < 32; l++)
        {
            int isc = l / 16;

            int q1 = ((ql_half[l]      & 0x0F) | (((qh_half[l] >> 0) & 3) << 4)) - 32;
            int q2 = ((ql_half[l + 32] & 0x0F) | (((qh_half[l] >> 2) & 3) << 4)) - 32;
            int q3 = ((ql_half[l]      >> 4)    | (((qh_half[l] >> 4) & 3) << 4)) - 32;
            int q4 = ((ql_half[l + 32] >> 4)    | (((qh_half[l] >> 6) & 3) << 4)) - 32;

            out_half[l]      = __float2half(d * (float)sc_half[isc]     * (float)q1);
            out_half[l + 32] = __float2half(d * (float)sc_half[isc + 2] * (float)q2);
            out_half[l + 64] = __float2half(d * (float)sc_half[isc + 4] * (float)q3);
            out_half[l + 96] = __float2half(d * (float)sc_half[isc + 6] * (float)q4);
        }
    }
}
