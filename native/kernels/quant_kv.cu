// ─────────────────────────────────────────────────────────────────────
//  KV-cache quantization kernels: FP16 → Q8_0 and FP16 → Q4_0
//  Used for quantize-on-evict in CudaQuantizedKvCache.
// ─────────────────────────────────────────────────────────────────────

#include <cuda_fp16.h>
#include <stdint.h>

// ── Q8_0: 34 bytes per 32 values ────────────────────────────────────
// struct block_q8_0 { half d; int8_t qs[32]; };
#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BLOCK_BYTES 34

// One thread per block of 32 elements.
// Quantizes a row of FP16 values to Q8_0 format.
extern "C" __global__ void quant_f16_to_q8_0(
    const half* __restrict__ src,
    uint8_t* __restrict__ dst,
    const int total_blocks)
{
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    const half* in = src + (size_t)block_idx * Q8_0_BLOCK_SIZE;
    uint8_t* out = dst + (size_t)block_idx * Q8_0_BLOCK_BYTES;

    // Find max absolute value
    float max_abs = 0.0f;
    float vals[Q8_0_BLOCK_SIZE];
    #pragma unroll 8
    for (int j = 0; j < Q8_0_BLOCK_SIZE; j++)
    {
        vals[j] = __half2float(in[j]);
        float a = fabsf(vals[j]);
        if (a > max_abs) max_abs = a;
    }

    // Compute scale and write
    float d = max_abs / 127.0f;
    *reinterpret_cast<half*>(out) = __float2half(d);

    int8_t* qs = reinterpret_cast<int8_t*>(out + 2);
    if (d == 0.0f)
    {
        #pragma unroll 8
        for (int j = 0; j < Q8_0_BLOCK_SIZE; j++)
            qs[j] = 0;
    }
    else
    {
        float inv_d = 1.0f / d;
        #pragma unroll 8
        for (int j = 0; j < Q8_0_BLOCK_SIZE; j++)
        {
            int v = __float2int_rn(vals[j] * inv_d);
            qs[j] = (int8_t)max(-127, min(127, v));
        }
    }
}

// ── Q4_0: 18 bytes per 32 values ────────────────────────────────────
// struct block_q4_0 { half d; uint8_t qs[16]; };
#define Q4_0_BLOCK_SIZE 32
#define Q4_0_BLOCK_BYTES 18

extern "C" __global__ void quant_f16_to_q4_0(
    const half* __restrict__ src,
    uint8_t* __restrict__ dst,
    const int total_blocks)
{
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    const half* in = src + (size_t)block_idx * Q4_0_BLOCK_SIZE;
    uint8_t* out = dst + (size_t)block_idx * Q4_0_BLOCK_BYTES;

    // Find max absolute value
    float max_abs = 0.0f;
    float vals[Q4_0_BLOCK_SIZE];
    #pragma unroll 8
    for (int j = 0; j < Q4_0_BLOCK_SIZE; j++)
    {
        vals[j] = __half2float(in[j]);
        float a = fabsf(vals[j]);
        if (a > max_abs) max_abs = a;
    }

    float d = max_abs / 7.0f;
    *reinterpret_cast<half*>(out) = __float2half(d);

    uint8_t* qs = out + 2;
    if (d == 0.0f)
    {
        #pragma unroll 8
        for (int j = 0; j < 16; j++)
            qs[j] = 0x88; // (8 << 4) | 8
    }
    else
    {
        float inv_d = 1.0f / d;
        #pragma unroll 8
        for (int j = 0; j < 16; j++)
        {
            int lo = max(0, min(15, __float2int_rn(vals[2 * j] * inv_d) + 8));
            int hi = max(0, min(15, __float2int_rn(vals[2 * j + 1] * inv_d) + 8));
            qs[j] = (uint8_t)((hi << 4) | lo);
        }
    }
}
