// ISO port of quant_kv.cu

#include <metal_stdlib>
using namespace metal;

// ── Q8_0: 34 bytes per 32 values ─────────────────────────────────────
// struct block_q8_0 { half d; int8_t qs[32]; };
#define Q8_0_BLOCK_SIZE  32
#define Q8_0_BLOCK_BYTES 34

kernel void quant_f16_to_q8_0(
    device const half*    src           [[buffer(0)]],
    device uint8_t*       dst           [[buffer(1)]],
    constant int&         total_blocks  [[buffer(2)]],
    uint                  gid           [[thread_position_in_grid]])
{
    int block_idx = (int)gid;
    if (block_idx >= total_blocks) return;

    device const half* in  = src + (uint)block_idx * Q8_0_BLOCK_SIZE;
    device uint8_t*    out = dst + (uint)block_idx * Q8_0_BLOCK_BYTES;

    // Find max absolute value across the 32 elements.
    float max_abs = 0.0f;
    float vals[Q8_0_BLOCK_SIZE];
    for (int j = 0; j < Q8_0_BLOCK_SIZE; j++)
    {
        vals[j] = float(in[j]);
        float a = fabs(vals[j]);
        if (a > max_abs) max_abs = a;
    }

    // Write scale (half) and quantize.
    float d = max_abs / 127.0f;
    *(device half*)(out) = half(d);

    device char* qs = (device char*)(out + 2);
    if (d == 0.0f)
    {
        for (int j = 0; j < Q8_0_BLOCK_SIZE; j++)
            qs[j] = 0;
    }
    else
    {
        float inv_d = 1.0f / d;
        for (int j = 0; j < Q8_0_BLOCK_SIZE; j++)
        {
            int v = int(round(vals[j] * inv_d));
            qs[j] = (char)clamp(v, -127, 127);
        }
    }
}

// ── Q4_0: 18 bytes per 32 values ─────────────────────────────────────
// struct block_q4_0 { half d; uint8_t qs[16]; };
// Values are stored with +8 offset so they fit in [0, 15].
#define Q4_0_BLOCK_SIZE  32
#define Q4_0_BLOCK_BYTES 18

kernel void quant_f16_to_q4_0(
    device const half*    src           [[buffer(0)]],
    device uint8_t*       dst           [[buffer(1)]],
    constant int&         total_blocks  [[buffer(2)]],
    uint                  gid           [[thread_position_in_grid]])
{
    int block_idx = (int)gid;
    if (block_idx >= total_blocks) return;

    device const half* in  = src + (uint)block_idx * Q4_0_BLOCK_SIZE;
    device uint8_t*    out = dst + (uint)block_idx * Q4_0_BLOCK_BYTES;

    // Find max absolute value across the 32 elements.
    float max_abs = 0.0f;
    float vals[Q4_0_BLOCK_SIZE];
    for (int j = 0; j < Q4_0_BLOCK_SIZE; j++)
    {
        vals[j] = float(in[j]);
        float a = fabs(vals[j]);
        if (a > max_abs) max_abs = a;
    }

    // Scale maps [-max_abs, max_abs] → [-7, 7].
    float d = max_abs / 7.0f;
    *(device half*)(out) = half(d);

    device uint8_t* qs = out + 2;
    if (d == 0.0f)
    {
        // Center value (8) represents 0 in offset encoding.
        for (int j = 0; j < 16; j++)
            qs[j] = 0x88; // (8 << 4) | 8
    }
    else
    {
        float inv_d = 1.0f / d;
        for (int j = 0; j < 16; j++)
        {
            int lo = clamp(int(round(vals[2 * j]     * inv_d)) + 8, 0, 15);
            int hi = clamp(int(round(vals[2 * j + 1] * inv_d)) + 8, 0, 15);
            qs[j] = (uint8_t)((hi << 4) | lo);
        }
    }
}
