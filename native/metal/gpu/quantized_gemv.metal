// ISO port of quantized_gemv.cu.
//
// y[n] = W[n,k] @ x[k]  where W is quantized, x is FP16, output is FP16.
// Operates directly on quantized weight blocks — no dequant to FP16 intermediate.
// FP32 accumulation. One threadgroup per output row, warp reduction for the sum.
//
// CUDA → Metal mapping (identical in every kernel below):
//   blockIdx.x      → row          [[threadgroup_position_in_grid]]
//   threadIdx.x     → threadIdx_x  [[thread_position_in_threadgroup]]
//   blockDim.x      → blockDim_x   [[threads_per_threadgroup]]
//   __shfl_down_sync→ simd_shuffle_down (warpSize = 32 on Apple Silicon)
//   __shared__      → threadgroup
//   __syncthreads() → threadgroup_barrier(mem_flags::mem_threadgroup)

#include <metal_stdlib>
using namespace metal;

// ── Q8_0: 34 bytes per 32 values ────────────────────────────────────
// scale (half) applied once per block: acc += float(scale) * sum(qs[j] * x[j])
kernel void quantized_gemv_q8_0(
    device const uchar* weight [[buffer(0)]],
    device const half*  x      [[buffer(1)]],
    device       half*  y      [[buffer(2)]],
    constant int&       n      [[buffer(3)]],
    constant int&       k      [[buffer(4)]],
    uint row         [[threadgroup_position_in_grid]],
    uint threadIdx_x [[thread_position_in_threadgroup]],
    uint blockDim_x  [[threads_per_threadgroup]])
{
    if ((int)row >= n) return;

    const int blocks_per_row = k / 32;
    device const uchar* w_row = weight + (size_t)row * blocks_per_row * 34;

    float acc = 0.0f;

    for (int b = (int)threadIdx_x; b < blocks_per_row; b += (int)blockDim_x)
    {
        device const uchar* block = w_row + b * 34;
        float d = float(*(device const half*)block);
        device const char* qs = (device const char*)(block + 2);

        float block_sum = 0.0f;
        for (int j = 0; j < 32; j++)
            block_sum += (float)qs[j] * float(x[b * 32 + j]);

        acc += d * block_sum;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += simd_shuffle_down(acc, offset);

    threadgroup float warp_sums[32];
    int lane = threadIdx_x % 32;
    int warp_id = threadIdx_x / 32;

    if (lane == 0) warp_sums[warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_id == 0)
    {
        int num_warps = (blockDim_x + 31) / 32;
        acc = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += simd_shuffle_down(acc, offset);
    }

    if (threadIdx_x == 0)
        y[row] = half(acc);
}

// ── Q4_K: 144 bytes per 256 values ──────────────────────────────────
kernel void quantized_gemv_q4_k(
    device const uchar* weight [[buffer(0)]],
    device const half*  x      [[buffer(1)]],
    device       half*  y      [[buffer(2)]],
    constant int&       n      [[buffer(3)]],
    constant int&       k      [[buffer(4)]],
    uint row         [[threadgroup_position_in_grid]],
    uint threadIdx_x [[thread_position_in_threadgroup]],
    uint blockDim_x  [[threads_per_threadgroup]])
{
    if ((int)row >= n) return;

    const int superblocks_per_row = k / 256;
    device const uchar* w_row = weight + (size_t)row * superblocks_per_row * 144;

    float acc = 0.0f;

    for (int sb = (int)threadIdx_x; sb < superblocks_per_row; sb += (int)blockDim_x)
    {
        device const uchar* block = w_row + sb * 144;
        float d = float(*(device const half*)block);
        float dmin = float(*(device const half*)(block + 2));
        device const uchar* scales_raw = block + 4;
        device const uchar* qs = block + 16;

        // Q4_K: 4 pairs of sub-blocks, each pair shares 32 qs bytes.
        // Lower nibbles → even sub-block, upper nibbles → odd sub-block.
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

            device const uchar* pair_qs = qs + pair * 32;
            int base_x_idx = sb * 256 + pair * 64;

            for (int j = 0; j < 32; j++)
            {
                uchar byte_val = pair_qs[j];
                float x_even = float(x[base_x_idx + j]);
                float x_odd = float(x[base_x_idx + j + 32]);

                acc += (scale0 * (float)(byte_val & 0x0F) - min0) * x_even;
                acc += (scale1 * (float)(byte_val >> 4) - min1) * x_odd;
            }
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += simd_shuffle_down(acc, offset);

    threadgroup float warp_sums[32];
    int lane = threadIdx_x % 32;
    int warp_id = threadIdx_x / 32;

    if (lane == 0) warp_sums[warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_id == 0)
    {
        int num_warps = (blockDim_x + 31) / 32;
        acc = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += simd_shuffle_down(acc, offset);
    }

    if (threadIdx_x == 0)
        y[row] = half(acc);
}

// ── Q6_K: 210 bytes per 256 values ──────────────────────────────────
kernel void quantized_gemv_q6_k(
    device const uchar* weight [[buffer(0)]],
    device const half*  x      [[buffer(1)]],
    device       half*  y      [[buffer(2)]],
    constant int&       n      [[buffer(3)]],
    constant int&       k      [[buffer(4)]],
    uint row         [[threadgroup_position_in_grid]],
    uint threadIdx_x [[thread_position_in_threadgroup]],
    uint blockDim_x  [[threads_per_threadgroup]])
{
    if ((int)row >= n) return;

    const int superblocks_per_row = k / 256;
    device const uchar* w_row = weight + (size_t)row * superblocks_per_row * 210;

    float acc = 0.0f;

    for (int sb = (int)threadIdx_x; sb < superblocks_per_row; sb += (int)blockDim_x)
    {
        device const uchar* block = w_row + sb * 210;
        device const uchar* ql = block;
        device const uchar* qh = block + 128;
        device const char* scales = (device const char*)(block + 192);
        float d = float(*(device const half*)(block + 208));

        int base_x = sb * 256;

        // Two 128-element halves, matching Q6_K GGUF layout.
        for (int half_idx = 0; half_idx < 2; half_idx++)
        {
            device const uchar* ql_half = ql + half_idx * 64;
            device const uchar* qh_half = qh + half_idx * 32;
            device const char* sc_half = scales + half_idx * 8;
            int x_off = base_x + half_idx * 128;

            for (int l = 0; l < 32; l++)
            {
                int isc = l / 16;

                int q1 = ((ql_half[l]      & 0x0F) | (((qh_half[l] >> 0) & 3) << 4)) - 32;
                int q2 = ((ql_half[l + 32] & 0x0F) | (((qh_half[l] >> 2) & 3) << 4)) - 32;
                int q3 = ((ql_half[l]      >> 4)    | (((qh_half[l] >> 4) & 3) << 4)) - 32;
                int q4 = ((ql_half[l + 32] >> 4)    | (((qh_half[l] >> 6) & 3) << 4)) - 32;

                float s0 = d * (float)sc_half[isc];
                float s1 = d * (float)sc_half[isc + 2];
                float s2 = d * (float)sc_half[isc + 4];
                float s3 = d * (float)sc_half[isc + 6];

                acc += s0 * (float)q1 * float(x[x_off + l]);
                acc += s1 * (float)q2 * float(x[x_off + l + 32]);
                acc += s2 * (float)q3 * float(x[x_off + l + 64]);
                acc += s3 * (float)q4 * float(x[x_off + l + 96]);
            }
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += simd_shuffle_down(acc, offset);

    threadgroup float warp_sums[32];
    int lane = threadIdx_x % 32;
    int warp_id = threadIdx_x / 32;

    if (lane == 0) warp_sums[warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_id == 0)
    {
        int num_warps = (blockDim_x + 31) / 32;
        acc = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += simd_shuffle_down(acc, offset);
    }

    if (threadIdx_x == 0)
        y[row] = half(acc);
}

// ── Q5_0: 22 bytes per 32 values ────────────────────────────────────
// struct block_q5_0 { half d; uint32_t qh; uint8_t qs[16]; };
kernel void quantized_gemv_q5_0(
    device const uchar* weight [[buffer(0)]],
    device const half*  x      [[buffer(1)]],
    device       half*  y      [[buffer(2)]],
    constant int&       n      [[buffer(3)]],
    constant int&       k      [[buffer(4)]],
    uint row         [[threadgroup_position_in_grid]],
    uint threadIdx_x [[thread_position_in_threadgroup]],
    uint blockDim_x  [[threads_per_threadgroup]])
{
    if ((int)row >= n) return;

    const int blocks_per_row = k / 32;
    device const uchar* w_row = weight + (size_t)row * blocks_per_row * 22;

    float acc = 0.0f;

    for (int b = (int)threadIdx_x; b < blocks_per_row; b += (int)blockDim_x)
    {
        device const uchar* block = w_row + b * 22;
        float d = float(*(device const half*)block);
        // Read qh as 4 bytes (Q5_0 blocks are 22 bytes — may be unaligned)
        uint qh = (uint)block[2] | ((uint)block[3] << 8) |
                  ((uint)block[4] << 16) | ((uint)block[5] << 24);
        device const uchar* qs = block + 6;

        float block_sum = 0.0f;
        for (int j = 0; j < 16; j++)
        {
            uchar packed = qs[j];
            int lo = (packed & 0x0F) | (((qh >> j) & 1) << 4);
            int hi = (packed >> 4) | (((qh >> (j + 16)) & 1) << 4);
            // Element ordering matches dequant: out[j]=lo, out[j+16]=hi
            block_sum += (float)(lo - 16) * float(x[b * 32 + j]);
            block_sum += (float)(hi - 16) * float(x[b * 32 + j + 16]);
        }

        acc += d * block_sum;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += simd_shuffle_down(acc, offset);

    threadgroup float warp_sums[32];
    int lane = threadIdx_x % 32;
    int warp_id = threadIdx_x / 32;

    if (lane == 0) warp_sums[warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_id == 0)
    {
        int num_warps = (blockDim_x + 31) / 32;
        acc = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += simd_shuffle_down(acc, offset);
    }

    if (threadIdx_x == 0)
        y[row] = half(acc);
}

// ── Q5_K: 176 bytes per 256 values ──────────────────────────────────
// struct block_q5_K { half d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
kernel void quantized_gemv_q5_k(
    device const uchar* weight [[buffer(0)]],
    device const half*  x      [[buffer(1)]],
    device       half*  y      [[buffer(2)]],
    constant int&       n      [[buffer(3)]],
    constant int&       k      [[buffer(4)]],
    uint row         [[threadgroup_position_in_grid]],
    uint threadIdx_x [[thread_position_in_threadgroup]],
    uint blockDim_x  [[threads_per_threadgroup]])
{
    if ((int)row >= n) return;

    const int superblocks_per_row = k / 256;
    device const uchar* w_row = weight + (size_t)row * superblocks_per_row * 176;

    float acc = 0.0f;

    for (int sb = (int)threadIdx_x; sb < superblocks_per_row; sb += (int)blockDim_x)
    {
        device const uchar* block = w_row + sb * 176;
        float d = float(*(device const half*)block);
        float dmin = float(*(device const half*)(block + 2));
        device const uchar* scales_raw = block + 4;
        device const uchar* qh = block + 16;   // 32 bytes
        device const uchar* qs = block + 48;   // 128 bytes
        int base_x = sb * 256;

        for (int sub = 0; sub < 8; sub++)
        {
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

            int nibble_half = sub & 1;
            device const uchar* pair_qs = qs + (sub / 2) * 32;
            int x_off = base_x + sub * 32;

            for (int i = 0; i < 32; i++)
            {
                int lo4  = nibble_half == 0 ? (pair_qs[i] & 0x0F) : (pair_qs[i] >> 4);
                int bit5 = (qh[i] >> sub) & 1;
                acc += (scale * (float)(lo4 | (bit5 << 4)) - min_val) * float(x[x_off + i]);
            }
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += simd_shuffle_down(acc, offset);

    threadgroup float warp_sums[32];
    int lane = threadIdx_x % 32;
    int warp_id = threadIdx_x / 32;

    if (lane == 0) warp_sums[warp_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_id == 0)
    {
        int num_warps = (blockDim_x + 31) / 32;
        acc = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += simd_shuffle_down(acc, offset);
    }

    if (threadIdx_x == 0)
        y[row] = half(acc);
}
