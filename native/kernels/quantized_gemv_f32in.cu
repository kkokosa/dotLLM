// Quantized GEMV with FP32 input (instead of FP16).
// y[n] = W_q8_0[n,k] @ x_f32[k], output FP32.

#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void quantized_gemv_q8_0_f32in(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int n, const int k)
{
    int row = blockIdx.x;
    if (row >= n) return;

    const int bpr = k / 32;
    const uint8_t* w_row = weight + (size_t)row * bpr * 34;
    float acc = 0.0f;

    for (int b = threadIdx.x; b < bpr; b += blockDim.x)
    {
        const uint8_t* block = w_row + b * 34;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
        float s = 0.0f;
        #pragma unroll 8
        for (int j = 0; j < 32; j++) s += (float)qs[j] * x[b * 32 + j];
        acc += d * s;
    }

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    __shared__ float ws[32];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    if (lane == 0) ws[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        int nw = (blockDim.x + 31) / 32;
        acc = (lane < nw) ? ws[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    }
    if (threadIdx.x == 0) y[row] = acc;
}
