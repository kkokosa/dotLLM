// Fused SwiGLU activation kernel for dotLLM.
// out[i] = SiLU(gate[i]) * up[i] = gate[i] * sigmoid(gate[i]) * up[i]

#include <cuda_fp16.h>

extern "C" __global__ void swiglu_f16(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    const int n,
    const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * seq_len;

    if (idx < total)
    {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        // SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g))
        float silu_g = g / (1.0f + expf(-g));
        output[idx] = __float2half(silu_g * u);
    }
}
