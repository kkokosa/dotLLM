// Fused SwiGLU activation kernel for dotLLM.
// out[i] = SiLU(gate[i]) * up[i] = gate[i] * sigmoid(gate[i]) * up[i]
// Vectorized: half2 loads/stores, FP32 computation for sigmoid precision.

#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256) swiglu_f16(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    const int n,
    const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * seq_len;
    int total2 = total / 2;

    if (idx < total2)
    {
        half2 g2 = reinterpret_cast<const half2*>(gate)[idx];
        half2 u2 = reinterpret_cast<const half2*>(up)[idx];

        float g0 = __low2float(g2), g1 = __high2float(g2);
        float u0 = __low2float(u2), u1 = __high2float(u2);

        // SiLU(g) = g / (1 + exp(-g))
        float s0 = g0 / (1.0f + expf(-g0)) * u0;
        float s1 = g1 / (1.0f + expf(-g1)) * u1;

        reinterpret_cast<half2*>(output)[idx] = __floats2half2_rn(s0, s1);
    }

    // Handle odd trailing element
    if ((total & 1) && idx == 0)
    {
        int last = total - 1;
        float g = __half2float(gate[last]);
        float u = __half2float(up[last]);
        output[last] = __float2half(g / (1.0f + expf(-g)) * u);
    }
}
