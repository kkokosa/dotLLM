// Fused SwiGLU with FP32 data.

#include <math.h>

extern "C" __global__ void __launch_bounds__(256) swiglu_f32(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ output, const int n, const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * seq_len)
    {
        float g = gate[idx], u = up[idx];
        output[idx] = (g / (1.0f + expf(-g))) * u;
    }
}
