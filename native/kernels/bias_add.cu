// Bias addition kernel for dotLLM.
// output[t, i] += bias[i]  for t in [0, seqLen)

#include <cuda_fp16.h>

extern "C" __global__ void bias_add_f16(
    half* __restrict__ output,
    const half* __restrict__ bias,
    const int dim,
    const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dim * seq_len;

    if (idx < total)
    {
        int col = idx % dim;
        float v = __half2float(output[idx]);
        float b = __half2float(bias[col]);
        output[idx] = __float2half(v + b);
    }
}
