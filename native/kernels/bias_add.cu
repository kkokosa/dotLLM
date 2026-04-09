// Bias addition kernel for dotLLM.
// output[t, i] += bias[i]  for t in [0, seqLen)
// Vectorized: half2 packed operations when dim is even.

#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256) bias_add_f16(
    half* __restrict__ output,
    const half* __restrict__ bias,
    const int dim,
    const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dim * seq_len;
    int dim2 = dim / 2;
    int total2 = total / 2;

    // dim is always even for transformer hidden sizes, so half2 is safe
    if (idx < total2)
    {
        half2* out2 = reinterpret_cast<half2*>(output);
        const half2* bias2 = reinterpret_cast<const half2*>(bias);

        // Map half2 index back to row/col pair index
        // Each row has dim elements = dim/2 half2 elements
        int col2 = idx % dim2;
        out2[idx] = __hadd2(out2[idx], bias2[col2]);
    }

    // Handle odd dim (shouldn't happen for transformers, but be safe)
    if ((total & 1) && idx == 0)
    {
        int last = total - 1;
        int col = last % dim;
        output[last] = __float2half(__half2float(output[last]) + __half2float(bias[col]));
    }
}
