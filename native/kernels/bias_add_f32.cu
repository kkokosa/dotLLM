// FP32 bias addition.

#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256) bias_add_f32(
    float* __restrict__ output, const half* __restrict__ bias,
    const int dim, const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim * seq_len)
        output[idx] += __half2float(bias[idx % dim]);
}
