// FP32 element-wise addition for the residual stream.
// output[i] = a[i] + b[i]  (all FP32, in-place safe)

extern "C" __global__ void add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] = a[idx] + b[idx];
}

// FP16 input + FP32 accumulator: output_f32 = a_f32 + b_f16
// Used when adding FP16 projection output into FP32 residual stream.
#include <cuda_fp16.h>

extern "C" __global__ void add_f32_f16(
    const float* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ output,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] = a[idx] + __half2float(b[idx]);
}
