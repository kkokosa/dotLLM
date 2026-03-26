// Full FP32 RMS Normalization: FP32 input, FP32 weight, FP32 output.

extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    const float* x = input + (size_t)row * n;
    float* y = output + (size_t)row * n;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    { float v = x[i]; sum_sq += v * v; }

    for (int off = 32/2; off > 0; off >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);
    __shared__ float ws[32];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    if (lane == 0) ws[wid] = sum_sq;
    __syncthreads();
    if (wid == 0) {
        int nw = (blockDim.x + 31) / 32;
        sum_sq = (lane < nw) ? ws[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);
    }
    __shared__ float ri;
    if (threadIdx.x == 0) ri = rsqrtf(sum_sq / (float)n + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        y[i] = x[i] * ri * weight[i];
}
