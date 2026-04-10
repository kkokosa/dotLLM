// Per-head RMS normalization with FP32 data and FP32 weight.

extern "C" __global__ void __launch_bounds__(256) per_head_rmsnorm_f32(
    float* __restrict__ qk, const float* __restrict__ weight,
    const float eps, const int num_heads, const int head_dim, const int seq_len)
{
    int block_id = blockIdx.x;
    int t = block_id / num_heads, h = block_id % num_heads;
    if (t >= seq_len) return;

    float* vec = qk + (size_t)t * num_heads * head_dim + h * head_dim;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) { float v = vec[i]; sum_sq += v * v; }
    for (int off = warpSize/2; off > 0; off >>= 1) sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);
    __shared__ float ws[32]; int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    if (lane == 0) ws[wid] = sum_sq; __syncthreads();
    if (wid == 0) { int nw = (blockDim.x+warpSize-1)/warpSize; sum_sq = (lane<nw)?ws[lane]:0.0f;
        for (int off=warpSize/2;off>0;off>>=1) sum_sq+=__shfl_down_sync(0xFFFFFFFF,sum_sq,off); }
    __shared__ float ri; if (threadIdx.x==0) ri = rsqrtf(sum_sq/(float)head_dim+eps); __syncthreads();
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        vec[i] = vec[i] * ri * weight[i];
}
