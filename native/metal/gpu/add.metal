// ISO port of add.cu.

#include <metal_stdlib>
using namespace metal;

// ── add_f16 ──────────────────────────────────────────────────────────────────
// Port of add.cu::add_f16
kernel void add_f16(
    device const half* a      [[buffer(0)]],
    device const half* b      [[buffer(1)]],
    device       half* output [[buffer(2)]],
    constant uint&     n      [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    uint n2 = n / 2u;

    if (idx < n2)
    {
        device const half2* a2   = (device const half2*)a;
        device const half2* b2   = (device const half2*)b;
        device       half2* out2 = (device half2*)output;
        out2[idx] = a2[idx] + b2[idx];
    }

    // Handle odd trailing element (single thread)
    if ((n & 1u) && idx == 0u)
    {
        uint last = n - 1u;
        output[last] = a[last] + b[last];
    }
}
