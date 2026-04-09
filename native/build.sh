#!/bin/bash
# Compile all .cu kernels to PTX for dotLLM CUDA backend.
# Requires: nvcc (CUDA Toolkit)
# Output: native/ptx/*.ptx
#
# PTX is forward-compatible: compute_61 PTX runs on all GPUs from Pascal onward.
# The CUDA driver JIT-compiles PTX → SASS for the specific GPU at load time.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/ptx"
KERNEL_DIR="$SCRIPT_DIR/kernels"

mkdir -p "$OUT_DIR"

# Target the lowest compute capability we support (Pascal / GTX 10xx).
# The driver will JIT to the actual GPU's native ISA at load time.
ARCH="compute_61"

# Kernels where --use_fast_math is safe (element-wise ops, no precision-sensitive math):
FAST_MATH_KERNELS="add add_f32 swiglu swiglu_f32 convert bias_add bias_add_f32 embedding embedding_f32out dequant quant_kv quantized_gemv quantized_gemv_f32in"

# Kernels requiring precise math (expf, rsqrtf, sinf, cosf, powf):
# - softmax/attention: expf in softmax accumulates error
# - rmsnorm/fused_add_rmsnorm: rsqrtf precision matters
# - rope: sinf/cosf/powf precision matters for position encoding
PRECISE_KERNELS="softmax rmsnorm rmsnorm_f32 rmsnorm_f32in rope rope_f32 attention attention_f32 fused_add_rmsnorm per_head_rmsnorm per_head_rmsnorm_f32"

is_fast_math_kernel() {
    local name="$1"
    for fm in $FAST_MATH_KERNELS; do
        [ "$fm" = "$name" ] && return 0
    done
    return 1
}

echo "Compiling CUDA kernels → PTX (target: $ARCH)..."

for cu_file in "$KERNEL_DIR"/*.cu; do
    [ -f "$cu_file" ] || continue
    base=$(basename "$cu_file" .cu)

    if is_fast_math_kernel "$base"; then
        nvcc -ptx -arch="$ARCH" \
             --use_fast_math \
             -o "$OUT_DIR/$base.ptx" \
             "$cu_file"
        echo "  $base.cu → $base.ptx (fast_math)"
    else
        nvcc -ptx -arch="$ARCH" \
             -o "$OUT_DIR/$base.ptx" \
             "$cu_file"
        echo "  $base.cu → $base.ptx (precise)"
    fi
done

echo "Done. PTX files in $OUT_DIR/"
