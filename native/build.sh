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

echo "Compiling CUDA kernels → PTX (target: $ARCH)..."

for cu_file in "$KERNEL_DIR"/*.cu; do
    [ -f "$cu_file" ] || continue
    base=$(basename "$cu_file" .cu)

    nvcc -ptx -arch="$ARCH" \
         --use_fast_math \
         -o "$OUT_DIR/$base.ptx" \
         "$cu_file"

    echo "  $base.cu → $base.ptx"
done

echo "Done. PTX files in $OUT_DIR/"
