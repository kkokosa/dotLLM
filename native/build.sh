#!/bin/bash
# Compile all .cu kernels to PTX for dotLLM CUDA backend.
# Requires: nvcc (CUDA Toolkit)
# Output: native/ptx/*.ptx
#
# PTX is forward-compatible: compute_61 PTX runs on all GPUs from Pascal onward.
# The CUDA driver JIT-compiles PTX → SASS for the specific GPU at load time.
#
# ── Arch-tiered PTX (optional) ────────────────────────────────────────────────
# In addition to the universal compute_61 "<kernel>.ptx" (always emitted), this
# script can OPTIONALLY emit higher-arch PTX variants named "<kernel>.sm_<arch>.ptx"
# for a curated subset of kernels. The runtime loader (CudaModule.LoadForArch)
# picks the highest-arch variant whose arch is <= the device compute capability,
# and falls back to the compute_61 "<kernel>.ptx" when no variant is present.
#
# This is opt-in. With no extra arguments the script produces EXACTLY today's
# output: only compute_61 "<kernel>.ptx" files. To also emit higher-arch variants:
#
#   EXTRA_ARCHS="80 86"            ./build.sh   # variants for the default kernel list
#   EXTRA_ARCHS="80" \
#   EXTRA_ARCH_KERNELS="quantized_gemv" ./build.sh
#
# EXTRA_ARCHS         space-separated SM numbers (e.g. "75 80 86 90"). Empty = none.
# EXTRA_ARCH_KERNELS  space-separated kernel base names to also build for EXTRA_ARCHS.
#                     Defaults to ARCH_TIERED_KERNELS below. Only kernels with a
#                     genuinely arch-specific implementation belong here; none exist
#                     yet, so the default list is empty (true no-op).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/ptx"
KERNEL_DIR="$SCRIPT_DIR/kernels"

mkdir -p "$OUT_DIR"

# Target the lowest compute capability we support (Pascal / GTX 10xx).
# The driver will JIT to the actual GPU's native ISA at load time.
ARCH="compute_61"

# Optional higher-arch PTX variants (see header). Default: none → no-op.
EXTRA_ARCHS="${EXTRA_ARCHS:-}"
# Curated kernel list eligible for higher-arch variants. Empty until an
# arch-specific kernel implementation actually exists.
ARCH_TIERED_KERNELS=""
EXTRA_ARCH_KERNELS="${EXTRA_ARCH_KERNELS:-$ARCH_TIERED_KERNELS}"

# Kernels where --use_fast_math is safe (element-wise ops, no precision-sensitive math):
FAST_MATH_KERNELS="add add_f32 swiglu swiglu_f32 convert bias_add bias_add_f32 embedding embedding_f32out dequant quant_kv"

# Kernels requiring precise math (expf, rsqrtf, sinf, cosf, powf):
# - softmax/attention: expf in softmax accumulates error
# - rmsnorm/fused_add_rmsnorm: rsqrtf precision matters
# - rope: sinf/cosf/powf precision matters for position encoding
# - quantized_gemv: feeds precision-sensitive downstream ops
PRECISE_KERNELS="softmax rmsnorm rmsnorm_f32 rmsnorm_f32in rope rope_f32 attention attention_f32 fused_add_rmsnorm per_head_rmsnorm per_head_rmsnorm_f32 quantized_gemv quantized_gemv_f32in"

is_fast_math_kernel() {
    local name="$1"
    for fm in $FAST_MATH_KERNELS; do
        [ "$fm" = "$name" ] && return 0
    done
    return 1
}

is_in_list() {
    local name="$1"; shift
    for item in $@; do
        [ "$item" = "$name" ] && return 0
    done
    return 1
}

# compile <cu_file> <arch> <out_ptx>
compile_ptx() {
    local cu_file="$1" arch="$2" out_ptx="$3"
    local base
    base=$(basename "$cu_file" .cu)
    if is_fast_math_kernel "$base"; then
        nvcc -ptx -arch="$arch" --use_fast_math -o "$out_ptx" "$cu_file"
    else
        nvcc -ptx -arch="$arch" -o "$out_ptx" "$cu_file"
    fi
}

echo "Compiling CUDA kernels → PTX (target: $ARCH)..."

for cu_file in "$KERNEL_DIR"/*.cu; do
    [ -f "$cu_file" ] || continue
    base=$(basename "$cu_file" .cu)

    # Universal compute_61 PTX — always emitted (today's behavior).
    compile_ptx "$cu_file" "$ARCH" "$OUT_DIR/$base.ptx"
    if is_fast_math_kernel "$base"; then
        echo "  $base.cu → $base.ptx (fast_math)"
    else
        echo "  $base.cu → $base.ptx (precise)"
    fi

    # Optional higher-arch variants for the curated kernel list.
    if [ -n "$EXTRA_ARCHS" ] && is_in_list "$base" $EXTRA_ARCH_KERNELS; then
        for sm in $EXTRA_ARCHS; do
            compile_ptx "$cu_file" "compute_$sm" "$OUT_DIR/$base.sm_$sm.ptx"
            echo "  $base.cu → $base.sm_$sm.ptx (arch-tiered)"
        done
    fi
done

echo "Done. PTX files in $OUT_DIR/"
