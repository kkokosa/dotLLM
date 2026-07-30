#!/usr/bin/env bash
set -euo pipefail

# delete bin folder and don't crash if not exists
rm -R bin 2> /dev/null || true

mkdir bin
mkdir bin/gpu

DYLIB="bin/libdotllmmetal.dylib"

clang++ \
  -std=c++17 \
  -shared \
  -fobjc-arc \
  -x objective-c++ \
  bridge.mm \
  -framework Foundation \
  -framework Metal \
  -framework MetalPerformanceShaders \
  -o "$DYLIB"

# Compile each .metal → .air, then link them all into a single dotllm_kernels.metallib.
# This single archive is loaded once at context creation; no runtime MSL→AIR
# compilation, no .metal source files needed at runtime.
for metal_file in gpu/*.metal; do
    name="${metal_file%.metal}"
    echo "founf metal file $metal_file"
    xcrun -sdk macosx metal -c "$metal_file" -o "bin/${name}.air"
done

xcrun -sdk macosx metallib bin/gpu/*.air -o bin/dotllm_kernels.metallib

# Per-shader .air intermediates are no longer needed once the archive is built.
rm -f bin/gpu/*.air

# ── Build summary ────────────────────────────────────────────────────────────
echo
echo "── Build summary ────────────────────────────────────────────"

# 1) Dylib size (KB) — quick sanity check that linking succeeded.
size_kb=$(($(stat -f%z "$DYLIB") / 1024))
echo "dylib              : $DYLIB (${size_kb} KB)"

# 2) Architecture (arm64 / x86_64 / universal).
arch=$(otool -hv "$DYLIB" | awk 'NR==4 {print $2}')
echo "architecture       : $arch"

# 3) Frameworks linked — filtered to what we care about. A missing framework
#    here means the build silently dropped a -framework flag.
echo "frameworks linked  :"
otool -L "$DYLIB" \
  | awk '/Metal|Foundation/ {print "    "$1}'

# 4) Exported entry points — every dotllm_metal_* symbol must be present,
#    otherwise P/Invoke fails at runtime with EntryPointNotFoundException.
exports=$(nm -gU "$DYLIB" 2>/dev/null | grep -c "_dotllm_metal_" || true)
echo "exported symbols   : $exports dotllm_metal_* entry points"

# 5) Combined Metal library — single archive loaded once at runtime.
metal_count=$(ls *.metal 2>/dev/null | wc -l | tr -d ' ')
if [ -f bin/dotllm_kernels.metallib ]; then
    metallib_kb=$(($(stat -f%z bin/dotllm_kernels.metallib) / 1024))
    echo "metallib           : bin/dotllm_kernels.metallib (${metallib_kb} KB, $metal_count shaders linked)"
else
    echo "metallib           : MISSING — runtime will fail to load"
fi

echo "─────────────────────────────────────────────────────────────"
