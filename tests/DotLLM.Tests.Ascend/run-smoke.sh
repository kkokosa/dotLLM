#!/usr/bin/env bash
# Run a prebuilt managed test package with a bridge compiled against this worker's CANN.
# Package layout: managed/, native/ascend/, dotnet-runtime.tar.gz, SHA256SUMS.
set -eo pipefail
package_dir=$(realpath "${1:?Usage: run-smoke.sh PACKAGE_DIR OUTPUT_DIR [DEVICE]}")
output_dir=${2:?Missing output directory}
device_id=${3:-0}
mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")
exec > >(tee "$output_dir/smoke.log") 2>&1
trap 'result=$?; printf "{\"exitCode\":%s}\n" "$result" > "$output_dir/exit.json"' EXIT

if [[ $(uname -m) != x86_64 ]]; then
    echo "This smoke package contains the linux-x64 runtime."
    exit 2
fi
for command_name in cmake g++ npu-smi tar sha256sum timeout; do
    command -v "$command_name"
done
uname -a
cat /etc/os-release
npu-smi info
cd "$package_dir"
sha256sum --check SHA256SUMS

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi
set -u
toolkit_root=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}
test -f "$toolkit_root/include/acl/acl.h"
test -f "$toolkit_root/include/aclnnop/aclnn_matmul.h"
# Some CANN images export a compiler path that is absent from their filesystem.
# Select the installed compiler explicitly after sourcing the toolkit environment.
compiler_path=$(command -v g++)
test -x "$compiler_path"
export CXX="$compiler_path"
printf 'CANN toolkit: %s\n' "$toolkit_root"
if [[ -f "$toolkit_root/version.cfg" ]]; then cat "$toolkit_root/version.cfg"; fi

work_dir=$(mktemp -d /tmp/dotllm-ascend-smoke.XXXXXX)
mkdir -p "$work_dir/runtime" "$work_dir/managed"
tar -xzf "$package_dir/dotnet-runtime.tar.gz" -C "$work_dir/runtime"
cp -a "$package_dir/managed/." "$work_dir/managed/"
cmake -S "$package_dir/native/ascend" -B "$work_dir/build" \
    -DCMAKE_BUILD_TYPE=Release -DASCEND_TOOLKIT_ROOT="$toolkit_root" \
    -DCMAKE_CXX_COMPILER="$compiler_path"
cmake --build "$work_dir/build" --parallel 4
cp "$work_dir/build/libdotllm_ascend.so" "$work_dir/managed/"
ldd "$work_dir/managed/libdotllm_ascend.so"
export DOTNET_ROOT="$work_dir/runtime"
export DOTNET_EnableWriteXorExecute=0
export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
"$work_dir/runtime/dotnet" --info
timeout --signal=TERM --kill-after=10s 300s "$work_dir/runtime/dotnet" \
    "$work_dir/managed/DotLLM.Tests.Ascend.dll" --device "$device_id"
cp "$work_dir/build/libdotllm_ascend.so" "$output_dir/"
cp "$package_dir/SHA256SUMS" "$output_dir/"
