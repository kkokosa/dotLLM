#!/usr/bin/env bash
# Run inside a prepared package. Models and independent goldens use run-full.sh's layout.
set -eo pipefail
device="${1:?ascend or cuda}"
package_root="${2:-/tmp/package}"
comparison_dir="${3:-compare-followup}"
cd "$package_root"
export DOTNET_EnableWriteXorExecute=0
export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
if [[ "$device" == "cuda" ]]; then
    # These optional paths describe the tested worker's explicit compatibility setup.
    if [[ -d /tmp/dotllm-cublas12/nvidia/cublas/lib ]]; then
        export LD_LIBRARY_PATH="/usr/local/cuda/compat:/tmp/dotllm-cublas12/nvidia/cublas/lib:/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
    fi
elif [[ "$device" == "ascend" ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cmake -S native/ascend -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++
    cmake --build build --parallel 2
    export DOTLLM_ASCEND_LIBRARY="$package_root/build/libdotllm_ascend.so"
else
    echo "device must be ascend or cuda" >&2
    exit 2
fi
mkdir -p "results/$device-smol-followup" "results/$device-qwen-followup"
runtime/dotnet "$comparison_dir/DotLLM.Tests.BackendComparison.dll" "$device" \
    models/SmolLM-135M.Q8_0.gguf models/smol-text.json models/smol-numpy-f32.bin \
    "results/$device-smol-followup" all > "results/$device-smol-followup/metrics.jsonl" \
    2> "results/$device-smol-followup/stderr.txt"
runtime/dotnet "$comparison_dir/DotLLM.Tests.BackendComparison.dll" "$device" \
    models/qwen2.5-0.5b-instruct-q8_0.gguf models/qwen-text.json models/qwen-numpy-f32.bin \
    "results/$device-qwen-followup" all > "results/$device-qwen-followup/metrics.jsonl" \
    2> "results/$device-qwen-followup/stderr.txt"
if [[ "$device" == "cuda" ]]; then
    mkdir -p results/cuda-qwen-rope-diagnostic
    COMPARE_ROPE_OVERRIDE=1 runtime/dotnet "$comparison_dir/DotLLM.Tests.BackendComparison.dll" cuda \
        models/qwen2.5-0.5b-instruct-q8_0.gguf models/qwen-text.json models/qwen-numpy-f32.bin \
        results/cuda-qwen-rope-diagnostic all > results/cuda-qwen-rope-diagnostic/metrics.jsonl \
        2> results/cuda-qwen-rope-diagnostic/stderr.txt
fi
echo "Execution complete; inspect numerical PASS/FAIL records independently."
