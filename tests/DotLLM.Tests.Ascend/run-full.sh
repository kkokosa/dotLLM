#!/usr/bin/env bash
# Run from an extracted validation package, containing cli/, tests/, models/, native/ and runtime.tar.gz.
set -eo pipefail
package_root="$(cd "${1:?validation package directory}" && pwd)"
cd "$package_root"
mkdir -p results runtime build
if [[ ! -x runtime/dotnet ]]; then tar -xzf runtime.tar.gz -C runtime; fi
sha256sum -c SHA256SUMS > results/manifest-check.txt
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export DOTNET_EnableWriteXorExecute=0 DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
export DOTLLM_ASCEND_LIBRARY="$package_root/build/libdotllm_ascend.so"
export DOTNET_CLI_TELEMETRY_OPTOUT=1
{
    date -Is
    uname -a
    runtime/dotnet --list-runtimes
    g++ --version
    npu-smi info
    cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max 2>/dev/null ||
        cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true
    cat /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info 2>/dev/null || true
    physical_card=$(npu-smi info -l | awk '/NPU ID/ {print $NF; exit}')
    if [[ -n "$physical_card" ]]; then npu-smi info -t board -i "$physical_card" -c 0 || true; fi
} > results/environment.txt 2>&1
cmake -S native/ascend -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ > results/native-build.txt 2>&1
cmake --build build --parallel 2 >> results/native-build.txt 2>&1
nm -D --defined-only build/libdotllm_ascend.so > results/native-symbols.txt
status=0
run_case() {
    local case_name="$1"; shift
    if "$@" > "results/$case_name.stdout" 2> "results/$case_name.stderr"; then
        printf '{"stage":"%s","status":"PASS"}\n' "$case_name" | tee -a results/stages.jsonl
    else
        status=1
        printf '{"stage":"%s","status":"FAIL"}\n' "$case_name" | tee -a results/stages.jsonl
    fi
    cat "results/$case_name.stdout" "results/$case_name.stderr"
}
run_case kernels runtime/dotnet tests/DotLLM.Tests.Ascend.dll --device 0
run_case capability runtime/dotnet tests/DotLLM.Tests.Ascend.dll --capability-self-test fixtures/tiny-llama-F32.gguf
run_case smol runtime/dotnet tests/DotLLM.Tests.Ascend.dll --model-test models/SmolLM-135M.Q8_0.gguf --reference models/SmolLM-135M.Q8_0.gguf --math-reference models/smol-numpy-f32.bin
run_case qwen runtime/dotnet tests/DotLLM.Tests.Ascend.dll --model-test models/qwen2.5-0.5b-instruct-q8_0.gguf --reference models/qwen2.5-0.5b-instruct-q8_0.gguf --math-reference models/qwen-numpy-f32.bin
for architecture in llama qwen2; do
    for storage in F32 F16 Q8_0; do
        fixture="fixtures/tiny-$architecture-$storage.gguf"
        run_case "tiny-$architecture-$storage" runtime/dotnet tests/DotLLM.Tests.Ascend.dll --model-test "$fixture" --reference "$fixture.reference" --math-reference "$fixture.golden"
    done
done
run_case smol-text runtime/dotnet tests/DotLLM.Tests.Ascend.dll --text-test models/SmolLM-135M.Q8_0.gguf models/smol-text.json models/smol-text.bin
run_case qwen-text runtime/dotnet tests/DotLLM.Tests.Ascend.dll --text-test models/qwen2.5-0.5b-instruct-q8_0.gguf models/qwen-text.json models/qwen-text.bin
run_case memory runtime/dotnet tests/DotLLM.Tests.Ascend.dll --memory-test models/SmolLM-135M.Q8_0.gguf
run_case nonfinite runtime/dotnet tests/DotLLM.Tests.Ascend.dll --nonfinite-test fixtures/nonfinite.gguf
gcc -shared -fPIC -Wall -Wextra -Werror fault_bridge.c -o build/fault-bridge.so
run_case fault env DOTLLM_ASCEND_LIBRARY="$package_root/build/fault-bridge.so" runtime/dotnet tests/DotLLM.Tests.Ascend.dll --fault-test
run_case service python3 service_smoke.py --dotnet "$package_root/runtime/dotnet" --cli "$package_root/cli/DotLLM.Cli.dll" --models "$package_root/models/SmolLM-135M.Q8_0.gguf" "$package_root/models/qwen2.5-0.5b-instruct-q8_0.gguf" --output "$package_root/results/service"
printf '{"exitCode":%d}\n' "$status" | tee results/exit.json
exit "$status"
