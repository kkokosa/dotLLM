# Backend comparison harness

Runs the same GGUF model and fixed token sequence on Ascend or CUDA. Use
`DotLLM.Tests.Ascend/numpy_reference.py` to generate independent FP32 goldens
from the **decoded source GGUF weights**. This is a developer measurement
program, not a unit-test project or an automatic skip when hardware is absent.

```bash
dotnet build tests/DotLLM.Tests.BackendComparison/DotLLM.Tests.BackendComparison.csproj \
  -c Release -p:Platform="Any CPU"
dotnet path/to/DotLLM.Tests.BackendComparison.dll \
  cuda model.gguf text-cases.json teacher.bin results/cuda all \
  > results-metrics.jsonl
```

Arguments: `device model cases.json teacher.bin output-directory mode`.
Device is `ascend` or `cuda`; modes are `all`, `warm`, `teacher`, `profile`.
Ascend requires the built native bridge, CANN runtime and a working NPU.
CUDA requires a compatible driver, cuBLAS and a working GPU. Missing libraries,
missing devices and execution exceptions return nonzero.

- `warm`: English 1/32/128/512-token full forward, three timed repetitions,
  split cached prefill and rollback/replay comparisons.
- `teacher`: one 8-token prefill and 32 fixed decode tokens for the standard
  fixture. The actual step count is read from `teacher.bin`. Writes FP32 logits
  to `teacher.bin` in the output directory.
- `profile`: one warm-up, one cached 32-token prefill and eight forced decode
  steps. Emits UTC phase markers for alignment with profiler session clocks.
- `all`: `warm` plus `teacher`; model stays loaded throughout.

Timed forward includes managed submission, device work, synchronization and
logits D2H. It excludes sampling and HTTP. Profiling adds overhead. Memory
properties reflect the backend's own accounting; CUDA `AllocatedBytes=0` here
is not evidence of zero device KV allocation.

**Exit code zero means execution completed. Numerical PASS/FAIL is independent.**
Gate a comparison job on every numerical JSON record as well as process exit.
Teacher checks use `RMS(error) <= 0.03 + 0.015*RMS(reference)`,
`max_abs <= 0.25`, finite values and `cosine >= 0.999`.
Split checks use `RMS(error) <= 0.015*RMS(reference)` and `max_abs <= 0.05`;
rollback/replay requires exactly zero max and RMS error.

`run-comparison.sh` runs both standard models in a package prepared using
`DotLLM.Tests.Ascend/run-full.sh` conventions. CUDA also runs a separate Qwen
diagnostic with `COMPARE_ROPE_OVERRIDE=1`. This uses the existing internal
`DebugRopeTypeOverride` property to pass raw kernel RoPE type 1. It is an
explicit diagnostic experiment; it does not modify the production model
source or establish that the baseline CUDA Qwen implementation is correct.
Baseline and diagnostic have distinct output directories.

Examples after preparing the package:

```bash
bash run-comparison.sh cuda /path/to/package compare-followup
bash run-comparison.sh ascend /path/to/package compare-followup
```

Measurements and source references are archived separately on
[archive/ascend-research-20261024](https://github.com/Cheongroach/dotLLM/tree/archive/ascend-research-20261024).
