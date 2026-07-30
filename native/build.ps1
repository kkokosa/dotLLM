# Compile all .cu kernels to PTX for dotLLM CUDA backend.
# Requires: nvcc (CUDA Toolkit) on PATH
# Output: native\ptx\*.ptx
#
# PTX is forward-compatible: compute_61 PTX runs on all GPUs from Pascal onward.
#
# ── Arch-tiered PTX (optional) ────────────────────────────────────────────────
# In addition to the universal compute_61 "<kernel>.ptx" (always emitted), this
# script can OPTIONALLY emit higher-arch PTX variants named "<kernel>.sm_<arch>.ptx"
# for a curated subset of kernels. The runtime loader (CudaModule.LoadForArch)
# picks the highest-arch variant whose arch is <= the device compute capability,
# and falls back to the compute_61 "<kernel>.ptx" when no variant is present.
#
# This is opt-in. With no parameters the script produces EXACTLY today's output:
# only compute_61 "<kernel>.ptx" files. To also emit higher-arch variants:
#
#   ./build.ps1 -ExtraArchs 80,86
#   ./build.ps1 -ExtraArchs 80 -ExtraArchKernels quantized_gemv
#
# -ExtraArchs         SM numbers (e.g. 75,80,86,90). Empty = none.
# -ExtraArchKernels   kernel base names to also build for -ExtraArchs. Defaults to
#                     $archTieredKernels below. Only kernels with a genuinely
#                     arch-specific implementation belong here; none exist yet, so
#                     the default list is empty (true no-op).

param(
    [int[]] $ExtraArchs = @(),
    [string[]] $ExtraArchKernels = $null
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$outDir = Join-Path $scriptDir "ptx"
$kernelDir = Join-Path $scriptDir "kernels"

if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$arch = "compute_61"

# Curated kernel list eligible for higher-arch variants. Empty until an
# arch-specific kernel implementation actually exists.
$archTieredKernels = @()
if ($null -eq $ExtraArchKernels) { $ExtraArchKernels = $archTieredKernels }

Write-Host "Compiling CUDA kernels -> PTX (target: $arch)..."

foreach ($cuFile in Get-ChildItem "$kernelDir\*.cu") {
    $base = $cuFile.BaseName

    # Universal compute_61 PTX — always emitted (today's behavior).
    & nvcc -ptx -arch=$arch `
        --use_fast_math `
        -o "$outDir\$base.ptx" `
        $cuFile.FullName

    if ($LASTEXITCODE -ne 0) {
        throw "nvcc failed for $($cuFile.Name)"
    }

    Write-Host "  $($cuFile.Name) -> $base.ptx"

    # Optional higher-arch variants for the curated kernel list.
    if ($ExtraArchs.Count -gt 0 -and $ExtraArchKernels -contains $base) {
        foreach ($sm in $ExtraArchs) {
            & nvcc -ptx -arch="compute_$sm" `
                --use_fast_math `
                -o "$outDir\$base.sm_$sm.ptx" `
                $cuFile.FullName

            if ($LASTEXITCODE -ne 0) {
                throw "nvcc failed for $($cuFile.Name) (sm_$sm)"
            }

            Write-Host "  $($cuFile.Name) -> $base.sm_$sm.ptx (arch-tiered)"
        }
    }
}

Write-Host "Done. PTX files in $outDir\"
