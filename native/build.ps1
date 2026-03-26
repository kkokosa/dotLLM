# Compile all .cu kernels to PTX for dotLLM CUDA backend.
# Requires: nvcc (CUDA Toolkit) on PATH
# Output: native\ptx\*.ptx
#
# PTX is forward-compatible: compute_61 PTX runs on all GPUs from Pascal onward.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$outDir = Join-Path $scriptDir "ptx"
$kernelDir = Join-Path $scriptDir "kernels"

if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$arch = "compute_61"

Write-Host "Compiling CUDA kernels -> PTX (target: $arch)..."

foreach ($cuFile in Get-ChildItem "$kernelDir\*.cu") {
    $base = $cuFile.BaseName

    & nvcc -ptx -arch=$arch `
        --use_fast_math `
        -o "$outDir\$base.ptx" `
        $cuFile.FullName

    if ($LASTEXITCODE -ne 0) {
        throw "nvcc failed for $($cuFile.Name)"
    }

    Write-Host "  $($cuFile.Name) -> $base.ptx"
}

Write-Host "Done. PTX files in $outDir\"
