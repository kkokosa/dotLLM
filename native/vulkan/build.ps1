# Compile all GLSL compute shaders to SPIR-V for the dotLLM Vulkan backend.
# Requires: glslc (ships with the Vulkan SDK) on PATH.
# Output: native\vulkan\spv\*.spv
#
# End users do NOT need the Vulkan SDK — .spv blobs ship alongside the .NET
# assembly and are loaded verbatim at runtime. Only shader *authors* need glslc.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$outDir    = Join-Path $scriptDir "spv"
$shaderDir = Join-Path $scriptDir "shaders"

if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$targetEnv = "vulkan1.2"

Write-Host "Compiling GLSL compute shaders -> SPIR-V (target: $targetEnv)..."

foreach ($compFile in Get-ChildItem "$shaderDir\*.comp") {
    $base = $compFile.BaseName

    & glslc --target-env=$targetEnv -o "$outDir\$base.spv" $compFile.FullName

    if ($LASTEXITCODE -ne 0) {
        throw "glslc failed for $($compFile.Name)"
    }

    Write-Host "  $($compFile.Name) -> $base.spv"
}

Write-Host "Done. SPIR-V files in $outDir\"
