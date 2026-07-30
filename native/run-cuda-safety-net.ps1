param(
    [ValidateSet("all", "memcheck", "initcheck", "racecheck")]
    [string]$Tool = "all",
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [switch]$FullSanitizer
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$project = "tests\DotLLM.Tests.Unit\DotLLM.Tests.Unit.csproj"
$coreKernelFilter = "Category=GPU&FullyQualifiedName~DotLLM.Tests.Unit.Cuda.CudaKernelTests"
$comparisonKernelFilter = "Category=GPU&FullyQualifiedName~DotLLM.Tests.Unit.Cuda.CudaKernelComparisonTests"
$parityFilter = "Category=GPU&(FullyQualifiedName~DotLLM.Tests.Unit.Cuda.CudaKernelTests|FullyQualifiedName~DotLLM.Tests.Unit.Cuda.CudaKernelComparisonTests)"
$memcheckFilters = @($coreKernelFilter, $comparisonKernelFilter)
$followUpFilters = if ($FullSanitizer) { @($coreKernelFilter, $comparisonKernelFilter) } else { @($coreKernelFilter) }

Write-Host "CUDA safety net parity pass"
dotnet test $project -c $Configuration --filter $parityFilter

function Invoke-Sanitizer {
    param(
        [Parameter(Mandatory = $true)][string]$SanitizerTool,
        [Parameter(Mandatory = $true)][string]$Filter
    )

    Write-Host "compute-sanitizer --tool $SanitizerTool filter: $Filter"
    $sanitizerArgs = @(
        "--tool", $SanitizerTool,
        "--target-processes", "all",
        "--error-exitcode", "1",
        "dotnet", "test", $project,
        "-c", $Configuration,
        "--no-build",
        "--filter", $Filter
    )
    & compute-sanitizer @sanitizerArgs
}

if ($Tool -in @("all", "memcheck")) {
    foreach ($filter in $memcheckFilters) {
        Invoke-Sanitizer -SanitizerTool memcheck -Filter $filter
    }
}

if ($Tool -in @("all", "initcheck")) {
    foreach ($filter in $followUpFilters) {
        Invoke-Sanitizer -SanitizerTool initcheck -Filter $filter
    }
}

if ($Tool -in @("all", "racecheck")) {
    foreach ($filter in $followUpFilters) {
        Invoke-Sanitizer -SanitizerTool racecheck -Filter $filter
    }
}
