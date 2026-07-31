@echo off
REM Build all CUDA kernels to PTX.
REM Requires: %CUDA_PATH% set to a CUDA toolkit that supports the host MSVC
REM (CUDA 13.x supports VS 2022/2026 MSVC; CUDA 11.8 does not).
REM Usage: build_ptx.bat [arch]     (default: compute_75)
REM
REM compute_75 = Turing, the CUDA 13 floor. PTX is forward-compatible so this
REM runs on any Turing (SM 7.5), Ampere (8.0/8.6), Ada (8.9), Hopper (9.0),
REM or Blackwell (10.0/12.0) GPU. CUDA 13 dropped Pascal/Volta (SM 6.x/7.0).
setlocal EnableDelayedExpansion

set ARCH=%1
if "%ARCH%"=="" set ARCH=compute_75

if not defined CUDA_PATH (
    echo CUDA_PATH is not set. Install a CUDA toolkit and ensure CUDA_PATH points at it.
    exit /b 1
)
set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
if not exist "%NVCC%" (
    echo nvcc.exe not found at "!NVCC!"
    exit /b 1
)

REM Locate a CUDA-compatible MSVC host compiler. Resolution order:
REM   1. MSVC_DIR if pre-set (a VC\Tools\MSVC\<version> directory) — explicit override.
REM   2. cl.exe already on PATH — i.e. running from a Developer Command Prompt.
REM   3. vswhere.exe (ships with the Visual Studio Installer) — enumerate every VS
REM      instance with the C++ toolset and pick the newest CUDA-compatible one.
REM MSVC 14.5x (VS 2026, _MSC_VER >= 1950) is rejected by CUDA 13.1's host_config.h
REM and by nvcc's OS-target check, so a 14.3x/14.4x toolset is preferred when both
REM are installed; a 14.5x toolset is used only as a last resort (with a warning).
if defined MSVC_DIR goto :have_msvc

where cl.exe >nul 2>&1
if not errorlevel 1 (
    echo Using cl.exe already on PATH.
    goto :msvc_ready
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo Could not find vswhere.exe, cl.exe on PATH, or a pre-set MSVC_DIR.
    echo Install the "Desktop development with C++" workload, run this from a
    echo Developer Command Prompt, or set MSVC_DIR to a VC\Tools\MSVC\^<version^> directory.
    exit /b 1
)

set "MSVC_COMPAT="
set "MSVC_ANY="
for /f "usebackq delims=" %%I in (`"%VSWHERE%" -products * -latest -prerelease -sort -format value -property installationPath 2^>nul`) do (
    call :scan_instance "%%I"
)
for /f "usebackq delims=" %%I in (`"%VSWHERE%" -products * -all -prerelease -format value -property installationPath 2^>nul`) do (
    call :scan_instance "%%I"
)
if defined MSVC_COMPAT (
    set "MSVC_DIR=!MSVC_COMPAT!"
) else if defined MSVC_ANY (
    echo WARNING: only MSVC 14.5x+ found; CUDA 13.x host_config.h rejects it.
    echo          Relying on -allow-unsupported-compiler; install the VS 2022 ^(14.3x/14.4x^) toolset if this fails.
    set "MSVC_DIR=!MSVC_ANY!"
) else (
    echo Could not locate a VC Tools install via vswhere.
    exit /b 1
)

:have_msvc
REM Install paths routinely contain "(x86)", so every expansion inside a
REM parenthesised block below must be delayed (!VAR!) — %VAR% would splice the
REM closing paren into the block and abort the parse.
set "MSVC_BIN=%MSVC_DIR%\bin\Hostx64\x64"
if not exist "%MSVC_BIN%\cl.exe" (
    echo cl.exe not found at "!MSVC_BIN!"
    exit /b 1
)
set "PATH=%MSVC_BIN%;%PATH%"
echo Using MSVC: %MSVC_DIR%

:msvc_ready

set "SCRIPT_DIR=%~dp0"
set "KERNEL_DIR=%SCRIPT_DIR%kernels"
set "OUT_DIR=%SCRIPT_DIR%ptx"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

REM Kernels safe under --use_fast_math (elementwise; no expf/rsqrtf/sin/cos/pow):
set "FAST_MATH=add add_f32 swiglu swiglu_f32 convert bias_add bias_add_f32 embedding embedding_f32out dequant quant_kv"

echo Using nvcc: %NVCC%
echo Compiling CUDA kernels -^> PTX (target: %ARCH%)...

set FAIL=0
for %%F in ("%KERNEL_DIR%\*.cu") do (
    set "BASE=%%~nF"
    set "FAST_FLAG="
    for %%M in (%FAST_MATH%) do (
        if /I "%%~nF"=="%%M" set "FAST_FLAG=--use_fast_math"
    )
    "!NVCC!" -ptx -arch=%ARCH% !FAST_FLAG! -allow-unsupported-compiler -o "!OUT_DIR!\!BASE!.ptx" "%%F"
    if errorlevel 1 (
        echo FAILED: %%~nxF
        set FAIL=1
    ) else (
        if defined FAST_FLAG (
            echo   %%~nxF -^> !BASE!.ptx ^(fast_math^)
        ) else (
            echo   %%~nxF -^> !BASE!.ptx ^(precise^)
        )
    )
)

if "%FAIL%"=="1" exit /b 1
echo Done. PTX files in %OUT_DIR%
exit /b 0

REM ---------------------------------------------------------------------------
REM :scan_instance <visual-studio-installationPath>
REM Records the newest usable VC toolset found under the instance:
REM   MSVC_COMPAT — newest toolset older than 14.50 (CUDA 13.x compatible)
REM   MSVC_ANY    — newest toolset of any version (last-resort fallback)
REM ---------------------------------------------------------------------------
:scan_instance
set "VC_ROOT=%~1\VC\Tools\MSVC"
if not exist "%VC_ROOT%\" exit /b 0
for /d %%D in ("%VC_ROOT%\*") do (
    if exist "%%D\bin\Hostx64\x64\cl.exe" (
        if "%%~nxD" GTR "!MSVC_ANY_VER!" (
            set "MSVC_ANY_VER=%%~nxD"
            set "MSVC_ANY=%%D"
        )
        for /f "tokens=1,2 delims=." %%a in ("%%~nxD") do (
            set /a "TOOLSET_MAJOR=%%a, TOOLSET_MINOR=%%b" 2>nul
            if !TOOLSET_MAJOR! LEQ 14 if !TOOLSET_MINOR! LSS 50 (
                if "%%~nxD" GTR "!MSVC_COMPAT_VER!" (
                    set "MSVC_COMPAT_VER=%%~nxD"
                    set "MSVC_COMPAT=%%D"
                )
            )
        )
    )
)
exit /b 0
