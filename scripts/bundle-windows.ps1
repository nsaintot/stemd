# Assemble the Windows build.
#
# It is about 165 MB and holds no CUDA library. The executable delay-loads its
# CUDA imports and decides at startup whether they resolve, so the same
# directory runs on a machine with no NVIDIA hardware and uses a usable card
# where there is one. `install-cuda.cmd` beside it downloads the runtime on
# request. Headers do ship: MLX compiles some kernels at run time with NVRTC,
# and those include CCCL and the CUDA runtime headers, which must be on disk
# # on every machine, because any GPU separation can trigger a run-time compile.
#
# Do not retry copying CUDA's bin\x64 whole: it makes a 2.4 GB directory, half
# of it libraries the program cannot call, and the rest is an optional download.
#
# Two things must ship: the MSVC runtime, which a machine without Visual Studio
# may lack, and libopenblas, which MLX builds as a DLL and cargo has no install
# rule for. Both are hard imports: without either, the process dies with
# 0xC0000135 before `main`, with no message.
[CmdletBinding()]
param(
    [string]$Out,
    # The CLI is a separate deliverable and the window does not need it.
    [switch]$WithCli
)
$ErrorActionPreference = 'Stop'

# Set here, outside the param defaults: $PSScriptRoot is not reliably populated
# while those are bound, and Join-Path on an empty path fails before the body
# runs.
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
if (-not $Out) { $Out = Join-Path $root 'dist\stemd-windows' }

# CUDA is opt-in in mlx-sys, and silent when off. `build.rs` turns it on for the
# `cuda` feature or for MLX_BUILD_CUDA; without either it builds a CPU-only MLX,
# links a 21 MB executable where the CUDA build is 84 MB, and produces a server
# that starts, reports `device: cpu`, separates correctly and takes 610 seconds
# over a 64-second clip that the card separates in a few seconds. Cargo reports
# no error, because the CPU-only build is valid.
#
# It is set here, without relying on the environment: an interactive shell that
# has it and an ssh session that does not would build two different products
# from the same command.
$env:MLX_BUILD_CUDA = '1'
if (-not $env:MLX_CUDA_ARCHITECTURES) { $env:MLX_CUDA_ARCHITECTURES = '86' }

# cuDNN is the other input the build takes from the environment, and it is often
# set in only one shell, in neither the Machine nor the User scope. Unset, the
# build gets sixty lines into a CMake trace before saying
#
#   FindCUDNN.cmake:56: Unable to find cudnn.h, please make sure cuDNN is
#   installed and pass CUDNN_INCLUDE_PATH to cmake
#
# which does not mention the variable that would fix it.
#
# The installer's default location is probed because most machines have cuDNN
# there. Any other location must be given, in one line.
if (-not $env:CUDNN_PATH) {
    $found = Get-ChildItem 'C:\Program Files\NVIDIA\CUDNN' -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Where-Object { Test-Path (Join-Path $_.FullName 'include\cudnn.h') } |
        Select-Object -First 1
    if ($found) { $env:CUDNN_PATH = $found.FullName }
}
if (-not $env:CUDNN_PATH -or -not (Test-Path (Join-Path $env:CUDNN_PATH 'include\cudnn.h'))) {
    throw "cuDNN not found. MLX needs it to build the CUDA backend, and nothing " +
          "on this machine says where it is. Set CUDNN_PATH to the root of a " +
          "toolkit-shaped tree, the directory holding include\cudnn.h and " +
          "lib\x64, and run this again."
}
Write-Host "  cuDNN: $($env:CUDNN_PATH)"

# cargo writes its progress to stderr, and under `Stop` a redirected native
# stderr becomes a terminating error, so the script would die on cargo's first
# "Compiling" line whenever its output is piped to a log, as a build over ssh
# must be. Exit codes say whether a native command failed, and they are checked
# either way.
function Invoke-Native {
    param([string]$What, [scriptblock]$Command)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Command } finally { $ErrorActionPreference = $previous }
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

# Source paths go into the binary, in every panic message and in the debug info,
# as this machine's absolute paths. They are rewritten to names that say what
# each file is and nothing about the disk it was built on.
$cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE '.cargo' }
# The source trees are remapped, never the whole checkout: a prefix covering
# target\ stops rustc from finding the proc-macro crates it just built there.
$sysroot = (& rustc --print sysroot).Trim()
$env:RUSTFLAGS = ("$($env:RUSTFLAGS) --remap-path-prefix=$root\crates=stemd\crates " +
    "--remap-path-prefix=$root\vendor=stemd\vendor --remap-path-prefix=$cargoHome=cargo " +
    "--remap-path-prefix=$sysroot=rustc").Trim()

Write-Host "building release binaries (CUDA on, sm_$($env:MLX_CUDA_ARCHITECTURES))..."
Invoke-Native 'cargo build -p stemd-server' {
    cargo build --release --manifest-path (Join-Path $root 'Cargo.toml') -p stemd-server
}
if ($WithCli) {
    Invoke-Native 'cargo build -p stemd-cli' {
        cargo build --release --manifest-path (Join-Path $root 'Cargo.toml') -p stemd-cli
    }
}

# Check that CUDA was built in, without trusting the variable. The delay-loaded
# CUDA imports are the difference between the two builds, so this reads the
# executable itself.
#
# It reads the bytes and does not run dumpbin. Delay-load descriptors hold their
# DLL names as plain ASCII in the PE, so this needs no tooling, while dumpbin
# needs a Visual Studio developer prompt, which an ssh session lacks, and ssh
# builds are what this check is for.
#
# The names are the ones MLX imports, without their version suffixes. nvcuda.dll
# is left out on purpose: it is the driver, and `stemd_mlx::device` names it to
# ask whether a card is present, so it is in the CPU-only build too, and
# checking for it would pass on the binary this check exists to reject.
$target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $root 'target' }
$release = Join-Path $target 'release'
$exe = Join-Path $release 'stemd-server.exe'
$text = [System.Text.Encoding]::ASCII.GetString([System.IO.File]::ReadAllBytes($exe))
$wanted = 'cublasLt64', 'cufft64', 'nvrtc64'
$missing = $wanted | Where-Object { $text -notmatch $_ }
if ($missing) {
    throw "$exe names no $($missing -join ', '): it was linked against a " +
          "CPU-only MLX and would ship as a server that starts, reports " +
          "device: cpu, separates correctly and runs about 190x slower. Check " +
          "MLX_BUILD_CUDA and CUDNN_PATH, delete " +
          "target\release\build\mlx-sys-*, and build again."
}
Write-Host "  CUDA imports present ($($wanted -join ', '))"

Remove-Item $Out -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $Out | Out-Null

Copy-Item (Join-Path $release 'stemd-server.exe') $Out
if ($WithCli) { Copy-Item (Join-Path $release 'stemd-cli.exe') $Out }

# MLX builds OpenBLAS as a DLL inside mlx-sys's OUT_DIR and nothing installs it.
# A prebuilt MLX carries it in its lib directory.
$blas = if ($env:MLX_PREBUILT_DIR) {
    Join-Path $env:MLX_PREBUILT_DIR 'lib\libopenblas.dll'
} else {
    Get-ChildItem (Join-Path $release 'build') -Directory -Filter 'mlx-sys-*' |
        ForEach-Object { Join-Path $_.FullName 'out\build\libopenblas.dll' } |
        Where-Object { Test-Path $_ } | Select-Object -First 1
}
if ($blas -and -not (Test-Path $blas)) { $blas = $null }
if (-not $blas) { throw 'no libopenblas.dll in the prebuilt MLX or under target\release\build\mlx-sys-*; build stemd-server first' }
Copy-Item $blas $Out

# The MSVC runtime, from the redistributable directory the toolchain ships.
# The files are named, never globbed: the redist folder also holds the debug
# CRT, which must never ship and does not load on a machine without Visual
# Studio.
$needed = @(
    'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
    'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll',
    'vcruntime140.dll', 'vcruntime140_1.dll', 'vcruntime140_threads.dll',
    'concrt140.dll', 'vccorlib140.dll'
)
$redist = Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT' `
    -Directory -ErrorAction SilentlyContinue | Sort-Object FullName | Select-Object -Last 1
if (-not $redist) { throw 'no VC redistributable directory found; is the MSVC toolchain installed?' }
$missing = @()
foreach ($dll in $needed) {
    $src = Join-Path $redist.FullName $dll
    if (Test-Path $src) { Copy-Item $src $Out } else { $missing += $dll }
}
if ($missing) { Write-Warning "not in $($redist.FullName): $($missing -join ', ')" }

# A double-clickable script, because the log line that asks for it is read by
# someone looking at a window, who should not have to open a terminal to pass a
# flag.
@'
@echo off
rem Fetch the CUDA runtime beside stemd-server.exe. About 1.2 GB, once, and
rem only useful on a machine with an NVIDIA card. Refuses in seconds if the
rem card already works or if there is no driver at all.
"%~dp0stemd-server.exe" --install-cuda
pause
'@ | Set-Content (Join-Path $Out 'install-cuda.cmd') -Encoding ascii

# The headers NVRTC needs at run time. MLX looks for CCCL at include\cccl beside
# the executable, and for the runtime headers under CUDA_HOME, which the server
# points at cuda\ beside itself. Without them, every GPU separation fails on its
# first kernel with "cannot open source file cuda/std/tuple", on a machine that
# started fine and reported device: gpu.
$toolkit = $env:CUDA_PATH
if (-not $toolkit) {
    $toolkit = Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName 'include\cuda.h') } |
        Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $toolkit -or -not (Test-Path (Join-Path $toolkit 'include\cccl\cuda\std\tuple'))) {
    throw "no CCCL headers under the CUDA toolkit ($toolkit). MLX compiles kernels at " +
          "run time and they include <cuda/std/tuple>; set CUDA_PATH to a toolkit " +
          "that has include\cccl."
}
New-Item -ItemType Directory -Force -Path (Join-Path $Out 'include\cccl'), (Join-Path $Out 'cuda\include') | Out-Null
# Only the cuda tree is included by the run-time sources; cub and thrust are
# build-time dependencies and stay out.
foreach ($tree in 'cuda', 'nv') {
    $src = Join-Path $toolkit "include\cccl\$tree"
    if (Test-Path $src) { Copy-Item $src (Join-Path $Out 'include\cccl') -Recurse -Force }
}
Get-ChildItem (Join-Path $toolkit 'include') | Where-Object { $_.Name -ne 'cccl' } |
    Copy-Item -Destination (Join-Path $Out 'cuda\include') -Recurse -Force
$hdrs = (Get-ChildItem (Join-Path $Out 'include'), (Join-Path $Out 'cuda') -Recurse -File | Measure-Object).Count
Write-Host "  headers: $hdrs files for the run-time kernel compiler"

# A CUDA library here means the delay-load flags stopped working and the
# directory has grown back to 2.4 GB.
$cuda = Get-ChildItem $Out -Filter '*.dll' |
    Where-Object { $_.Name -match '^(cu|nv|npp)' }
if ($cuda) { throw "CUDA libraries in the base bundle: $($cuda.Name -join ', ')" }

$size = (Get-ChildItem $Out -Recurse | Measure-Object -Property Length -Sum)
Write-Host ''
Write-Host "done: $Out"
Write-Host ("  {0} files, {1:N1} MB" -f $size.Count, ($size.Sum / 1MB))
Write-Host '  no CUDA libraries: runs on any machine, uses a card where one is usable'
Write-Host '  install-cuda.cmd adds the runtime on a machine with one'
