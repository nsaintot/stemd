# Assemble the Windows build.
#
# Twelve files, about 134 MB, and no CUDA in any of them. The executable
# delay-loads its CUDA imports and decides at startup whether they resolve, so
# this same directory runs on a machine with no NVIDIA anything and uses a card
# where there is a usable one. `install-cuda.cmd` beside it fetches the runtime
# for whoever wants it.
#
# That replaces staging CUDA's bin\x64 wholesale, which is where the 2.4 GB
# directory came from: half of it was libraries nothing in the program can call,
# and the rest is now a download rather than a precondition.
#
# What has to be carried is the MSVC runtime, because a machine without Visual
# Studio has no reason to have it, and libopenblas, which MLX builds as a DLL
# and cargo has no install rule for. Both are hard imports: leave either out and
# the process dies at 0xC0000135 before `main`, with no message.
[CmdletBinding()]
param(
    [string]$Out,
    # The CLI is a separate deliverable and the window does not need it.
    [switch]$WithCli
)
$ErrorActionPreference = 'Stop'

# Not a param default: $PSScriptRoot is not reliably populated while those are
# being bound, and Join-Path on an empty path fails before the body runs.
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
if (-not $Out) { $Out = Join-Path $root 'dist\stemd-windows' }

# CUDA is opt-in in mlx-sys, and silent when it is off. `build.rs` turns it on
# for the `cuda` feature or for MLX_BUILD_CUDA; without either it builds a
# CPU-only MLX, links a 21 MB executable instead of an 84 MB one, and produces a
# server that starts, says `device: cpu`, separates correctly and takes 610
# seconds over a 64 second clip that the card does in two. Nothing in the build
# says anything is wrong, because from cargo's point of view nothing is.
#
# Set here rather than expected from the environment. An interactive shell that
# happens to have it and an ssh session that does not are the same command
# producing two different products, which is exactly how a CPU-only binary got
# built on this machine and staged for testing.
$env:MLX_BUILD_CUDA = '1'
if (-not $env:MLX_CUDA_ARCHITECTURES) { $env:MLX_CUDA_ARCHITECTURES = '86' }

# cuDNN is the other thing the build needs from the environment and the other
# thing nobody has persistently: on the machine this was developed on,
# CUDNN_PATH was set in a shell and in neither the Machine nor the User scope,
# so the build worked in that window and nowhere else. Unset, it gets sixty
# lines into a CMake trace before saying
#
#   FindCUDNN.cmake:56: Unable to find cudnn.h, please make sure cuDNN is
#   installed and pass CUDNN_INCLUDE_PATH to cmake
#
# which is true and does not mention the variable that would fix it.
#
# The installer's own location is probed because that is where most machines
# have it. Anywhere else has to be said, and saying it is one line.
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

# cargo writes its progress to stderr, and under `Stop` a *redirected* native
# stderr becomes a terminating error. So this script ran fine by hand and died
# on cargo's first "Compiling" line the moment anyone piped it to a log, which
# is what a build kicked off over ssh has to do. Exit codes are what actually
# say whether a native command failed, and they are checked either way.
function Invoke-Native {
    param([string]$What, [scriptblock]$Command)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Command } finally { $ErrorActionPreference = $previous }
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

Write-Host "building release binaries (CUDA on, sm_$($env:MLX_CUDA_ARCHITECTURES))..."
Invoke-Native 'cargo build -p stemd-server' {
    cargo build --release --manifest-path (Join-Path $root 'Cargo.toml') -p stemd-server
}
if ($WithCli) {
    Invoke-Native 'cargo build -p stemd-cli' {
        cargo build --release --manifest-path (Join-Path $root 'Cargo.toml') -p stemd-cli
    }
}

# And check it took, rather than trusting the variable. The delay-loaded CUDA
# imports are the difference between the two builds, so ask the executable
# instead of asking the environment that made it.
#
# By reading the bytes rather than by running dumpbin. Delay-load descriptors
# hold their DLL names as plain ASCII in the PE, so this needs no tooling at
# all, and dumpbin needs a Visual Studio developer prompt: it was absent over
# ssh, which is precisely the environment this check exists for. A guard that
# excuses itself in the case it was written for is not a guard.
# The names are the ones MLX imports, minus their version suffixes, and
# deliberately not nvcuda.dll: that is the driver, `stemd_mlx::device` names it
# to ask whether a card is present, and it is therefore in the CPU-only build
# too. Picking it would have been a check that passes on exactly the binary it
# exists to reject.
$exe = Join-Path $root 'target\release\stemd-server.exe'
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

$release = Join-Path $root 'target\release'
Remove-Item $Out -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $Out | Out-Null

Copy-Item (Join-Path $release 'stemd-server.exe') $Out
if ($WithCli) { Copy-Item (Join-Path $release 'stemd-cli.exe') $Out }

# MLX builds OpenBLAS as a DLL inside mlx-sys's OUT_DIR and nothing installs it.
$blas = Get-ChildItem (Join-Path $release 'build') -Directory -Filter 'mlx-sys-*' |
    ForEach-Object { Join-Path $_.FullName 'out\build\libopenblas.dll' } |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $blas) { throw 'no libopenblas.dll under target\release\build\mlx-sys-*; build stemd-server first' }
Copy-Item $blas $Out

# The MSVC runtime, from the redistributable directory the toolchain ships.
# Named rather than globbed: the redist folder also holds the debug CRT, which
# must never be shipped and will not load on a machine without Visual Studio.
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

# Double-clickable, because the log line that asks for this is read by someone
# looking at a window, and telling them to pass a flag is telling them to open a
# terminal first.
@'
@echo off
rem Fetch the CUDA runtime beside stemd-server.exe. About 1.2 GB, once, and
rem only useful on a machine with an NVIDIA card. Refuses in seconds if the
rem card already works or if there is no driver at all.
"%~dp0stemd-server.exe" --install-cuda
pause
'@ | Set-Content (Join-Path $Out 'install-cuda.cmd') -Encoding ascii

# A CUDA library in here means the delay-load flags stopped working and the
# directory has quietly become the 2.4 GB one again.
$cuda = Get-ChildItem $Out -Filter '*.dll' |
    Where-Object { $_.Name -match '^(cu|nv|npp)' }
if ($cuda) { throw "CUDA libraries in the base bundle: $($cuda.Name -join ', ')" }

$size = (Get-ChildItem $Out -Recurse | Measure-Object -Property Length -Sum)
Write-Host ''
Write-Host "done: $Out"
Write-Host ("  {0} files, {1:N1} MB" -f $size.Count, ($size.Sum / 1MB))
Write-Host '  no CUDA: runs on any machine, uses a card where one is usable'
Write-Host '  install-cuda.cmd adds the runtime on a machine with one'
