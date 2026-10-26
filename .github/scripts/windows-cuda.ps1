# Install the parts of the CUDA toolkit and cuDNN that the Windows build needs,
# from NVIDIA's redistributable archives, into one toolkit-shaped tree.
#
#   windows-cuda.ps1 [-Dest C:\cuda]
#
# Sets CUDA_PATH and CUDNN_PATH (both the tree) and puts nvcc on PATH for the
# later steps of the job. The versions match the ones stemd fetches at run
# time (crates/stemd-server/src/cuda.rs).
param([string]$Dest = 'C:\cuda')
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$cudaVersion = '13.3.1'
$cudnnVersion = '9.14.0'
$components = 'cccl', 'cuda_crt', 'cuda_cudart', 'cuda_nvcc', 'cuda_nvrtc',
    'libcublas', 'libcufft', 'libnvvm'

$cudaRoot = 'https://developer.download.nvidia.com/compute/cuda/redist/'
$cudnnRoot = 'https://developer.download.nvidia.com/compute/cudnn/redist/'
$cuda = Invoke-RestMethod "${cudaRoot}redistrib_$cudaVersion.json"
$cudnn = Invoke-RestMethod "${cudnnRoot}redistrib_$cudnnVersion.json"

$archives = @(foreach ($name in $components) {
    $entry = $cuda.$name.'windows-x86_64'
    if (-not $entry) { throw "no Windows archive for $name in CUDA $cudaVersion" }
    [pscustomobject]@{ Url = $cudaRoot + $entry.relative_path; Sha256 = $entry.sha256 }
})
$entry = $cudnn.cudnn.'windows-x86_64'.cuda13
if (-not $entry) { throw "no Windows CUDA 13 archive for cuDNN $cudnnVersion" }
$archives += [pscustomobject]@{ Url = $cudnnRoot + $entry.relative_path; Sha256 = $entry.sha256 }

$work = Join-Path $env:RUNNER_TEMP 'cuda-archives'
New-Item -ItemType Directory -Force -Path $work, $Dest | Out-Null
foreach ($a in $archives) {
    $zip = Join-Path $work (Split-Path $a.Url -Leaf)
    Invoke-WebRequest $a.Url -OutFile $zip
    $got = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    if ($got -ne $a.Sha256) { throw "$zip is $got, expected $($a.Sha256)" }
    & 7z x -y "-o$work" $zip | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "7z failed on $zip" }
    $dir = Join-Path $work ([IO.Path]::GetFileNameWithoutExtension($zip))
    Get-ChildItem $dir -Exclude LICENSE | Copy-Item -Destination $Dest -Recurse -Force
    Remove-Item $zip, $dir -Recurse -Force
}

# cuDNN keeps its DLLs in bin\; mlx-sys and MLX's FindCUDNN expect bin\x64.
New-Item -ItemType Directory -Force -Path (Join-Path $Dest 'bin\x64') | Out-Null
Get-ChildItem (Join-Path $Dest 'bin\cudnn*.dll') | Move-Item -Destination (Join-Path $Dest 'bin\x64') -Force

"CUDA_PATH=$Dest" | Out-File $env:GITHUB_ENV -Append -Encoding utf8
"CUDNN_PATH=$Dest" | Out-File $env:GITHUB_ENV -Append -Encoding utf8
(Join-Path $Dest 'bin') | Out-File $env:GITHUB_PATH -Append -Encoding utf8
& (Join-Path $Dest 'bin\nvcc.exe') --version | Select-Object -Last 2
