# Put Ninja and ccache on PATH for the later steps of the job.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

python -m pip install --quiet ninja==1.13.2
if ($LASTEXITCODE -ne 0) { throw 'pip install ninja failed' }

$ccache = 'ccache-4.14.2-windows-x86_64'
$zip = Join-Path $env:RUNNER_TEMP "$ccache.zip"
Invoke-WebRequest "https://github.com/ccache/ccache/releases/download/v4.14.2/$ccache.zip" -OutFile $zip
& 7z x -y "-o$env:RUNNER_TEMP" $zip | Out-Null
if ($LASTEXITCODE -ne 0) { throw "7z failed on $zip" }
(Join-Path $env:RUNNER_TEMP $ccache) | Out-File $env:GITHUB_PATH -Append -Encoding utf8
