# Separate a short generated mix on a machine without a GPU: the server starts,
# runs on the CPU and returns stems.
#
#   cpu-separation.ps1 -Server stemd-server.exe -Cli stemd-cli.exe
param(
    [Parameter(Mandatory)] [string]$Server,
    [Parameter(Mandatory)] [string]$Cli
)
$ErrorActionPreference = 'Stop'

$work = Join-Path $env:RUNNER_TEMP 'cpu-separation'
New-Item -ItemType Directory -Force -Path $work | Out-Null
python (Join-Path $PSScriptRoot 'test-mix.py') "$work\mix.wav"
if ($LASTEXITCODE -ne 0) { throw 'test-mix.py failed' }

$out = "$work\server.out"
$err = "$work\server.err"
$proc = Start-Process $Server -PassThru -NoNewWindow `
    -ArgumentList '--headless', '--no-mdns', '--bind', '127.0.0.1:7979', '--cache-dir', "$work\cache" `
    -RedirectStandardOutput $out -RedirectStandardError $err
function Get-Log { (Get-Content $out, $err -ErrorAction SilentlyContinue) -join "`n" }
function Stop-WithLog([string]$why) {
    Get-Log | Write-Host
    if (-not $proc.HasExited) { Stop-Process $proc -Force }
    throw $why
}

$deadline = (Get-Date).AddMinutes(10)
while ((Get-Log) -notmatch 'listening') {
    if ($proc.HasExited) { Stop-WithLog "the server exited with 0x$('{0:X8}' -f $proc.ExitCode)" }
    if ((Get-Date) -gt $deadline) { Stop-WithLog 'the server is not listening after 10 minutes' }
    Start-Sleep -Seconds 1
}
if ((Get-Log) -notmatch 'via mlx on cpu') { Stop-WithLog 'the server is not on the CPU' }

& $Cli --host 127.0.0.1:7979 --out "$work\stems" "$work\mix.wav"
if ($LASTEXITCODE -ne 0) { Stop-WithLog 'the separation failed' }
Stop-Process $proc -Force
Get-ChildItem "$work\stems"
