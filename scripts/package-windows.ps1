# Turn the staged Windows bundle into the two things people download.
#
# `bundle-windows.ps1` produces the directory; this wraps it into an installer
# and a zip. Both, because they answer different questions: the installer is
# what somebody who wants the program double-clicks, and the zip is for whoever
# would rather not run an installer at all, or wants it on a stick.
#
# Needs NSIS. `winget install NSIS.NSIS` puts makensis where this looks.
[CmdletBinding()]
param(
    # The staged bundle. Built first when it is not there.
    [string]$Source,
    [string]$Out,
    # Skip the build and package what is already staged.
    [switch]$NoBuild
)
$ErrorActionPreference = 'Stop'

# Defaults here rather than in the param block, because $PSScriptRoot is not
# reliably populated while those are being bound: it comes out empty under some
# ways of invoking a script, and Join-Path on an empty path is a hard error
# before the first line of the body runs.
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Source) { $Source = Join-Path $root 'dist\stemd-windows' }
if (-not $Out)    { $Out    = Join-Path $root 'dist' }

function Say($m) { Write-Host "  $m" }

# The version, from the one place it is declared. Read rather than passed, so an
# installer can never be named for a version the binaries in it are not.
$manifest = Get-Content (Join-Path $root 'Cargo.toml') -Raw
if ($manifest -notmatch '(?m)^version\s*=\s*"(\d+\.\d+\.\d+)"') {
    throw "no workspace version in Cargo.toml"
}
$version = $Matches[1]

if (-not $NoBuild) {
    & powershell -NoProfile -ExecutionPolicy Bypass `
        -File (Join-Path $root 'scripts\bundle-windows.ps1') -WithCli -Out $Source
    if ($LASTEXITCODE -ne 0) { throw "bundle-windows.ps1 failed with exit code $LASTEXITCODE" }
}
if (-not (Test-Path (Join-Path $Source 'stemd-server.exe'))) {
    throw "no stemd-server.exe in $Source; run bundle-windows.ps1 first, or drop -NoBuild"
}

# The licences travel with the binaries, not just with the source. Both, because
# the grant is either at the reader's choice.
foreach ($name in 'LICENSE-MIT', 'LICENSE-APACHE') {
    Copy-Item (Join-Path $root $name) (Join-Path $Source "$name.txt") -Force
}

$makensis = (Get-Command makensis -ErrorAction SilentlyContinue).Source
if (-not $makensis) {
    $makensis = Get-ChildItem 'C:\Program Files*\NSIS\makensis.exe' -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $makensis) {
    throw "makensis not found. NSIS builds the installer: winget install NSIS.NSIS, then run this again."
}
Say "nsis: $makensis"

New-Item -ItemType Directory -Force -Path $Out | Out-Null
$installer = Join-Path (Resolve-Path $Out) "stemd-$version-setup.exe"
$icon      = Join-Path $root 'resources\stemd.ico'
if (-not (Test-Path $icon)) { throw "no $icon; run scripts/make-icons.sh" }

# makensis writes its progress to stdout and its complaints to stderr, and under
# `Stop` a redirected native stderr is a terminating error. Same trap as cargo.
$previous = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $makensis /V2 `
        "/DVERSION=$version" `
        "/DSOURCE=$((Resolve-Path $Source).Path)" `
        "/DOUTFILE=$installer" `
        "/DICON=$icon" `
        (Join-Path $root 'scripts\installer-windows.nsi')
} finally { $ErrorActionPreference = $previous }
if ($LASTEXITCODE -ne 0) { throw "makensis failed with exit code $LASTEXITCODE" }

# The same staged files, for anyone who would rather not run an installer. The
# archive holds one directory rather than loose files: a zip that explodes over
# the Downloads folder is a zip that loses its DLLs.
$zip = Join-Path (Resolve-Path $Out) "stemd-$version-windows-x64.zip"
Remove-Item $zip -Force -ErrorAction SilentlyContinue
Compress-Archive -Path $Source -DestinationPath $zip -CompressionLevel Optimal

$staged = ((Get-ChildItem $Source -Recurse | Measure-Object Length -Sum).Sum)
Write-Host ''
Write-Host ("done, from {0:N1} MB staged:" -f ($staged / 1MB))
foreach ($f in $installer, $zip) {
    Write-Host ("  {0,-34} {1,7:N1} MB" -f (Split-Path $f -Leaf), ((Get-Item $f).Length / 1MB))
}
Write-Host '  the installer is per user, into %LOCALAPPDATA%\Programs\stemd: no'
Write-Host '  administrator, and install-cuda.cmd can write beside the executable'
Write-Host '  without one either'
