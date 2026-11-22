# Turn the staged Windows bundle into the two things people download.
#
# `bundle-windows.ps1` produces the directory; this wraps it into an installer
# and a zip. The installer is for people who want the program installed, and the
# zip is for those who would not run an installer, or want the program on a USB
# stick.
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

# Defaults are set here, outside the param block, because $PSScriptRoot is not
# reliably populated while the parameters are bound: it is empty under some ways
# of invoking a script, and Join-Path on an empty path is a hard error before
# the first line of the body runs.
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Source) { $Source = Join-Path $root 'dist\stemd-windows' }
if (-not $Out)    { $Out    = Join-Path $root 'dist' }

function Say($m) { Write-Host "  $m" }

# The version, from the one place it is declared. It is read here, never passed
# in, so an installer cannot be named for a version its binaries are not.
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

# The licences and notices listed in packaging/notices.list ship with the
# binaries. A file without an extension gets .txt, so that it opens on a
# double-click.
$notices = Get-Content (Join-Path $root 'packaging\notices.list') |
    Where-Object { $_ -and -not $_.StartsWith('#') }
foreach ($notice in $notices) {
    $name = Split-Path $notice -Leaf
    if (-not [System.IO.Path]::HasExtension($name)) { $name = "$name.txt" }
    Copy-Item (Join-Path $root $notice) (Join-Path $Source $name) -Force
}
Say "notices: $(@($notices).Count) files"

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
# `Stop` a redirected native stderr is a terminating error, as with cargo in
# bundle-windows.ps1.
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

# The same staged files, for anyone who would not run an installer. The archive
# holds one directory: a zip of loose files spills over the Downloads folder and
# loses its DLLs.
$zip = Join-Path (Resolve-Path $Out) "stemd-$version-windows-x64.zip"
Remove-Item $zip -Force -ErrorAction SilentlyContinue
Compress-Archive -Path $Source -DestinationPath $zip -CompressionLevel Optimal

$staged = ((Get-ChildItem $Source -Recurse | Measure-Object Length -Sum).Sum)
Write-Host ''
Write-Host ("done, from {0:N1} MB staged:" -f ($staged / 1MB))
foreach ($f in $installer, $zip) {
    Write-Host ("  {0,-34} {1,7:N1} MB" -f (Split-Path $f -Leaf), ((Get-Item $f).Length / 1MB))
}
Write-Host '  the installer is per user, into %LOCALAPPDATA%\Programs\Stemd: no'
Write-Host '  administrator, and install-cuda.cmd can write beside the executable'
Write-Host '  without one either'
