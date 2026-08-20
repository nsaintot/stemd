; The Windows installer, built from a staged bundle by package-windows.ps1.
;
; Two modes, chosen on the first page. Just for me, into
; %LOCALAPPDATA%\Programs\stemd, is the default and needs no administrator. For
; all users goes to Program Files and does.
;
; The per-user directory has to be `Programs\stemd` and not `stemd`, because
; %LOCALAPPDATA%\stemd is where the program keeps its weights and its cache, and
; an uninstaller pointed at that directory would take a gigabyte the user
; downloaded with it. MultiUser resolves FOLDERID_UserProgramFiles, so that is
; already what it picks; the constraint is written down here because the obvious
; simplification breaks it silently.
;
; The other thing the two modes differ in is CUDA. `install-cuda.cmd` writes
; about 1.2 GB beside the executable, which a per-user install can do and a
; Program Files one cannot, so the all-users install lays down a copy that asks
; for an administrator instead of failing with a permission error a gigabyte in.
;
; The uninstaller removes the install directory and the shortcut. Weights,
; settings and the stem cache live in the user data directory and are left
; alone: they are a download, not an opinion about whether stemd is installed.

Unicode true
ManifestDPIAware true
SetCompressor /SOLID lzma

!ifndef VERSION
  !error "VERSION is not defined; package-windows.ps1 passes it"
!endif
!ifndef SOURCE
  !error "SOURCE is not defined; it names the staged bundle directory"
!endif
!ifndef OUTFILE
  !error "OUTFILE is not defined; it names the installer to write"
!endif
!ifndef ICON
  !error "ICON is not defined; it names the .ico to brand this with"
!endif

!define NAME "stemd"
!define PUBLISHER "nsaintot"
!define HOMEPAGE "https://github.com/nsaintot/stemd"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\stemd"

; Highest rather than user, because the all-users option needs a token this
; process does not otherwise have and stock NSIS cannot elevate mid-run: that
; wants the UAC plugin, which is not part of NSIS. The cost is one consent
; prompt at launch on an administrator account, including for a per-user
; install. The benefit is that choosing all users works rather than failing on
; the first write.
!define MULTIUSER_EXECUTIONLEVEL Highest
!define MULTIUSER_MUI
!define MULTIUSER_INSTALLMODE_COMMANDLINE
!define MULTIUSER_INSTALLMODE_DEFAULT_CURRENTUSER
!define MULTIUSER_USE_PROGRAMFILES64
!define MULTIUSER_INSTALLMODE_INSTDIR "${NAME}"
!define MULTIUSER_INSTALLMODE_INSTDIR_REGISTRY_KEY "${UNINST_KEY}"
!define MULTIUSER_INSTALLMODE_INSTDIR_REGISTRY_VALUENAME "InstallLocation"

!include "MultiUser.nsh"
!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"

Name "${NAME} ${VERSION}"
OutFile "${OUTFILE}"
ShowInstDetails show
ShowUninstDetails show

VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "${NAME}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "FileVersion" "${VERSION}.0"
VIAddVersionKey "FileDescription" "stemd installer"
VIAddVersionKey "CompanyName" "${PUBLISHER}"
VIAddVersionKey "LegalCopyright" "MIT OR Apache-2.0"

!define MUI_ICON "${ICON}"
!define MUI_UNICON "${ICON}"
!define MUI_ABORTWARNING

!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "Start stemd"
!define MUI_FINISHPAGE_RUN_FUNCTION LaunchStemd
; The readme slot, repurposed. Unticked, because it is a 1.2 GB download, and
; offered at all because the alternative is a program that runs on the CPU and
; a window that says why without saying where to click.
!define MUI_FINISHPAGE_SHOWREADME ""
!define MUI_FINISHPAGE_SHOWREADME_TEXT "Fetch the CUDA runtime (about 1.2 GB, NVIDIA cards only)"
!define MUI_FINISHPAGE_SHOWREADME_NOTCHECKED
!define MUI_FINISHPAGE_SHOWREADME_FUNCTION FetchCuda
!define MUI_FINISHPAGE_LINK "github.com/nsaintot/stemd"
!define MUI_FINISHPAGE_LINK_LOCATION "${HOMEPAGE}"

!insertmacro MULTIUSER_PAGE_INSTALLMODE
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; makensis produces a 32-bit installer, so every HKLM\Software write is
; redirected into WOW6432Node unless the view is set. The program is 64-bit and
; its Add/Remove Programs entry belongs in the 64-bit view with it. Set before
; MULTIUSER_INIT, which reads the uninstall key itself to find a previous
; install, so both halves have to agree on where that key lives.
Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "stemd is built for 64-bit Windows only."
    Abort
  ${EndIf}
  SetRegView 64
  !insertmacro MULTIUSER_INIT
FunctionEnd

Function un.onInit
  SetRegView 64
  !insertmacro MULTIUSER_UNINIT
FunctionEnd

; Through explorer, so the program does not inherit the installer's token. On an
; administrator account this whole process is elevated, and a stemd started from
; it would run as administrator and write its settings and its cache somewhere
; the user's own next run will not look.
Function LaunchStemd
  Exec '"$WINDIR\explorer.exe" "$INSTDIR\stemd-server.exe"'
FunctionEnd

Function FetchCuda
  ExecShell "" "$INSTDIR\install-cuda.cmd"
FunctionEnd

; The all-users replacement for the shipped `install-cuda.cmd`: same call, with
; a consent prompt in front of it, because Program Files is not writable by the
; person who will double-click this.
Function WriteElevatingCudaCmd
  FileOpen $0 "$INSTDIR\install-cuda.cmd" w
  FileWrite $0 "@echo off$\r$\n"
  FileWrite $0 "rem Fetch the CUDA runtime beside stemd-server.exe. About 1.2 GB,$\r$\n"
  FileWrite $0 "rem once, and only useful on a machine with an NVIDIA card. This copy$\r$\n"
  FileWrite $0 "rem of stemd is installed for all users, so the libraries land where$\r$\n"
  FileWrite $0 "rem writing needs an administrator.$\r$\n"
  FileWrite $0 "net session >nul 2>&1$\r$\n"
  FileWrite $0 "if errorlevel 1 ($\r$\n"
  FileWrite $0 "  powershell -NoProfile -Command $\"Start-Process -Verb RunAs -FilePath '%~f0'$\"$\r$\n"
  FileWrite $0 "  exit /b$\r$\n"
  FileWrite $0 ")$\r$\n"
  FileWrite $0 "$\"%~dp0stemd-server.exe$\" --install-cuda$\r$\n"
  FileWrite $0 "pause$\r$\n"
  FileClose $0
FunctionEnd

Section "stemd" SecMain
  SectionIn RO
  SetOutPath "$INSTDIR"
  ; Overwritten in place rather than uninstalled first: a previous install may
  ; hold the CUDA runtime, and removing it would mean downloading it again for
  ; no reason. The file set is fixed, so nothing is left stale.
  File /r "${SOURCE}\*"
  File "${ICON}"

  ${If} $MultiUser.InstallMode == "AllUsers"
    Call WriteElevatingCudaCmd
  ${EndIf}

  ; SHCTX and $SMPROGRAMS both follow the mode, so this is the all-users Start
  ; menu for an all-users install and this account's for the other.
  CreateDirectory "$SMPROGRAMS\${NAME}"
  CreateShortCut "$SMPROGRAMS\${NAME}\${NAME}.lnk" "$INSTDIR\stemd-server.exe" \
    "" "$INSTDIR\stemd-server.exe" 0

  WriteUninstaller "$INSTDIR\Uninstall.exe"
  WriteRegStr   SHCTX "${UNINST_KEY}" "DisplayName"     "${NAME}"
  WriteRegStr   SHCTX "${UNINST_KEY}" "DisplayVersion"  "${VERSION}"
  WriteRegStr   SHCTX "${UNINST_KEY}" "DisplayIcon"     "$INSTDIR\stemd.ico"
  WriteRegStr   SHCTX "${UNINST_KEY}" "Publisher"       "${PUBLISHER}"
  WriteRegStr   SHCTX "${UNINST_KEY}" "URLInfoAbout"    "${HOMEPAGE}"
  WriteRegStr   SHCTX "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  ; With the mode on it, so the uninstaller reads the hive it was written to
  ; rather than guessing from whatever token happens to be running it.
  WriteRegStr   SHCTX "${UNINST_KEY}" "UninstallString" \
    '"$INSTDIR\Uninstall.exe" /$MultiUser.InstallMode'
  WriteRegStr   SHCTX "${UNINST_KEY}" "QuietUninstallString" \
    '"$INSTDIR\Uninstall.exe" /$MultiUser.InstallMode /S'
  WriteRegDWORD SHCTX "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD SHCTX "${UNINST_KEY}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD SHCTX "${UNINST_KEY}" "EstimatedSize" "$0"
SectionEnd

Section "Uninstall"
  Delete "$SMPROGRAMS\${NAME}\${NAME}.lnk"
  RMDir  "$SMPROGRAMS\${NAME}"
  ; Guarded, because `RMDir /r` on a directory this did not create is the worst
  ; thing an uninstaller can do, and the registry value it comes from is not
  ; ours to trust.
  IfFileExists "$INSTDIR\stemd-server.exe" 0 +2
    RMDir /r "$INSTDIR"
  DeleteRegKey SHCTX "${UNINST_KEY}"
SectionEnd
