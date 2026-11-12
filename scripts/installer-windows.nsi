; The Windows installer, built from a staged bundle by package-windows.ps1.
;
; Two modes, chosen on the first page. Just for me, into
; %LOCALAPPDATA%\Programs\Stemd, is the default and needs no administrator. For
; all users goes to Program Files and does.
;
; The per-user directory has to be `Programs\Stemd` and not `Stemd`, because
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

!define NAME "Stemd"
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
; An update keeps the mode of the copy it replaces: the hive holding its
; uninstall key, this account's first.
!define MULTIUSER_INSTALLMODE_DEFAULT_REGISTRY_KEY "${UNINST_KEY}"
!define MULTIUSER_INSTALLMODE_DEFAULT_REGISTRY_VALUENAME "InstallLocation"

!include "MultiUser.nsh"
!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"
!include "Sections.nsh"

Name "${NAME} ${VERSION}"
OutFile "${OUTFILE}"
ShowInstDetails show
ShowUninstDetails show

VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "${NAME}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "FileVersion" "${VERSION}.0"
VIAddVersionKey "FileDescription" "Stemd installer"
VIAddVersionKey "CompanyName" "${PUBLISHER}"
VIAddVersionKey "LegalCopyright" "MIT OR Apache-2.0"

!define MUI_ICON "${ICON}"
!define MUI_UNICON "${ICON}"
!define MUI_ABORTWARNING

!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "Start Stemd"
!define MUI_FINISHPAGE_RUN_FUNCTION LaunchStemd
!define MUI_FINISHPAGE_LINK "github.com/nsaintot/stemd"
!define MUI_FINISHPAGE_LINK_LOCATION "${HOMEPAGE}"

!insertmacro MULTIUSER_PAGE_INSTALLMODE
!define MUI_PAGE_CUSTOMFUNCTION_PRE ComponentsPre
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; Through explorer, so the program does not inherit the installer's token. On an
; administrator account this whole process is elevated, and a stemd started from
; it would run as administrator and write its settings and its cache somewhere
; the user's own next run will not look.
Function LaunchStemd
  Exec '"$WINDIR\explorer.exe" "$INSTDIR\stemd-server.exe"'
FunctionEnd

; The all-users replacement for the shipped `install-cuda.cmd`: same call, with
; a consent prompt in front of it, because Program Files is not writable by the
; person who will double-click this.
Function WriteElevatingCudaCmd
  FileOpen $0 "$INSTDIR\install-cuda.cmd" w
  FileWrite $0 "@echo off$\r$\n"
  FileWrite $0 "rem Fetch the CUDA runtime beside stemd-server.exe. About 1.2 GB,$\r$\n"
  FileWrite $0 "rem once, and only useful on a machine with an NVIDIA card. This copy$\r$\n"
  FileWrite $0 "rem of Stemd is installed for all users, so the libraries land where$\r$\n"
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

Section "Stemd" SecMain
  SectionIn RO
  Call WaitForStemdToExit
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
  ; menu for an all-users install and this account's for the other. Removed
  ; first: Windows keeps a name's old case when a file is overwritten, so an
  ; upgrade would keep the lowercase shortcut.
  Delete "$SMPROGRAMS\${NAME}\${NAME}.lnk"
  RMDir  "$SMPROGRAMS\${NAME}"
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
  ; Set again by the GPU section when it runs, so the next install offers the
  ; same choice.
  WriteRegDWORD SHCTX "${UNINST_KEY}" "GpuSupport" 0
  WriteRegDWORD SHCTX "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD SHCTX "${UNINST_KEY}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD SHCTX "${UNINST_KEY}" "EstimatedSize" "$0"
SectionEnd

; The CUDA runtime and cuDNN, about 1.2 GB from NVIDIA, laid beside the program
; during the install with the download in the details pane. A component rather
; than a script beside the executable, because nobody finds the script, and the
; difference is a separation in seconds instead of minutes. A failed download
; leaves a working program on the CPU and the script to try again.
Section "GPU support (CUDA runtime, 1.2 GB download)" SecCuda
  WriteRegDWORD SHCTX "${UNINST_KEY}" "GpuSupport" 1
  DetailPrint "Fetching the CUDA runtime beside Stemd, about 1.2 GB..."
  nsExec::ExecToLog '"$INSTDIR\stemd-server.exe" --install-cuda'
  Pop $0
  ${If} $0 != 0
    DetailPrint "The CUDA runtime could not be fetched (exit $0); Stemd runs on the CPU until install-cuda.cmd is run."
    ${IfNot} ${Silent}
      MessageBox MB_ICONEXCLAMATION|MB_OK "The CUDA runtime could not be fetched, so Stemd will run on the CPU for now.$\r$\n$\r$\nRun install-cuda.cmd in the install folder to try again."
    ${EndIf}
  ${EndIf}
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain} "Stemd, its command-line client and the runtime they need."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecCuda} "Downloads the CUDA runtime and cuDNN from NVIDIA beside Stemd, so separations run on the card rather than the CPU. Needs an NVIDIA driver."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; Below the sections, because .onInit names one of them.
; makensis produces a 32-bit installer, so every HKLM\Software write is
; redirected into WOW6432Node unless the view is set. The program is 64-bit and
; its Add/Remove Programs entry belongs in the 64-bit view with it. Set before
; MULTIUSER_INIT, which reads the uninstall key itself to find a previous
; install, so both halves have to agree on where that key lives.
Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "Stemd is built for 64-bit Windows only."
    Abort
  ${EndIf}
  SetRegView 64
  !insertmacro MULTIUSER_INIT
  ; No NVIDIA driver, nothing for the runtime to reach: the GPU component is
  ; left unticked and greyed out rather than downloading 1.2 GB for nothing.
  ${IfNot} ${FileExists} "$SYSDIR\nvcuda.dll"
    SectionSetFlags ${SecCuda} ${SF_RO}
  ${Else}
    Call RememberGpuChoice
  ${EndIf}
FunctionEnd

; The install mode whose GPU choice RememberGpuChoice last applied.
Var GpuMode

; Tick GPU support as the copy being replaced had it, in the hive of the
; current install mode. A copy installed before `GpuSupport` existed counts as
; having it when cudart64_13.dll is beside it. With the runtime in place the
; section downloads nothing.
Function RememberGpuChoice
  StrCpy $GpuMode $MultiUser.InstallMode
  !insertmacro SelectSection ${SecCuda}
  ClearErrors
  ReadRegDWORD $0 SHCTX "${UNINST_KEY}" "GpuSupport"
  ${If} ${Errors}
    ${If} ${FileExists} "$INSTDIR\Uninstall.exe"
    ${AndIfNot} ${FileExists} "$INSTDIR\cudart64_13.dll"
      !insertmacro UnselectSection ${SecCuda}
    ${EndIf}
  ${ElseIf} $0 == 0
    !insertmacro UnselectSection ${SecCuda}
  ${EndIf}
FunctionEnd

; The mode page may have switched to the other install: read its choice.
Function ComponentsPre
  ${If} ${FileExists} "$SYSDIR\nvcuda.dll"
  ${AndIf} $GpuMode != $MultiUser.InstallMode
    Call RememberGpuChoice
  ${EndIf}
FunctionEnd

; The updater starts Setup just before Stemd quits. Overwriting a running
; executable fails, and silent mode ignores the failure, so wait up to 60 s
; for it to be released.
Function WaitForStemdToExit
  StrCpy $1 0
  ${Do}
    ${IfNot} ${FileExists} "$INSTDIR\stemd-server.exe"
      ${Break}
    ${EndIf}
    ClearErrors
    FileOpen $0 "$INSTDIR\stemd-server.exe" a
    ${IfNot} ${Errors}
      FileClose $0
      ${Break}
    ${EndIf}
    ${If} $1 >= 120
      DetailPrint "Stemd is still running; close it and run Setup again."
      ${IfNot} ${Silent}
        MessageBox MB_ICONSTOP|MB_OK "Stemd is still running. Close it and run Setup again."
      ${EndIf}
      Abort
    ${EndIf}
    Sleep 500
    IntOp $1 $1 + 1
  ${Loop}
FunctionEnd

; `/S /relaunch`, as the app's updater runs it: start Stemd once installed.
Function .onInstSuccess
  ${If} ${Silent}
    ${GetParameters} $0
    ClearErrors
    ${GetOptions} $0 "/relaunch" $1
    ${IfNot} ${Errors}
      Call LaunchStemd
    ${EndIf}
  ${EndIf}
FunctionEnd

Function un.onInit
  SetRegView 64
  !insertmacro MULTIUSER_UNINIT
FunctionEnd

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
