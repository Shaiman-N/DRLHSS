; DIREWOLF Security System - Windows Installer Script
; NSIS Script for creating Windows 11 installer

;--------------------------------
; Includes

!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "WinVer.nsh"

;--------------------------------
; General

; Name and file
Name "DIREWOLF Security System"
OutFile "DIREWOLF_Setup_v1.0.0.exe"
Unicode True

; Default installation folder
InstallDir "$PROGRAMFILES64\DIREWOLF"

; Get installation folder from registry if available
InstallDirRegKey HKLM "Software\DIREWOLF" "InstallDir"

; Request application privileges for Windows Vista+
RequestExecutionLevel admin

; Branding
BrandingText "DIREWOLF Security System v1.0.0"

;--------------------------------
; Variables

Var StartMenuFolder

;--------------------------------
; Interface Settings

!define MUI_ABORTWARNING
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_BITMAP "${NSISDIR}\Contrib\Graphics\Header\nsis.bmp"
!define MUI_WELCOMEFINISHPAGE_BITMAP "${NSISDIR}\Contrib\Graphics\Wizard\win.bmp"

;--------------------------------
; Pages

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE.txt"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY

; Start Menu Folder Page Configuration
!define MUI_STARTMENUPAGE_REGISTRY_ROOT "HKLM" 
!define MUI_STARTMENUPAGE_REGISTRY_KEY "Software\DIREWOLF" 
!define MUI_STARTMENUPAGE_REGISTRY_VALUENAME "Start Menu Folder"
!insertmacro MUI_PAGE_STARTMENU Application $StartMenuFolder

!insertmacro MUI_PAGE_INSTFILES

; Finish page
!define MUI_FINISHPAGE_RUN "$INSTDIR\bin\direwolf.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Launch DIREWOLF Security System"
!define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\README.txt"
!define MUI_FINISHPAGE_SHOWREADME_TEXT "View README"
!insertmacro MUI_PAGE_FINISH

; Uninstaller pages
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

;--------------------------------
; Languages

!insertmacro MUI_LANGUAGE "English"

;--------------------------------
; Version Information

VIProductVersion "1.0.0.0"
VIAddVersionKey "ProductName" "DIREWOLF Security System"
VIAddVersionKey "CompanyName" "DIREWOLF Security"
VIAddVersionKey "LegalCopyright" "Copyright (C) 2024 DIREWOLF Security"
VIAddVersionKey "FileDescription" "DIREWOLF Security System Installer"
VIAddVersionKey "FileVersion" "1.0.0.0"
VIAddVersionKey "ProductVersion" "1.0.0.0"

;--------------------------------
; Installer Sections

Section "DIREWOLF Core" SecCore
  SectionIn RO  ; Read-only, always installed
  
  ; Set output path to the installation directory
  SetOutPath "$INSTDIR\bin"
  
  ; Copy main executable
  File "..\build_desktop\Release\direwolf.exe"
  
  ; Create directories
  CreateDirectory "$INSTDIR\config"
  CreateDirectory "$INSTDIR\logs"
  CreateDirectory "$INSTDIR\data"
  CreateDirectory "$INSTDIR\models"
  
  ; Copy documentation
  SetOutPath "$INSTDIR"
  File /oname=README.txt "..\DESKTOP_APP_QUICK_START.md"
  File /oname=LICENSE.txt "..\LICENSE"
  
  ; Store installation folder
  WriteRegStr HKLM "Software\DIREWOLF" "InstallDir" $INSTDIR
  
  ; Create uninstaller
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  
  ; Write uninstall information to registry
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DIREWOLF" "DisplayName" "DIREWOLF Security System"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DIREWOLF" "UninstallString" "$\"$INSTDIR\Uninstall.exe$\""
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DIREWOLF" "QuietUninstallString" "$\"$INSTDIR\Uninstall.exe$\" /S"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DIREWOLF" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DIREWOLF" "DisplayIcon" "$INSTDIR\bin\direwolf.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DIREWOLF" "Publisher" "DIREWOLF Security"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DIREWOLF" "DisplayVersion" "1.0.0"
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DIREWOLF" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DIREWOLF" "NoRepair" 1
  
  ; Create Start Menu shortcuts
  !insertmacro MUI_STARTMENU_WRITE_BEGIN Application
    CreateDirectory "$SMPROGRAMS\$StartMenuFolder"
    CreateShortcut "$SMPROGRAMS\$StartMenuFolder\DIREWOLF.lnk" "$INSTDIR\bin\direwolf.exe"
    CreateShortcut "$SMPROGRAMS\$StartMenuFolder\Setup Admin.lnk" "$INSTDIR\bin\direwolf.exe" "--setup-admin"
    CreateShortcut "$SMPROGRAMS\$StartMenuFolder\README.lnk" "$INSTDIR\README.txt"
    CreateShortcut "$SMPROGRAMS\$StartMenuFolder\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
  !insertmacro MUI_STARTMENU_WRITE_END
  
SectionEnd

Section "Desktop Shortcut" SecDesktop
  CreateShortcut "$DESKTOP\DIREWOLF.lnk" "$INSTDIR\bin\direwolf.exe"
SectionEnd

Section "Windows Service" SecService
  ; Install as Windows service
  nsExec::ExecToLog 'sc create DIREWOLF binPath= "$INSTDIR\bin\direwolf.exe --service" start= auto DisplayName= "DIREWOLF Security System"'
  Pop $0
  ${If} $0 == 0
    DetailPrint "Windows Service installed successfully"
  ${Else}
    DetailPrint "Windows Service installation failed (may already exist)"
  ${EndIf}
SectionEnd

Section "Auto-Start with Windows" SecAutoStart
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "DIREWOLF" "$INSTDIR\bin\direwolf.exe"
SectionEnd

;--------------------------------
; Descriptions

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecCore} "Core DIREWOLF application files (required)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} "Create a desktop shortcut for quick access"
  !insertmacro MUI_DESCRIPTION_TEXT ${SecService} "Install DIREWOLF as a Windows service for automatic startup"
  !insertmacro MUI_DESCRIPTION_TEXT ${SecAutoStart} "Start DIREWOLF automatically when Windows starts"
!insertmacro MUI_FUNCTION_DESCRIPTION_END

;--------------------------------
; Installer Functions

Function .onInit
  ; Check Windows version
  ${IfNot} ${AtLeastWin10}
    MessageBox MB_OK|MB_ICONSTOP "DIREWOLF requires Windows 10 or later.$\n$\nThis system is running an older version of Windows."
    Abort
  ${EndIf}
  
  ; Check if already installed
  ReadRegStr $0 HKLM "Software\DIREWOLF" "InstallDir"
  ${If} $0 != ""
    MessageBox MB_YESNO|MB_ICONQUESTION "DIREWOLF is already installed at:$\n$\n$0$\n$\nDo you want to reinstall?" IDYES continue
    Abort
    continue:
  ${EndIf}
  
  ; Check for admin privileges
  UserInfo::GetAccountType
  Pop $0
  ${If} $0 != "admin"
    MessageBox MB_OK|MB_ICONSTOP "Administrator privileges required.$\n$\nPlease run the installer as Administrator."
    Abort
  ${EndIf}
FunctionEnd

Function .onInstSuccess
  MessageBox MB_YESNO "DIREWOLF has been installed successfully!$\n$\nWould you like to setup your admin account now?" IDNO skipSetup
  Exec '"$INSTDIR\bin\direwolf.exe" --setup-admin'
  skipSetup:
FunctionEnd

;--------------------------------
; Uninstaller Section

Section "Uninstall"
  ; Stop and remove Windows service
  nsExec::ExecToLog 'sc stop DIREWOLF'
  nsExec::ExecToLog 'sc delete DIREWOLF'
  
  ; Remove auto-start registry entry
  DeleteRegValue HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "DIREWOLF"
  
  ; Remove files and directories
  Delete "$INSTDIR\bin\direwolf.exe"
  Delete "$INSTDIR\README.txt"
  Delete "$INSTDIR\LICENSE.txt"
  Delete "$INSTDIR\Uninstall.exe"
  
  ; Remove directories (only if empty)
  RMDir "$INSTDIR\bin"
  RMDir "$INSTDIR\config"
  RMDir "$INSTDIR\logs"
  RMDir "$INSTDIR\data"
  RMDir "$INSTDIR\models"
  RMDir "$INSTDIR"
  
  ; Remove Start Menu shortcuts
  !insertmacro MUI_STARTMENU_GETFOLDER Application $StartMenuFolder
  Delete "$SMPROGRAMS\$StartMenuFolder\DIREWOLF.lnk"
  Delete "$SMPROGRAMS\$StartMenuFolder\Setup Admin.lnk"
  Delete "$SMPROGRAMS\$StartMenuFolder\README.lnk"
  Delete "$SMPROGRAMS\$StartMenuFolder\Uninstall.lnk"
  RMDir "$SMPROGRAMS\$StartMenuFolder"
  
  ; Remove desktop shortcut
  Delete "$DESKTOP\DIREWOLF.lnk"
  
  ; Remove registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DIREWOLF"
  DeleteRegKey HKLM "Software\DIREWOLF"
  
SectionEnd

;--------------------------------
; Uninstaller Functions

Function un.onInit
  MessageBox MB_YESNO "Are you sure you want to uninstall DIREWOLF Security System?" IDYES continue
  Abort
  continue:
FunctionEnd

Function un.onUninstSuccess
  MessageBox MB_OK "DIREWOLF has been successfully removed from your computer."
FunctionEnd
