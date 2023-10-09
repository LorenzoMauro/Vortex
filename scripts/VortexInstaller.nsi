!include MUI2.nsh

; Set modern UI options and themes
!define MUI_ICON "E:/Dev/Vortex/assets/Vortex.ico"  ; You can replace 'pathToYourIcon.ico' with the path to your icon file
!define MUI_UNICON "E:/Dev/Vortex/assets/Vortex.ico"  ; Uninstaller icon, change as needed
;!define MUI_HEADERIMAGE
;!define MUI_HEADERIMAGE_RIGHT
;!define MUI_HEADERIMAGE_BITMAP "E:/Dev/Vortex/pathToHeaderBitmap.bmp"  ; Header image for a modern touch
!define MUI_PRODUCT "Vortex" ; The name of your software
!define MUI_ABORTWARNING ; Warns the user if they want to abort the installation
!define MUI_UNABORTWARNING ; Warns the user if they want to abort the uninstallation

; Pages
!insertmacro MUI_PAGE_WELCOME
;!insertmacro MUI_PAGE_LICENSE "E:/Dev/Vortex/License.txt" ; If you have a license file, this will add a license page
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; Languages
!insertmacro MUI_LANGUAGE "English"

Outfile "../installer/VortexInstaller.exe"
InstallDir "$PROGRAMFILES64\Vortex"
!define INSTALL_PATH "E:/Dev/Vortex/build/install"

Section "VortexApp" SecVortex
    SetOutPath "$INSTDIR"
    File /r "${INSTALL_PATH}\*.*"
    SetShellVarContext all
    CreateShortCut "$DESKTOP\Vortex.lnk" "$INSTDIR\Vortex.exe"
    WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\Vortex.exe"
    Delete "$INSTDIR\Uninstall.exe"
    RMDir /r "$INSTDIR"
    Delete "$DESKTOP\Vortex.lnk"
SectionEnd
