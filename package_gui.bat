@echo off
echo ========================================
echo DIREWOLF GUI Deployment Package Creator
echo ========================================
echo.

REM Set Qt path
set QT_PATH=C:\Qt\6.5.3\msvc2019_64
set PATH=%QT_PATH%\bin;%PATH%

REM Check if build exists
if not exist "build_gui\bin\Release\direwolf_gui.exe" (
    echo Error: Build not found! Run build_gui.bat first.
    pause
    exit /b 1
)

REM Create deployment directory
set DEPLOY_DIR=direwolf_gui_deploy
if exist "%DEPLOY_DIR%" rmdir /s /q "%DEPLOY_DIR%"
mkdir "%DEPLOY_DIR%"

echo Copying executable...
copy "build_gui\bin\Release\direwolf_gui.exe" "%DEPLOY_DIR%\"

echo.
echo Running windeployqt...
cd "%DEPLOY_DIR%"
windeployqt --qmldir "..\qml" direwolf_gui.exe
cd ..

echo.
echo Copying additional files...
copy "README.md" "%DEPLOY_DIR%\"
copy "GUI_QUICK_START.md" "%DEPLOY_DIR%\"
copy "PHASE4_GUI_COMPLETE.md" "%DEPLOY_DIR%\"
copy "LICENSE" "%DEPLOY_DIR%\" 2>nul

echo.
echo Creating launcher script...
(
echo @echo off
echo echo Starting DIREWOLF GUI Dashboard...
echo start "" direwolf_gui.exe
) > "%DEPLOY_DIR%\run.bat"

echo.
echo Creating README...
(
echo DIREWOLF GUI Dashboard - Deployment Package
echo ===========================================
echo.
echo To run the application:
echo   1. Double-click run.bat
echo   OR
echo   2. Double-click direwolf_gui.exe
echo.
echo Documentation:
echo   - GUI_QUICK_START.md - Quick start guide
echo   - PHASE4_GUI_COMPLETE.md - Full documentation
echo   - README.md - System overview
echo.
echo System Requirements:
echo   - Windows 10/11 64-bit
echo   - 4 GB RAM minimum
echo   - 500 MB disk space
echo   - Display: 1280x720 or higher
echo.
echo Support: See documentation files
) > "%DEPLOY_DIR%\README.txt"

echo.
echo Creating archive...
powershell Compress-Archive -Path "%DEPLOY_DIR%\*" -DestinationPath "direwolf_gui_v1.0.0.zip" -Force

echo.
echo ========================================
echo Deployment package created successfully!
echo ========================================
echo.
echo Package location: direwolf_gui_v1.0.0.zip
echo Deployment folder: %DEPLOY_DIR%\
echo.
echo Contents:
dir /b "%DEPLOY_DIR%"
echo.
echo Package is ready for distribution!
echo.

pause
