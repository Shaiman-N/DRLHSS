@echo off
REM DIREWOLF One-Click Installer
REM This script builds and installs DIREWOLF in one step

echo ========================================
echo DIREWOLF One-Click Installer
echo ========================================
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Administrator privileges required!
    echo.
    echo Please right-click this file and select "Run as Administrator"
    echo.
    pause
    exit /b 1
)

echo [OK] Running with Administrator privileges
echo.

REM Step 1: Build DIREWOLF
echo [Step 1/2] Building DIREWOLF...
echo.
call build_desktop_simple.bat
if %errorLevel% neq 0 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo [OK] Build complete!
echo.

REM Step 2: Install DIREWOLF
echo [Step 2/2] Installing DIREWOLF...
echo.

REM Run PowerShell installer
powershell.exe -ExecutionPolicy Bypass -File "installer\install_direwolf.ps1"

if %errorLevel% neq 0 (
    echo.
    echo [ERROR] Installation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo DIREWOLF has been installed successfully!
echo.
echo To run DIREWOLF:
echo   - Start Menu ^> DIREWOLF
echo   - Desktop shortcut
echo   - Command: direwolf
echo.
pause
