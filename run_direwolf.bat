@echo off
REM Quick launcher for DIREWOLF Desktop Application

echo ========================================
echo DIREWOLF Desktop Application
echo ========================================
echo.

set BUILD_DIR=n:\CPPfiles\DRLHSS\build_desktop
set EXE_PATH=%BUILD_DIR%\Release\direwolf.exe

REM Check if executable exists
if not exist "%EXE_PATH%" (
    echo [ERROR] DIREWOLF executable not found!
    echo.
    echo Please build first:
    echo   n:\CPPfiles\DRLHSS\build_desktop_simple.bat
    echo.
    pause
    exit /b 1
)

echo [OK] Found DIREWOLF at: %EXE_PATH%
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Running with Administrator privileges
) else (
    echo [INFO] Running without Administrator privileges
    echo [INFO] Some features may be limited
)

echo.
echo Choose launch mode:
echo   1. Normal (with authentication)
echo   2. Setup admin account
echo   3. Development mode (skip auth)
echo   4. Exit
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" goto normal
if "%choice%"=="2" goto setup
if "%choice%"=="3" goto devmode
if "%choice%"=="4" goto end

:normal
echo.
echo Launching DIREWOLF...
echo.
"%EXE_PATH%"
goto end

:setup
echo.
echo Launching DIREWOLF admin setup...
echo.
"%EXE_PATH%" --setup-admin
goto end

:devmode
echo.
echo Launching DIREWOLF in development mode (no auth)...
echo.
"%EXE_PATH%" --no-auth
goto end

:end
echo.
