@echo off
REM DRLHSS Complete Build Script for Windows
REM Builds the entire DRLHSS system with all components

echo =========================================
echo   DRLHSS Complete Build Script (Windows)
echo =========================================
echo.

REM Check for Visual Studio
where cl.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Visual Studio C++ compiler not found!
    echo Please run this script from "Developer Command Prompt for VS"
    echo or "x64 Native Tools Command Prompt for VS"
    pause
    exit /b 1
)

echo [OK] Visual Studio C++ compiler found
echo.

REM Check for CMake
where cmake.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake not found!
    echo Install CMake from: https://cmake.org/download/
    pause
    exit /b 1
)

echo [OK] CMake found
echo.

REM Check for ONNX Runtime
if not exist "external\onnxruntime" (
    echo [WARNING] ONNX Runtime not found in external\onnxruntime
    echo Download from: https://github.com/microsoft/onnxruntime/releases
    echo Extract to: external\onnxruntime
    echo.
)

REM Create build directory
echo Creating build directory...
if not exist "build" mkdir build
cd build

REM Configure with CMake
echo.
echo Configuring with CMake...
cmake .. -G "Visual Studio 16 2019" -A x64
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake configuration failed!
    cd ..
    pause
    exit /b 1
)

echo [OK] CMake configuration successful
echo.

REM Build
echo Building project (Release)...
cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed!
    cd ..
    pause
    exit /b 1
)

echo [OK] Build successful
echo.

REM List built executables
echo =========================================
echo   Build Complete!
echo =========================================
echo.
echo Built executables in build\Release\:
echo.

if exist "Release\DRLHSS_main.exe" (
    echo [OK] DRLHSS_main.exe - Legacy main executable
)

if exist "Release\drl_integration_example.exe" (
    echo [OK] drl_integration_example.exe - DRL integration example
)

if exist "Release\integrated_system_example.exe" (
    echo [OK] integrated_system_example.exe - Complete integrated system
)

if exist "Release\test_windows_sandbox.exe" (
    echo [OK] test_windows_sandbox.exe - Windows sandbox tests
)

echo.
echo To run the integrated system:
echo   cd build\Release
echo   integrated_system_example.exe
echo.
echo Note: Run as Administrator for full sandbox functionality
echo.
echo For more information, see:
echo   - docs\NIDPS_INTEGRATION_GUIDE.md
echo   - docs\DEPLOYMENT_GUIDE.md
echo   - COMPLETE_INTEGRATION_README.md
echo.
echo =========================================

cd ..
pause

