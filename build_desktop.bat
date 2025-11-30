@echo off
REM DIREWOLF Desktop Application Builder
REM Simplified build script for Windows

echo ========================================
echo DIREWOLF Desktop Application Builder
echo ========================================
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Running with Administrator privileges
) else (
    echo [WARNING] Not running as Administrator
    echo [INFO] Admin rights needed for installation to C:\DIREWOLF
    echo.
)

REM Set paths
set SOURCE_DIR=n:\CPPfiles\DRLHSS
set BUILD_DIR=%SOURCE_DIR%\build
set INSTALL_DIR=C:\DIREWOLF

echo [1/5] Checking dependencies...
echo.

REM Check for CMake
where cmake >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] CMake found
) else (
    echo [ERROR] CMake not found
    echo Please install CMake from https://cmake.org/download/
    pause
    exit /b 1
)

REM Check for Visual Studio
where cl >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Visual Studio compiler found
) else (
    echo [WARNING] Visual Studio compiler not in PATH
    echo Please run this from "Developer Command Prompt for VS 2022"
)

echo.
echo [2/5] Creating build directory...
if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
    echo [OK] Build directory created
) else (
    echo [OK] Build directory exists
)

echo.
echo [3/5] Configuring with CMake...
cd /d "%BUILD_DIR%"

REM Try to configure - if it fails, show helpful message
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" 2>cmake_error.txt
if %errorLevel% neq 0 (
    echo [ERROR] CMake configuration failed
    echo.
    echo Common issues:
    echo 1. SQLite3 not found - Download from https://www.sqlite.org/download.html
    echo 2. OpenSSL not found - Download from https://slproweb.com/products/Win32OpenSSL.html
    echo 3. ONNX Runtime not found - Check external/onnxruntime directory
    echo.
    echo See cmake_error.txt for details
    echo See DESKTOP_APP_BUILD_GUIDE.md for full setup instructions
    pause
    exit /b 1
)
echo [OK] CMake configuration successful

echo.
echo [4/5] Building the project...
cmake --build . --config Release --parallel
if %errorLevel% neq 0 (
    echo [ERROR] Build failed
    echo Please check the error messages above
    pause
    exit /b 1
)
echo [OK] Build successful

echo.
echo [5/5] Installation options...
echo.
echo Choose an option:
echo   1. Install to C:\DIREWOLF (requires admin)
echo   2. Run from build directory (development mode)
echo   3. Exit
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto rundev
if "%choice%"=="3" goto end

:install
echo.
echo Installing to %INSTALL_DIR%...
cmake --install . --config Release
if %errorLevel% neq 0 (
    echo [ERROR] Installation failed
    echo Make sure you're running as Administrator
    pause
    exit /b 1
)
echo [OK] Installation successful
echo.
echo DIREWOLF installed to: %INSTALL_DIR%
echo Executable: %INSTALL_DIR%\bin\direwolf.exe
echo.
echo To run DIREWOLF:
echo   %INSTALL_DIR%\bin\direwolf.exe
echo.
pause
goto end

:rundev
echo.
echo Running from build directory (development mode)...
echo.
echo Executable location: %BUILD_DIR%\Release\direwolf.exe
echo.
echo For development workflow:
echo   1. Make code changes in: %SOURCE_DIR%\src
echo   2. Rebuild: cd %BUILD_DIR% ^&^& cmake --build . --config Release
echo   3. Run: %BUILD_DIR%\Release\direwolf.exe
echo.
if exist "%BUILD_DIR%\Release\direwolf.exe" (
    echo Launching DIREWOLF...
    start "" "%BUILD_DIR%\Release\direwolf.exe"
) else (
    echo [WARNING] direwolf.exe not found
    echo The build may not have created the main executable yet
)
pause
goto end

:end
echo.
echo Build process complete!
echo.
