@echo off
REM DIREWOLF Desktop Application - Simple Builder
REM Builds minimal desktop app without heavy dependencies

echo ========================================
echo DIREWOLF Desktop Application Builder
echo Simple Build (No External Dependencies)
echo ========================================
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Running with Administrator privileges
) else (
    echo [WARNING] Not running as Administrator
    echo [INFO] Admin rights recommended for installation to C:\DIREWOLF
    echo.
)

REM Set paths
set SOURCE_DIR=n:\CPPfiles\DRLHSS
set BUILD_DIR=%SOURCE_DIR%\build_desktop
set INSTALL_DIR=C:\DIREWOLF

echo [1/4] Creating build directory...
if exist "%BUILD_DIR%" (
    echo [INFO] Cleaning existing build directory...
    rmdir /s /q "%BUILD_DIR%"
)
mkdir "%BUILD_DIR%"
echo [OK] Build directory created

echo.
echo [2/4] Configuring with CMake...
cd /d "%BUILD_DIR%"

REM Use the minimal desktop CMakeLists
copy /Y "%SOURCE_DIR%\CMakeLists_desktop.txt" "%SOURCE_DIR%\CMakeLists.txt.backup" >nul
copy /Y "%SOURCE_DIR%\CMakeLists_desktop.txt" "%SOURCE_DIR%\CMakeLists.txt" >nul

cmake "%SOURCE_DIR%" -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"

if %errorLevel% neq 0 (
    echo [ERROR] CMake configuration failed
    pause
    exit /b 1
)
echo [OK] CMake configuration successful

echo.
echo [3/4] Building the project...
cmake --build . --config Release --parallel

if %errorLevel% neq 0 (
    echo [ERROR] Build failed
    pause
    exit /b 1
)
echo [OK] Build successful

echo.
echo [4/4] Installation...
echo.
echo Executable built at: %BUILD_DIR%\Release\direwolf.exe
echo.
echo Choose an option:
echo   1. Install to C:\DIREWOLF (requires admin)
echo   2. Run from build directory
echo   3. Setup admin account
echo   4. Exit
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto rundev
if "%choice%"=="3" goto setup
if "%choice%"=="4" goto end

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
echo ========================================
echo DIREWOLF Installation Complete!
echo ========================================
echo.
echo Installation Directory: %INSTALL_DIR%
echo Executable: %INSTALL_DIR%\bin\direwolf.exe
echo.
echo To run DIREWOLF:
echo   %INSTALL_DIR%\bin\direwolf.exe
echo.
echo To setup admin account:
echo   %INSTALL_DIR%\bin\direwolf.exe --setup-admin
echo.
pause
goto end

:rundev
echo.
echo Running from build directory...
echo.
if exist "%BUILD_DIR%\Release\direwolf.exe" (
    "%BUILD_DIR%\Release\direwolf.exe"
) else (
    echo [ERROR] direwolf.exe not found
    echo Build may have failed
)
pause
goto end

:setup
echo.
echo Running admin setup...
echo.
if exist "%BUILD_DIR%\Release\direwolf.exe" (
    "%BUILD_DIR%\Release\direwolf.exe" --setup-admin
) else (
    echo [ERROR] direwolf.exe not found
    echo Build may have failed
)
pause
goto end

:end
echo.
echo Build process complete!
echo.
echo Development Workflow:
echo   1. Make code changes in: %SOURCE_DIR%\src
echo   2. Rebuild: cd %BUILD_DIR% ^&^& cmake --build . --config Release
echo   3. Run: %BUILD_DIR%\Release\direwolf.exe
echo.
