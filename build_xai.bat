@echo off
echo ================================================
echo   DIREWOLF XAI - Phase 1 Foundation Build
echo ================================================
echo.

REM Create build directory
if not exist "build_xai" mkdir build_xai
cd build_xai

echo Configuring CMake...
cmake -G "Visual Studio 17 2022" -A x64 -S .. -B . -DCMAKE_PREFIX_PATH=%Qt6_DIR% -f ../CMakeLists_xai.txt

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: CMake configuration failed!
    echo.
    echo Please ensure:
    echo   1. Qt6 is installed
    echo   2. Qt6_DIR environment variable is set
    echo   3. Visual Studio 2022 is installed
    echo.
    pause
    exit /b 1
)

echo.
echo Building project...
cmake --build . --config Release --parallel

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Build Complete!
echo ================================================
echo.
echo Executable location: build_xai\Release\direwolf_xai.exe
echo.
echo To run the application:
echo   cd build_xai\Release
echo   direwolf_xai.exe
echo.
pause
