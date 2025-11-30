@echo off
echo ========================================
echo DIREWOLF GUI Dashboard Build Script
echo ========================================
echo.

REM Set Qt path (adjust this to your Qt installation)
set QT_PATH=C:\Qt\6.5.3\msvc2019_64
set PATH=%QT_PATH%\bin;%PATH%

REM Create build directory
if not exist "build_gui" mkdir build_gui
cd build_gui

echo Configuring CMake...
cmake -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_PREFIX_PATH=%QT_PATH% ^
    -DCMAKE_BUILD_TYPE=Release ^
    -S .. -B . ^
    -C ../CMakeLists_gui.txt

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

echo.
echo Building DIREWOLF GUI...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo Executable: build_gui\bin\Release\direwolf_gui.exe
echo ========================================
echo.

pause
