@echo off
REM Complete DIREWOLF GUI Build and Installation
echo ========================================
echo DIREWOLF GUI Build and Installation
echo ========================================
echo.

cd /d N:\CPPfiles\DRLHSS

REM Clean previous build
echo [1/5] Cleaning previous build...
if exist build_gui rmdir /s /q build_gui
mkdir build_gui
cd build_gui

REM Configure with GUI CMakeLists
echo [2/5] Configuring CMake...
cmake -S .. -B . -DCMAKE_PREFIX_PATH=C:\Qt\6.5.0\msvc2019_64 -G "Visual Studio 17 2022" -A x64

if errorlevel 1 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

REM Build
echo [3/5] Building DIREWOLF GUI...
cmake --build . --config Release

if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

REM Install
echo [4/5] Installing to %LOCALAPPDATA%\DIREWOLF...
if not exist "%LOCALAPPDATA%\DIREWOLF" mkdir "%LOCALAPPDATA%\DIREWOLF"

REM Copy executable and DLLs
copy /Y Release\direwolf_gui.exe "%LOCALAPPDATA%\DIREWOLF\"
copy /Y Release\*.dll "%LOCALAPPDATA%\DIREWOLF\" 2>nul

REM Copy QML files
if exist ..\qml (
    if exist "%LOCALAPPDATA%\DIREWOLF\qml" rmdir /s /q "%LOCALAPPDATA%\DIREWOLF\qml"
    xcopy /E /I /Y ..\qml "%LOCALAPPDATA%\DIREWOLF\qml"
)

REM Copy config
if exist ..\config (
    if exist "%LOCALAPPDATA%\DIREWOLF\config" rmdir /s /q "%LOCALAPPDATA%\DIREWOLF\config"
    xcopy /E /I /Y ..\config "%LOCALAPPDATA%\DIREWOLF\config"
)

REM Copy Python files
if exist ..\python (
    if exist "%LOCALAPPDATA%\DIREWOLF\python" rmdir /s /q "%LOCALAPPDATA%\DIREWOLF\python"
    xcopy /E /I /Y ..\python "%LOCALAPPDATA%\DIREWOLF\python"
)

echo [5/5] Creating shortcuts...
powershell -ExecutionPolicy Bypass -File ..\create_gui_shortcuts.ps1

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo DIREWOLF GUI installed to: %LOCALAPPDATA%\DIREWOLF
echo Check your Start Menu for DIREWOLF!
echo.
pause
