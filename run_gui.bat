@echo off
echo ========================================
echo DIREWOLF GUI Dashboard Launcher
echo ========================================
echo.

REM Check if executable exists
if not exist "build_gui\bin\Release\direwolf_gui.exe" (
    echo Error: GUI executable not found!
    echo Please run build_gui.bat first.
    pause
    exit /b 1
)

echo Starting DIREWOLF GUI Dashboard...
echo.

REM Launch the application
start "" "build_gui\bin\Release\direwolf_gui.exe"

echo GUI launched successfully!
echo.
