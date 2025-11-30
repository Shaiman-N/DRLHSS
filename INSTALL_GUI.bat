@echo off
REM Simple launcher for DIREWOLF GUI installation
echo.
echo ========================================
echo DIREWOLF GUI Installation
echo ========================================
echo.
echo This will install DIREWOLF with a graphical interface.
echo.
pause

powershell -ExecutionPolicy Bypass -File "%~dp0install_complete_gui.ps1"

pause
