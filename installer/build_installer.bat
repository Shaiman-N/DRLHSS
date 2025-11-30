@echo off
REM DIREWOLF Installer Builder
REM Creates Windows installer using NSIS

echo ========================================
echo DIREWOLF Installer Builder
echo ========================================
echo.

REM Check for NSIS
where makensis >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] NSIS not found!
    echo.
    echo Please install NSIS from: https://nsis.sourceforge.io/Download
    echo.
    echo After installation, add NSIS to your PATH or run this script from NSIS directory.
    echo.
    pause
    exit /b 1
)

echo [OK] NSIS found
echo.

REM Check if executable exists
set EXE_PATH=..\build_desktop\Release\direwolf.exe
if not exist "%EXE_PATH%" (
    echo [ERROR] DIREWOLF executable not found!
    echo.
    echo Please build DIREWOLF first:
    echo   cd n:\CPPfiles\DRLHSS
    echo   .\build_desktop_simple.bat
    echo.
    pause
    exit /b 1
)

echo [OK] DIREWOLF executable found
echo.

REM Create LICENSE.txt if it doesn't exist
if not exist "..\LICENSE" (
    echo Creating LICENSE file...
    echo MIT License > ..\LICENSE
    echo. >> ..\LICENSE
    echo Copyright (c) 2024 DIREWOLF Security >> ..\LICENSE
    echo. >> ..\LICENSE
    echo Permission is hereby granted, free of charge, to any person obtaining a copy >> ..\LICENSE
    echo of this software and associated documentation files (the "Software"), to deal >> ..\LICENSE
    echo in the Software without restriction, including without limitation the rights >> ..\LICENSE
    echo to use, copy, modify, merge, publish, distribute, sublicense, and/or sell >> ..\LICENSE
    echo copies of the Software, and to permit persons to whom the Software is >> ..\LICENSE
    echo furnished to do so, subject to the following conditions: >> ..\LICENSE
    echo. >> ..\LICENSE
    echo The above copyright notice and this permission notice shall be included in all >> ..\LICENSE
    echo copies or substantial portions of the Software. >> ..\LICENSE
    echo. >> ..\LICENSE
    echo THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR >> ..\LICENSE
    echo IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, >> ..\LICENSE
    echo FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE >> ..\LICENSE
    echo AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER >> ..\LICENSE
    echo LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, >> ..\LICENSE
    echo OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE >> ..\LICENSE
    echo SOFTWARE. >> ..\LICENSE
    echo [OK] LICENSE created
)

echo Building installer...
echo.

REM Build the installer
makensis /V3 direwolf_installer.nsi

if %errorLevel% neq 0 (
    echo.
    echo [ERROR] Installer build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installer Build Complete!
echo ========================================
echo.
echo Installer created: DIREWOLF_Setup_v1.0.0.exe
echo.
echo To install DIREWOLF:
echo   1. Right-click DIREWOLF_Setup_v1.0.0.exe
echo   2. Select "Run as Administrator"
echo   3. Follow the installation wizard
echo.
echo The installer will:
echo   - Install DIREWOLF to Program Files
echo   - Create Start Menu shortcuts
echo   - Optionally create Desktop shortcut
echo   - Optionally install as Windows Service
echo   - Optionally enable auto-start
echo.
pause
