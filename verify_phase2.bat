@echo off
echo ================================================
echo   DIREWOLF XAI Phase 2 - Build Verification
echo ================================================
echo.

echo Building Phase 1 + 2...
call build_xai.bat

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Running Phase 2 Voice Test
echo ================================================
echo.

cd build_xai\Release
direwolf_xai_voice.exe

echo.
echo ================================================
echo   Phase 2 Verification Complete!
echo ================================================
echo.
echo If you saw:
echo   - Application initialized successfully
echo   - Text-to-Speech working
echo   - Speech Recognition working
echo   - All tests passed
echo.
echo Then Phase 2 is READY TO DEPLOY!
echo.
pause
