@echo off
REM Quick test of the demo scripts

echo Testing DIREWOLF Demo Scripts...
echo.

echo Test 1: Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)
echo OK: Python is installed
echo.

echo Test 2: Running 10-second traffic monitor test...
echo.
python demo_live_traffic_monitor.py 10 3.0

echo.
echo.
echo ========================================
echo Test Complete!
echo ========================================
echo.
echo If you saw network packets above, the demo is working!
echo.
echo To run full demos:
echo   - RUN_DEMO.bat (interactive menu)
echo   - python demo_direwolf_simulation.py (complete demo)
echo   - python demo_live_traffic_monitor.py (live traffic)
echo.
pause
