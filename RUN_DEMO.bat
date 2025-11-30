@echo off
REM DIREWOLF Demo Launcher
REM Easy way to run demonstrations

echo.
echo ========================================
echo DIREWOLF DEMONSTRATION LAUNCHER
echo ========================================
echo.
echo Select a demo to run:
echo.
echo 1. Complete System Demo (Full walkthrough)
echo 2. Live Traffic Monitor (Real-time packets)
echo 3. Live Traffic Monitor - Fast (3 packets/sec)
echo 4. Live Traffic Monitor - Continuous (Until stopped)
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Running Complete System Demo...
    echo.
    python demo_direwolf_simulation.py
    goto end
)

if "%choice%"=="2" (
    echo.
    echo Running Live Traffic Monitor (60 seconds, 2 packets/sec)...
    echo.
    python demo_live_traffic_monitor.py 60 2.0
    goto end
)

if "%choice%"=="3" (
    echo.
    echo Running Live Traffic Monitor - Fast (60 seconds, 3 packets/sec)...
    echo.
    python demo_live_traffic_monitor.py 60 3.0
    goto end
)

if "%choice%"=="4" (
    echo.
    echo Running Live Traffic Monitor - Continuous...
    echo Press Ctrl+C to stop
    echo.
    python demo_live_traffic_monitor.py 0 2.0
    goto end
)

if "%choice%"=="5" (
    echo.
    echo Exiting...
    goto end
)

echo.
echo Invalid choice. Please run again.

:end
echo.
pause
