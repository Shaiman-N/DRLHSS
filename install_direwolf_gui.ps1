# Complete DIREWOLF GUI Installation Script
# This script builds the GUI version and installs it with Start Menu shortcuts

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DIREWOLF GUI Complete Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set paths
$projectRoot = "N:\CPPfiles\DRLHSS"
$buildDir = Join-Path $projectRoot "build_gui"
$installDir = "$env:LOCALAPPDATA\DIREWOLF"
$startMenuPath = "$env:ProgramData\Microsoft\Windows\Start Menu\Programs\DIREWOLF"

# Step 1: Clean previous build
Write-Host "[1/6] Cleaning previous build..." -ForegroundColor Yellow
if (Test-Path $buildDir) {
    Remove-Item -Recurse -Force $buildDir
}
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null

# Step 2: Configure CMake for GUI
Write-Host "[2/6] Configuring CMake for GUI build..." -ForegroundColor Yellow
Set-Location $buildDir
cmake -S $projectRoot -B . -DCMAKE_BUILD_TYPE=Release -DBUILD_GUI=ON

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    exit 1
}

# Step 3: Build GUI
Write-Host "[3/6] Building DIREWOLF GUI..." -ForegroundColor Yellow
cmake --build . --config Release --target direwolf_gui

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Step 4: Create installation directory
Write-Host "[4/6] Creating installation directory..." -ForegroundColor Yellow
if (-not (Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
}

# Step 5: Copy files to installation directory
Write-Host "[5/6] Installing DIREWOLF GUI..." -ForegroundColor Yellow

# Copy executable
$guiExe = Join-Path $buildDir "Release\direwolf_gui.exe"
if (Test-Path $guiExe) {
    Copy-Item $guiExe $installDir -Force
    Write-Host "  Copied direwolf_gui.exe" -ForegroundColor Green
} else {
    Write-Host "  GUI executable not found at: $guiExe" -ForegroundColor Red
    exit 1
}

# Copy DLLs
$dllFiles = Get-ChildItem -Path (Join-Path $buildDir "Release") -Filter "*.dll"
foreach ($dll in $dllFiles) {
    Copy-Item $dll.FullName $installDir -Force
    Write-Host "  Copied $($dll.Name)" -ForegroundColor Green
}

# Copy QML files
$qmlSource = Join-Path $projectRoot "qml"
$qmlDest = Join-Path $installDir "qml"
if (Test-Path $qmlSource) {
    if (Test-Path $qmlDest) {
        Remove-Item -Recurse -Force $qmlDest
    }
    Copy-Item -Recurse $qmlSource $qmlDest -Force
    Write-Host "  Copied QML files" -ForegroundColor Green
}

# Copy config files
$configSource = Join-Path $projectRoot "config"
$configDest = Join-Path $installDir "config"
if (Test-Path $configSource) {
    if (Test-Path $configDest) {
        Remove-Item -Recurse -Force $configDest
    }
    Copy-Item -Recurse $configSource $configDest -Force
    Write-Host "  Copied config files" -ForegroundColor Green
}

# Copy Python XAI files
$pythonSource = Join-Path $projectRoot "python"
$pythonDest = Join-Path $installDir "python"
if (Test-Path $pythonSource) {
    if (Test-Path $pythonDest) {
        Remove-Item -Recurse -Force $pythonDest
    }
    Copy-Item -Recurse $pythonSource $pythonDest -Force
    Write-Host "  Copied Python XAI files" -ForegroundColor Green
}

# Step 6: Create Start Menu shortcut
Write-Host "[6/6] Creating Start Menu shortcut..." -ForegroundColor Yellow

if (-not (Test-Path $startMenuPath)) {
    New-Item -ItemType Directory -Path $startMenuPath -Force | Out-Null
}

$WScriptShell = New-Object -ComObject WScript.Shell
$shortcutPath = Join-Path $startMenuPath "DIREWOLF.lnk"
$shortcut = $WScriptShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $installDir "direwolf_gui.exe"
$shortcut.WorkingDirectory = $installDir
$shortcut.Description = "DIREWOLF Security System - GUI"
$shortcut.Save()

Write-Host "  Created Start Menu shortcut" -ForegroundColor Green

# Create desktop shortcut too
$desktopPath = [Environment]::GetFolderPath("Desktop")
$desktopShortcut = Join-Path $desktopPath "DIREWOLF.lnk"
$shortcut2 = $WScriptShell.CreateShortcut($desktopShortcut)
$shortcut2.TargetPath = Join-Path $installDir "direwolf_gui.exe"
$shortcut2.WorkingDirectory = $installDir
$shortcut2.Description = "DIREWOLF Security System - GUI"
$shortcut2.Save()

Write-Host "  Created Desktop shortcut" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "DIREWOLF GUI has been installed to:" -ForegroundColor Cyan
Write-Host "  $installDir" -ForegroundColor White
Write-Host ""
Write-Host "You can now:" -ForegroundColor Cyan
Write-Host "  1. Find DIREWOLF in your Start Menu" -ForegroundColor White
Write-Host "  2. Use the Desktop shortcut" -ForegroundColor White
Write-Host "  3. Run directly: $installDir\direwolf_gui.exe" -ForegroundColor White
Write-Host ""

# Ask if user wants to launch now
$launch = Read-Host "Would you like to launch DIREWOLF GUI now? (Y/N)"
if ($launch -eq "Y" -or $launch -eq "y") {
    Write-Host "Launching DIREWOLF GUI..." -ForegroundColor Cyan
    Start-Process (Join-Path $installDir "direwolf_gui.exe")
}
