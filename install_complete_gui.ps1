# Complete DIREWOLF GUI Installation with Qt Check
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DIREWOLF GUI Complete Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check for Qt installation
Write-Host "Checking for Qt6 installation..." -ForegroundColor Yellow

$qtPaths = @(
    "C:\Qt\6.5.0\msvc2019_64",
    "C:\Qt\6.6.0\msvc2019_64",
    "C:\Qt\6.7.0\msvc2019_64",
    "C:\Qt6\6.5.0\msvc2019_64",
    "$env:USERPROFILE\Qt\6.5.0\msvc2019_64"
)

$qtPath = $null
foreach ($path in $qtPaths) {
    if (Test-Path $path) {
        $qtPath = $path
        Write-Host "Found Qt at: $qtPath" -ForegroundColor Green
        break
    }
}

if (-not $qtPath) {
    Write-Host ""
    Write-Host "Qt6 is not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "To use DIREWOLF GUI, you need to install Qt6:" -ForegroundColor Yellow
    Write-Host "1. Download from: https://www.qt.io/download-qt-installer" -ForegroundColor White
    Write-Host "2. Run the installer" -ForegroundColor White
    Write-Host "3. Select Qt 6.5.0 or later" -ForegroundColor White
    Write-Host "4. Choose 'MSVC 2019 64-bit' component" -ForegroundColor White
    Write-Host "5. Complete installation" -ForegroundColor White
    Write-Host "6. Run this script again" -ForegroundColor White
    Write-Host ""
    
    $openBrowser = Read-Host "Would you like to open the Qt download page now? (Y/N)"
    if ($openBrowser -eq "Y" -or $openBrowser -eq "y") {
        Start-Process "https://www.qt.io/download-qt-installer"
    }
    
    Write-Host ""
    Write-Host "For now, you can use the CLI version:" -ForegroundColor Cyan
    Write-Host "  N:\CPPfiles\DRLHSS\build\Release\direwolf.exe" -ForegroundColor White
    Write-Host ""
    pause
    exit 1
}

# Set paths
$projectRoot = "N:\CPPfiles\DRLHSS"
$buildDir = Join-Path $projectRoot "build_gui"
$installDir = "$env:LOCALAPPDATA\DIREWOLF"

# Clean and create build directory
Write-Host ""
Write-Host "[1/6] Preparing build directory..." -ForegroundColor Yellow
if (Test-Path $buildDir) {
    Remove-Item -Recurse -Force $buildDir
}
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null

# Configure CMake
Write-Host "[2/6] Configuring CMake..." -ForegroundColor Yellow
Set-Location $buildDir

$cmakeArgs = @(
    "-S", $projectRoot,
    "-B", ".",
    "-DCMAKE_PREFIX_PATH=$qtPath",
    "-G", "Visual Studio 17 2022",
    "-A", "x64"
)

& cmake $cmakeArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    Write-Host "Make sure Visual Studio 2022 is installed" -ForegroundColor Yellow
    pause
    exit 1
}

# Build
Write-Host "[3/6] Building DIREWOLF GUI (this may take a few minutes)..." -ForegroundColor Yellow
cmake --build . --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    pause
    exit 1
}

# Create installation directory
Write-Host "[4/6] Creating installation directory..." -ForegroundColor Yellow
if (-not (Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
}

# Install files
Write-Host "[5/6] Installing DIREWOLF GUI..." -ForegroundColor Yellow

# Copy executable
$guiExe = Join-Path $buildDir "Release\direwolf_gui.exe"
if (Test-Path $guiExe) {
    Copy-Item $guiExe $installDir -Force
    Write-Host "  ✓ Copied direwolf_gui.exe" -ForegroundColor Green
} else {
    Write-Host "  ✗ GUI executable not found!" -ForegroundColor Red
    pause
    exit 1
}

# Copy DLLs
$dllFiles = Get-ChildItem -Path (Join-Path $buildDir "Release") -Filter "*.dll" -ErrorAction SilentlyContinue
foreach ($dll in $dllFiles) {
    Copy-Item $dll.FullName $installDir -Force
    Write-Host "  ✓ Copied $($dll.Name)" -ForegroundColor Green
}

# Run windeployqt to copy Qt dependencies
$windeployqt = Join-Path $qtPath "bin\windeployqt.exe"
if (Test-Path $windeployqt) {
    Write-Host "  Deploying Qt dependencies..." -ForegroundColor Cyan
    $qmlDir = Join-Path $projectRoot "qml"
    & $windeployqt --qmldir $qmlDir (Join-Path $installDir "direwolf_gui.exe")
    Write-Host "  ✓ Qt dependencies deployed" -ForegroundColor Green
}

# Copy QML files
$qmlSource = Join-Path $projectRoot "qml"
$qmlDest = Join-Path $installDir "qml"
if (Test-Path $qmlSource) {
    if (Test-Path $qmlDest) {
        Remove-Item -Recurse -Force $qmlDest
    }
    Copy-Item -Recurse $qmlSource $qmlDest -Force
    Write-Host "  ✓ Copied QML files" -ForegroundColor Green
}

# Copy config
$configSource = Join-Path $projectRoot "config"
$configDest = Join-Path $installDir "config"
if (Test-Path $configSource) {
    if (Test-Path $configDest) {
        Remove-Item -Recurse -Force $configDest
    }
    Copy-Item -Recurse $configSource $configDest -Force
    Write-Host "  ✓ Copied config files" -ForegroundColor Green
}

# Copy Python files
$pythonSource = Join-Path $projectRoot "python"
$pythonDest = Join-Path $installDir "python"
if (Test-Path $pythonSource) {
    if (Test-Path $pythonDest) {
        Remove-Item -Recurse -Force $pythonDest
    }
    Copy-Item -Recurse $pythonSource $pythonDest -Force
    Write-Host "  ✓ Copied Python XAI files" -ForegroundColor Green
}

# Create shortcuts
Write-Host "[6/6] Creating shortcuts..." -ForegroundColor Yellow

$WScriptShell = New-Object -ComObject WScript.Shell

# Start Menu shortcut
$startMenuPath = "$env:ProgramData\Microsoft\Windows\Start Menu\Programs"
$shortcut = $WScriptShell.CreateShortcut("$startMenuPath\DIREWOLF.lnk")
$shortcut.TargetPath = Join-Path $installDir "direwolf_gui.exe"
$shortcut.WorkingDirectory = $installDir
$shortcut.Description = "DIREWOLF Security System"
$shortcut.Save()
Write-Host "  ✓ Created Start Menu shortcut" -ForegroundColor Green

# Desktop shortcut
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcut2 = $WScriptShell.CreateShortcut("$desktopPath\DIREWOLF.lnk")
$shortcut2.TargetPath = Join-Path $installDir "direwolf_gui.exe"
$shortcut2.WorkingDirectory = $installDir
$shortcut2.Description = "DIREWOLF Security System"
$shortcut2.Save()
Write-Host "  ✓ Created Desktop shortcut" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "DIREWOLF GUI has been installed to:" -ForegroundColor Cyan
Write-Host "  $installDir" -ForegroundColor White
Write-Host ""
Write-Host "You can now launch DIREWOLF from:" -ForegroundColor Cyan
Write-Host "  • Start Menu" -ForegroundColor White
Write-Host "  • Desktop shortcut" -ForegroundColor White
Write-Host "  • Directly: $installDir\direwolf_gui.exe" -ForegroundColor White
Write-Host ""

$launch = Read-Host "Would you like to launch DIREWOLF GUI now? (Y/N)"
if ($launch -eq "Y" -or $launch -eq "y") {
    Write-Host "Launching DIREWOLF GUI..." -ForegroundColor Cyan
    Start-Process (Join-Path $installDir "direwolf_gui.exe")
}

Write-Host ""
Write-Host "Enjoy DIREWOLF!" -ForegroundColor Green
