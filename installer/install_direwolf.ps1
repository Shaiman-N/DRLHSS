# DIREWOLF Security System - PowerShell Installer
# Simple installer that doesn't require NSIS or WiX

param(
    [switch]$Uninstall,
    [switch]$Silent
)

$ErrorActionPreference = "Stop"

# Configuration
$AppName = "DIREWOLF Security System"
$InstallDir = "C:\Program Files\DIREWOLF"
$SourceDir = "..\build_desktop\Release"
$ExeName = "direwolf.exe"

# Colors
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

# Check admin privileges
function Test-Administrator {
    $user = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($user)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Banner
function Show-Banner {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  DIREWOLF Security System Installer" -ForegroundColor Cyan
    Write-Host "  Version 1.0.0" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

# Install function
function Install-Direwolf {
    Show-Banner
    
    # Check admin
    if (-not (Test-Administrator)) {
        Write-Error "[ERROR] Administrator privileges required!"
        Write-Info "Please run PowerShell as Administrator and try again."
        exit 1
    }
    
    Write-Success "[OK] Running with Administrator privileges"
    Write-Host ""
    
    # Check source files
    $exePath = Join-Path $SourceDir $ExeName
    if (-not (Test-Path $exePath)) {
        Write-Error "[ERROR] DIREWOLF executable not found!"
        Write-Info "Expected location: $exePath"
        Write-Info ""
        Write-Info "Please build DIREWOLF first:"
        Write-Info "  cd n:\CPPfiles\DRLHSS"
        Write-Info "  .\build_desktop_simple.bat"
        exit 1
    }
    
    Write-Success "[OK] DIREWOLF executable found"
    Write-Host ""
    
    # Check if already installed
    if (Test-Path $InstallDir) {
        if (-not $Silent) {
            $response = Read-Host "DIREWOLF is already installed. Reinstall? (Y/N)"
            if ($response -ne 'Y' -and $response -ne 'y') {
                Write-Info "Installation cancelled."
                exit 0
            }
        }
        Write-Info "Removing existing installation..."
        Remove-Item -Path $InstallDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    # Create installation directory
    Write-Info "[1/7] Creating installation directory..."
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    New-Item -ItemType Directory -Path "$InstallDir\bin" -Force | Out-Null
    New-Item -ItemType Directory -Path "$InstallDir\config" -Force | Out-Null
    New-Item -ItemType Directory -Path "$InstallDir\logs" -Force | Out-Null
    New-Item -ItemType Directory -Path "$InstallDir\data" -Force | Out-Null
    New-Item -ItemType Directory -Path "$InstallDir\models" -Force | Out-Null
    Write-Success "[OK] Directories created"
    
    # Copy executable
    Write-Info "[2/7] Copying DIREWOLF executable..."
    Copy-Item -Path $exePath -Destination "$InstallDir\bin\$ExeName" -Force
    Write-Success "[OK] Executable copied"
    
    # Copy documentation
    Write-Info "[3/7] Copying documentation..."
    if (Test-Path "..\DESKTOP_APP_QUICK_START.md") {
        Copy-Item -Path "..\DESKTOP_APP_QUICK_START.md" -Destination "$InstallDir\README.txt" -Force
    }
    if (Test-Path "..\LICENSE") {
        Copy-Item -Path "..\LICENSE" -Destination "$InstallDir\LICENSE.txt" -Force
    }
    Write-Success "[OK] Documentation copied"
    
    # Create Start Menu shortcuts
    Write-Info "[4/7] Creating Start Menu shortcuts..."
    $startMenuPath = "$env:ProgramData\Microsoft\Windows\Start Menu\Programs\DIREWOLF"
    New-Item -ItemType Directory -Path $startMenuPath -Force | Out-Null
    
    $WshShell = New-Object -ComObject WScript.Shell
    
    # Main shortcut
    $shortcut = $WshShell.CreateShortcut("$startMenuPath\DIREWOLF.lnk")
    $shortcut.TargetPath = "$InstallDir\bin\$ExeName"
    $shortcut.WorkingDirectory = "$InstallDir\bin"
    $shortcut.Description = "DIREWOLF Security System"
    $shortcut.Save()
    
    # Setup Admin shortcut
    $shortcut = $WshShell.CreateShortcut("$startMenuPath\Setup Admin.lnk")
    $shortcut.TargetPath = "$InstallDir\bin\$ExeName"
    $shortcut.Arguments = "--setup-admin"
    $shortcut.WorkingDirectory = "$InstallDir\bin"
    $shortcut.Description = "Setup DIREWOLF Administrator Account"
    $shortcut.Save()
    
    # README shortcut
    if (Test-Path "$InstallDir\README.txt") {
        $shortcut = $WshShell.CreateShortcut("$startMenuPath\README.lnk")
        $shortcut.TargetPath = "$InstallDir\README.txt"
        $shortcut.Save()
    }
    
    Write-Success "[OK] Start Menu shortcuts created"
    
    # Create Desktop shortcut
    Write-Info "[5/7] Creating Desktop shortcut..."
    if (-not $Silent) {
        $response = Read-Host "Create Desktop shortcut? (Y/N)"
        if ($response -eq 'Y' -or $response -eq 'y') {
            $desktopPath = [Environment]::GetFolderPath("Desktop")
            $shortcut = $WshShell.CreateShortcut("$desktopPath\DIREWOLF.lnk")
            $shortcut.TargetPath = "$InstallDir\bin\$ExeName"
            $shortcut.WorkingDirectory = "$InstallDir\bin"
            $shortcut.Description = "DIREWOLF Security System"
            $shortcut.Save()
            Write-Success "[OK] Desktop shortcut created"
        }
    }
    
    # Add to PATH
    Write-Info "[6/7] Adding to system PATH..."
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    if ($currentPath -notlike "*$InstallDir\bin*") {
        [Environment]::SetEnvironmentVariable("Path", "$currentPath;$InstallDir\bin", "Machine")
        Write-Success "[OK] Added to PATH"
    } else {
        Write-Info "[OK] Already in PATH"
    }
    
    # Registry entries
    Write-Info "[7/7] Creating registry entries..."
    $regPath = "HKLM:\Software\DIREWOLF"
    New-Item -Path $regPath -Force | Out-Null
    Set-ItemProperty -Path $regPath -Name "InstallDir" -Value $InstallDir
    Set-ItemProperty -Path $regPath -Name "Version" -Value "1.0.0"
    Set-ItemProperty -Path $regPath -Name "InstallDate" -Value (Get-Date -Format "yyyy-MM-dd")
    
    # Uninstall registry
    $uninstallPath = "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\DIREWOLF"
    New-Item -Path $uninstallPath -Force | Out-Null
    Set-ItemProperty -Path $uninstallPath -Name "DisplayName" -Value $AppName
    Set-ItemProperty -Path $uninstallPath -Name "DisplayVersion" -Value "1.0.0"
    Set-ItemProperty -Path $uninstallPath -Name "Publisher" -Value "DIREWOLF Security"
    Set-ItemProperty -Path $uninstallPath -Name "InstallLocation" -Value $InstallDir
    Set-ItemProperty -Path $uninstallPath -Name "UninstallString" -Value "powershell.exe -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Uninstall"
    Set-ItemProperty -Path $uninstallPath -Name "DisplayIcon" -Value "$InstallDir\bin\$ExeName"
    Set-ItemProperty -Path $uninstallPath -Name "NoModify" -Value 1 -Type DWord
    Set-ItemProperty -Path $uninstallPath -Name "NoRepair" -Value 1 -Type DWord
    
    Write-Success "[OK] Registry entries created"
    
    # Installation complete
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Installation Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Info "Installation Directory: $InstallDir"
    Write-Info "Executable: $InstallDir\bin\$ExeName"
    Write-Host ""
    Write-Info "To run DIREWOLF:"
    Write-Info "  1. Start Menu → DIREWOLF"
    Write-Info "  2. Desktop shortcut (if created)"
    Write-Info "  3. Command: direwolf"
    Write-Host ""
    Write-Info "To setup admin account:"
    Write-Info "  direwolf --setup-admin"
    Write-Host ""
    
    if (-not $Silent) {
        $response = Read-Host "Would you like to setup your admin account now? (Y/N)"
        if ($response -eq 'Y' -or $response -eq 'y') {
            Start-Process -FilePath "$InstallDir\bin\$ExeName" -ArgumentList "--setup-admin" -Wait
        }
    }
}

# Uninstall function
function Uninstall-Direwolf {
    Show-Banner
    
    # Check admin
    if (-not (Test-Administrator)) {
        Write-Error "[ERROR] Administrator privileges required!"
        exit 1
    }
    
    Write-Info "Uninstalling DIREWOLF Security System..."
    Write-Host ""
    
    if (-not (Test-Path $InstallDir)) {
        Write-Warning "DIREWOLF is not installed."
        exit 0
    }
    
    if (-not $Silent) {
        $response = Read-Host "Are you sure you want to uninstall DIREWOLF? (Y/N)"
        if ($response -ne 'Y' -and $response -ne 'y') {
            Write-Info "Uninstall cancelled."
            exit 0
        }
    }
    
    # Stop service if running
    Write-Info "[1/5] Stopping Windows service..."
    Stop-Service -Name "DIREWOLF" -ErrorAction SilentlyContinue
    sc.exe delete DIREWOLF 2>$null
    Write-Success "[OK] Service stopped"
    
    # Remove shortcuts
    Write-Info "[2/5] Removing shortcuts..."
    $startMenuPath = "$env:ProgramData\Microsoft\Windows\Start Menu\Programs\DIREWOLF"
    if (Test-Path $startMenuPath) {
        Remove-Item -Path $startMenuPath -Recurse -Force
    }
    $desktopShortcut = [Environment]::GetFolderPath("Desktop") + "\DIREWOLF.lnk"
    if (Test-Path $desktopShortcut) {
        Remove-Item -Path $desktopShortcut -Force
    }
    Write-Success "[OK] Shortcuts removed"
    
    # Remove from PATH
    Write-Info "[3/5] Removing from PATH..."
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $newPath = ($currentPath.Split(';') | Where-Object { $_ -notlike "*$InstallDir\bin*" }) -join ';'
    [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
    Write-Success "[OK] Removed from PATH"
    
    # Remove registry entries
    Write-Info "[4/5] Removing registry entries..."
    Remove-Item -Path "HKLM:\Software\DIREWOLF" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\DIREWOLF" -Force -ErrorAction SilentlyContinue
    Write-Success "[OK] Registry entries removed"
    
    # Remove installation directory
    Write-Info "[5/5] Removing installation directory..."
    Remove-Item -Path $InstallDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Success "[OK] Installation directory removed"
    
    Write-Host ""
    Write-Success "DIREWOLF has been successfully uninstalled."
    Write-Host ""
}

# Main
try {
    if ($Uninstall) {
        Uninstall-Direwolf
    } else {
        Install-Direwolf
    }
} catch {
    Write-Error "An error occurred: $_"
    exit 1
}
