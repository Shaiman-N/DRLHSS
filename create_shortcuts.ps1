# Create Start Menu Shortcuts for DIREWOLF
# Run this script as Administrator

$ErrorActionPreference = "Stop"

Write-Host "Creating DIREWOLF Start Menu Shortcuts..." -ForegroundColor Cyan

# Find where the executables are
$buildPath = "N:\CPPfiles\DRLHSS\build\Release"
$startMenuPath = "$env:ProgramData\Microsoft\Windows\Start Menu\Programs\DIREWOLF"

# Create Start Menu folder
if (-not (Test-Path $startMenuPath)) {
    New-Item -ItemType Directory -Path $startMenuPath -Force | Out-Null
    Write-Host "Created Start Menu folder: $startMenuPath" -ForegroundColor Green
}

# Function to create shortcut
function Create-Shortcut {
    param(
        [string]$ExeName,
        [string]$ShortcutName,
        [string]$Description
    )
    
    $exePath = Join-Path $buildPath $ExeName
    
    if (Test-Path $exePath) {
        $WScriptShell = New-Object -ComObject WScript.Shell
        $shortcutPath = Join-Path $startMenuPath "$ShortcutName.lnk"
        $shortcut = $WScriptShell.CreateShortcut($shortcutPath)
        $shortcut.TargetPath = $exePath
        $shortcut.WorkingDirectory = $buildPath
        $shortcut.Description = $Description
        $shortcut.Save()
        
        Write-Host "Created shortcut: $ShortcutName" -ForegroundColor Green
        return $true
    } else {
        Write-Host "Executable not found: $exePath" -ForegroundColor Yellow
        return $false
    }
}

# Create shortcuts for different DIREWOLF components
$created = 0

if (Create-Shortcut "direwolf.exe" "DIREWOLF" "DIREWOLF Security System") { $created++ }
if (Create-Shortcut "direwolf_gui.exe" "DIREWOLF GUI" "DIREWOLF Graphical Interface") { $created++ }
if (Create-Shortcut "direwolf_xai.exe" "DIREWOLF XAI" "DIREWOLF XAI Assistant") { $created++ }
if (Create-Shortcut "direwolf_desktop.exe" "DIREWOLF Desktop" "DIREWOLF Desktop Application") { $created++ }

Write-Host "`nCreated $created shortcut(s)" -ForegroundColor Cyan
Write-Host "Check your Start Menu for DIREWOLF!" -ForegroundColor Green
