# Create shortcuts for DIREWOLF GUI
$installDir = "$env:LOCALAPPDATA\DIREWOLF"
$exePath = Join-Path $installDir "direwolf_gui.exe"

if (-not (Test-Path $exePath)) {
    Write-Host "Error: direwolf_gui.exe not found at $exePath" -ForegroundColor Red
    exit 1
}

$WScriptShell = New-Object -ComObject WScript.Shell

# Create Start Menu shortcut
$startMenuPath = "$env:ProgramData\Microsoft\Windows\Start Menu\Programs"
if (-not (Test-Path $startMenuPath)) {
    New-Item -ItemType Directory -Path $startMenuPath -Force | Out-Null
}

$shortcut = $WScriptShell.CreateShortcut("$startMenuPath\DIREWOLF.lnk")
$shortcut.TargetPath = $exePath
$shortcut.WorkingDirectory = $installDir
$shortcut.Description = "DIREWOLF Security System"
$shortcut.Save()
Write-Host "Created Start Menu shortcut" -ForegroundColor Green

# Create Desktop shortcut
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcut2 = $WScriptShell.CreateShortcut("$desktopPath\DIREWOLF.lnk")
$shortcut2.TargetPath = $exePath
$shortcut2.WorkingDirectory = $installDir
$shortcut2.Description = "DIREWOLF Security System"
$shortcut2.Save()
Write-Host "Created Desktop shortcut" -ForegroundColor Green
