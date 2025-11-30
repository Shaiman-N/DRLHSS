# DIREWOLF Update Script
# This script rebuilds and creates a new installer for your existing DIREWOLF installation

Write-Host "🐺 DIREWOLF Update Script" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Clean duplicate folder
Write-Host "Step 1: Cleaning duplicate DRLHSS folder..." -ForegroundColor Yellow
if (Test-Path "DRLHSS") {
    Remove-Item -Recurse -Force "DRLHSS"
    Write-Host "✅ Duplicate folder removed" -ForegroundColor Green
} else {
    Write-Host "✅ No duplicate folder found" -ForegroundColor Green
}
Write-Host ""

# Step 2: Rebuild
Write-Host "Step 2: Rebuilding DIREWOLF..." -ForegroundColor Yellow
.\build_all.bat
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build successful" -ForegroundColor Green
} else {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 3: Create installer
Write-Host "Step 3: Creating installer..." -ForegroundColor Yellow
Push-Location installer
.\build_installer.bat
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Installer created successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Installer creation failed" -ForegroundColor Red
    Pop-Location
    exit 1
}
Pop-Location
Write-Host ""

# Step 4: Prompt to run installer
Write-Host "🎉 Update package ready!" -ForegroundColor Green
Write-Host ""
Write-Host "Your new installer is at:" -ForegroundColor Cyan
Write-Host "  N:\CPPfiles\DRLHSS\installer\direwolf_installer.exe" -ForegroundColor White
Write-Host ""
$response = Read-Host "Do you want to run the installer now? (Y/N)"
if ($response -eq 'Y' -or $response -eq 'y') {
    Write-Host "Launching installer..." -ForegroundColor Yellow
    Start-Process "installer\direwolf_installer.exe"
} else {
    Write-Host "You can run the installer manually later." -ForegroundColor Yellow
}
