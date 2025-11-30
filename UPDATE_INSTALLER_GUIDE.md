# 🔄 Update Your Installed DIREWOLF - Fixed Guide

## Issue 1: PowerShell Syntax ✅

In PowerShell, you need `.\` before batch files:

```powershell
# ❌ WRONG
build_all.bat

# ✅ CORRECT
.\build_all.bat
```

## Issue 2: Duplicate DRLHSS Folder ✅

Your project has a duplicate `DRLHSS/DRLHSS/` folder that needs to be removed.

---

## 🚀 Complete Update Process

### Step 1: Clean Up Duplicate Folder

```powershell
# Remove the duplicate DRLHSS folder
Remove-Item -Recurse -Force "DRLHSS"
```

### Step 2: Rebuild Everything

```powershell
# Navigate to your project
cd N:\CPPfiles\DRLHSS

# Rebuild with correct PowerShell syntax
.\build_all.bat
```

### Step 3: Create New Installer

```powershell
# Go to installer directory
cd installer

# Build the installer
.\build_installer.bat
```

### Step 4: Run the Installer

The installer will be created in the `installer` directory. You can run it:

```powershell
# Option 1: Run from PowerShell
.\direwolf_installer.exe

# Option 2: Double-click the file in Windows Explorer
# Navigate to: N:\CPPfiles\DRLHSS\installer\direwolf_installer.exe
```

---

## 📋 What the Installer Does

When you run the new installer on an existing installation:

✅ **Detects** your current DIREWOLF installation  
✅ **Preserves** all your settings and data  
✅ **Upgrades** binaries to the new version  
✅ **Maintains** your configuration files  

---

## 🔧 Alternative: Quick Manual Update

If you just want to test without running the full installer:

```powershell
# 1. Build
cd N:\CPPfiles\DRLHSS
.\build_all.bat

# 2. Copy to installation directory (adjust path if needed)
$installPath = "C:\Program Files\DIREWOLF"
Copy-Item "build\*.exe" $installPath -Force
Copy-Item "build\*.dll" $installPath -Force
```

---

## 🎯 Quick Commands (Copy & Paste)

```powershell
# Full update process
cd N:\CPPfiles\DRLHSS
Remove-Item -Recurse -Force "DRLHSS"
.\build_all.bat
cd installer
.\build_installer.bat
.\direwolf_installer.exe
```

---

## ❓ Troubleshooting

**If build_all.bat fails:**
- Make sure you have CMake and Visual Studio installed
- Check that all dependencies are available

**If the installer doesn't detect your installation:**
- The installer looks in `C:\Program Files\DIREWOLF`
- If you installed elsewhere, you may need to uninstall first

**If you get permission errors:**
- Run PowerShell as Administrator
- Right-click PowerShell → "Run as Administrator"
