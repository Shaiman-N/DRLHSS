# DIREWOLF Installers

This directory contains multiple installer options for DIREWOLF Security System.

## 🚀 Quick Start

### Easiest Method (Recommended):

```powershell
# Go to parent directory and run one-click installer
cd ..
# Right-click INSTALL_DIREWOLF.bat and "Run as Administrator"
```

Or from PowerShell:
```powershell
cd ..
.\INSTALL_DIREWOLF.bat
```

## 📦 Installer Files

### 1. PowerShell Installer
- **File:** `install_direwolf.ps1`
- **Usage:** `.\install_direwolf.ps1`
- **Requirements:** None (built into Windows)
- **Best for:** Personal use, quick installation

### 2. NSIS Installer
- **Files:** `direwolf_installer.nsi`, `build_installer.bat`
- **Usage:** `.\build_installer.bat`
- **Requirements:** NSIS (https://nsis.sourceforge.io/Download)
- **Best for:** Distribution, professional installer

### 3. WiX Installer
- **File:** `direwolf.wxs`
- **Usage:** `candle direwolf.wxs && light -ext WixUIExtension direwolf.wixobj`
- **Requirements:** WiX Toolset (https://wixtoolset.org/)
- **Best for:** Enterprise deployment, MSI packages

## 📖 Documentation

- **INSTALLER_GUIDE.md** - Complete installation guide
- **../INSTALLER_COMPLETE.md** - Overview and comparison

## 🎯 Which Installer Should I Use?

| Scenario | Recommended Installer |
|----------|----------------------|
| Installing on your PC | One-Click or PowerShell |
| Sharing with friends | NSIS |
| Corporate deployment | WiX |
| Quick testing | PowerShell |
| Professional distribution | NSIS |

## ⚡ Quick Commands

```powershell
# Install with PowerShell
.\install_direwolf.ps1

# Uninstall with PowerShell
.\install_direwolf.ps1 -Uninstall

# Build NSIS installer
.\build_installer.bat

# Build WiX installer
candle direwolf.wxs
light -ext WixUIExtension direwolf.wixobj -out DIREWOLF.msi
```

## 📋 Prerequisites

Before running any installer:

1. **Build DIREWOLF first:**
   ```powershell
   cd ..
   .\build_desktop_simple.bat
   ```

2. **Run as Administrator**
   - Right-click → "Run as Administrator"
   - Or use elevated PowerShell

## ✅ What Gets Installed

- **Location:** `C:\Program Files\DIREWOLF`
- **Executable:** `C:\Program Files\DIREWOLF\bin\direwolf.exe`
- **Start Menu:** DIREWOLF shortcuts
- **Desktop:** Optional shortcut
- **PATH:** Added to system PATH
- **Registry:** Proper Windows integration

## 🗑️ Uninstalling

```powershell
# Method 1: PowerShell
.\install_direwolf.ps1 -Uninstall

# Method 2: Windows Settings
Start → Settings → Apps → DIREWOLF → Uninstall

# Method 3: Start Menu
Start → DIREWOLF → Uninstall
```

## 📞 Help

For detailed instructions, see:
- `INSTALLER_GUIDE.md` - Complete guide
- `../INSTALLER_COMPLETE.md` - Overview
- `../DESKTOP_APP_QUICK_START.md` - Usage guide

## 🎉 Success!

After installation, run DIREWOLF:
```powershell
direwolf
```

Or from Start Menu: Start → DIREWOLF
