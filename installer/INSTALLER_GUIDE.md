# DIREWOLF Windows 11 Installer Guide

## 🎯 Three Installation Methods

I've created **three different installer options** for you. Choose the one that works best:

### Method 1: PowerShell Installer (EASIEST - RECOMMENDED)
✅ No additional software needed
✅ Works immediately
✅ Simple and reliable

### Method 2: NSIS Installer (PROFESSIONAL)
✅ Creates standard .exe installer
✅ Professional installation wizard
✅ Requires NSIS software

### Method 3: WiX Installer (ENTERPRISE)
✅ Creates .msi installer
✅ Enterprise-grade
✅ Requires WiX Toolset

---

## 🚀 Method 1: PowerShell Installer (RECOMMENDED)

This is the **easiest and fastest** method. No additional software required!

### Step 1: Build DIREWOLF

```powershell
cd n:\CPPfiles\DRLHSS
.\build_desktop_simple.bat
```

### Step 2: Run PowerShell Installer

```powershell
# Open PowerShell as Administrator
cd n:\CPPfiles\DRLHSS\installer

# Run installer
.\install_direwolf.ps1
```

### What It Does:
- ✅ Installs to `C:\Program Files\DIREWOLF`
- ✅ Creates Start Menu shortcuts
- ✅ Creates Desktop shortcut (optional)
- ✅ Adds to system PATH
- ✅ Creates registry entries
- ✅ Adds to Add/Remove Programs

### To Uninstall:
```powershell
.\install_direwolf.ps1 -Uninstall
```

### Silent Installation:
```powershell
.\install_direwolf.ps1 -Silent
```

---

## 📦 Method 2: NSIS Installer

Creates a professional Windows installer (.exe file).

### Prerequisites:

1. **Install NSIS:**
   - Download from: https://nsis.sourceforge.io/Download
   - Install to default location
   - Add to PATH or note installation directory

### Step 1: Build DIREWOLF

```powershell
cd n:\CPPfiles\DRLHSS
.\build_desktop_simple.bat
```

### Step 2: Build Installer

```powershell
cd n:\CPPfiles\DRLHSS\installer
.\build_installer.bat
```

### Step 3: Run Installer

```powershell
# Right-click and "Run as Administrator"
.\DIREWOLF_Setup_v1.0.0.exe
```

### Features:
- ✅ Professional installation wizard
- ✅ Component selection (Core, Desktop Shortcut, Service, Auto-Start)
- ✅ Custom installation directory
- ✅ Start Menu integration
- ✅ Proper uninstaller
- ✅ Add/Remove Programs integration

---

## 🏢 Method 3: WiX Installer

Creates an enterprise-grade MSI installer.

### Prerequisites:

1. **Install WiX Toolset:**
   - Download from: https://wixtoolset.org/releases/
   - Install WiX Toolset v3.11 or later
   - Add to PATH

### Step 1: Build DIREWOLF

```powershell
cd n:\CPPfiles\DRLHSS
.\build_desktop_simple.bat
```

### Step 2: Create License RTF

```powershell
cd n:\CPPfiles\DRLHSS\installer
# Create license.rtf from LICENSE file
# (Can use WordPad to convert)
```

### Step 3: Build MSI

```powershell
# Compile WiX source
candle direwolf.wxs

# Link to create MSI
light -ext WixUIExtension direwolf.wixobj -out DIREWOLF_Setup.msi
```

### Step 4: Install

```powershell
# Double-click DIREWOLF_Setup.msi
# Or command line:
msiexec /i DIREWOLF_Setup.msi
```

### Features:
- ✅ Enterprise-grade MSI installer
- ✅ Group Policy deployment support
- ✅ Feature selection
- ✅ Windows Service installation
- ✅ Proper upgrade/uninstall support

---

## 📋 Comparison

| Feature | PowerShell | NSIS | WiX |
|---------|-----------|------|-----|
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Setup Required** | None | NSIS | WiX Toolset |
| **Professional Look** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Enterprise Features** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **File Size** | N/A | Small | Medium |
| **Customization** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🎯 Quick Start (Recommended Path)

### For Personal Use:
```powershell
# Use PowerShell installer - it's the easiest!
cd n:\CPPfiles\DRLHSS\installer
.\install_direwolf.ps1
```

### For Distribution:
```powershell
# Use NSIS to create a professional installer
cd n:\CPPfiles\DRLHSS\installer
.\build_installer.bat
# Distribute: DIREWOLF_Setup_v1.0.0.exe
```

### For Enterprise:
```powershell
# Use WiX to create MSI for Group Policy deployment
cd n:\CPPfiles\DRLHSS\installer
candle direwolf.wxs
light -ext WixUIExtension direwolf.wixobj -out DIREWOLF_Setup.msi
```

---

## 📁 Installation Locations

All methods install to:
- **Program Files:** `C:\Program Files\DIREWOLF`
- **Executable:** `C:\Program Files\DIREWOLF\bin\direwolf.exe`
- **Config:** `C:\Program Files\DIREWOLF\config`
- **Logs:** `C:\Program Files\DIREWOLF\logs`
- **Data:** `C:\Program Files\DIREWOLF\data`

---

## 🔧 Post-Installation

### Setup Admin Account:
```powershell
direwolf --setup-admin
```

### Run DIREWOLF:
```powershell
# From Start Menu
Start → DIREWOLF

# From Desktop
Double-click DIREWOLF shortcut

# From Command Line
direwolf
```

### Check Installation:
```powershell
# Verify executable
direwolf --help

# Check version
Get-ItemProperty "HKLM:\Software\DIREWOLF"
```

---

## 🗑️ Uninstallation

### PowerShell Method:
```powershell
cd n:\CPPfiles\DRLHSS\installer
.\install_direwolf.ps1 -Uninstall
```

### NSIS/WiX Method:
```powershell
# Use Add/Remove Programs
Start → Settings → Apps → DIREWOLF → Uninstall

# Or run uninstaller directly
"C:\Program Files\DIREWOLF\Uninstall.exe"
```

---

## 🎨 Customization

### Change Installation Directory:

**PowerShell:**
Edit `install_direwolf.ps1`:
```powershell
$InstallDir = "D:\MyApps\DIREWOLF"  # Change this line
```

**NSIS:**
Edit `direwolf_installer.nsi`:
```nsis
InstallDir "D:\MyApps\DIREWOLF"  # Change this line
```

**WiX:**
User can choose during installation.

### Add Custom Components:

Edit the respective installer script to include:
- Additional executables
- Configuration files
- Models and data files
- Documentation

---

## 🐛 Troubleshooting

### PowerShell Execution Policy:
```powershell
# If script won't run
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### NSIS Not Found:
```powershell
# Add NSIS to PATH
$env:Path += ";C:\Program Files (x86)\NSIS"
```

### WiX Not Found:
```powershell
# Add WiX to PATH
$env:Path += ";C:\Program Files (x86)\WiX Toolset v3.11\bin"
```

### Admin Rights Required:
```powershell
# Always run as Administrator
# Right-click PowerShell → "Run as Administrator"
```

---

## ✅ Verification

After installation, verify:

```powershell
# Check installation directory
Test-Path "C:\Program Files\DIREWOLF\bin\direwolf.exe"

# Check registry
Get-ItemProperty "HKLM:\Software\DIREWOLF"

# Check PATH
$env:Path -split ';' | Select-String "DIREWOLF"

# Check Start Menu
Test-Path "$env:ProgramData\Microsoft\Windows\Start Menu\Programs\DIREWOLF"

# Run DIREWOLF
direwolf --help
```

---

## 🎉 Success!

You now have three professional installer options for DIREWOLF!

**Recommended for you:** Start with the **PowerShell installer** - it's the easiest and works immediately without any additional software.

```powershell
cd n:\CPPfiles\DRLHSS\installer
.\install_direwolf.ps1
```

Enjoy your DIREWOLF Security System! 🐺🛡️
