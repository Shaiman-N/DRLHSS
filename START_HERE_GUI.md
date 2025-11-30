# 🚀 DIREWOLF GUI Installation - START HERE

## What You Need to Know

You currently have the **CLI (command-line) version** of DIREWOLF built. To get the **GUI (graphical interface)** version, follow these simple steps.

## Quick Start (3 Steps)

### Step 1: Install Qt6 (if not already installed)
Qt6 is required for the graphical interface.

1. Download Qt installer: https://www.qt.io/download-qt-installer
2. Run the installer
3. Select **Qt 6.5.0** or later
4. Choose **MSVC 2019 64-bit** component
5. Complete installation

### Step 2: Run the Installer
Double-click this file:
```
INSTALL_GUI.bat
```

Or run in PowerShell:
```powershell
cd N:\CPPfiles\DRLHSS
.\install_complete_gui.ps1
```

### Step 3: Launch DIREWOLF
After installation, find DIREWOLF in your:
- **Start Menu** (search for "DIREWOLF")
- **Desktop** (shortcut will be created)
- Or run: `%LOCALAPPDATA%\DIREWOLF\direwolf_gui.exe`

## What the GUI Includes

✅ **Dashboard** - Real-time threat monitoring  
✅ **Network Visualization** - See network activity  
✅ **System Status** - Monitor system health  
✅ **Settings Panel** - Configure DIREWOLF  
✅ **XAI Chat** - Talk to DIREWOLF assistant  
✅ **Permission Manager** - Control system access  
✅ **Scan Controls** - Manual and scheduled scans  

## Troubleshooting

### "Qt not found" error
- Make sure Qt6 is installed
- Check that MSVC 2019 64-bit component was selected during Qt installation

### "Visual Studio not found" error
- Install Visual Studio 2022 Community Edition (free)
- Make sure "Desktop development with C++" workload is installed

### GUI doesn't start
- Run from command line to see errors:
  ```
  %LOCALAPPDATA%\DIREWOLF\direwolf_gui.exe
  ```
- Check that all files were copied to `%LOCALAPPDATA%\DIREWOLF`

## Alternative: Use CLI Version

If you prefer the command-line version or can't install Qt, you can use:
```
N:\CPPfiles\DRLHSS\build\Release\direwolf.exe
```

## Need Help?

Check these files:
- `COMPLETE_GUI_SETUP.md` - Detailed setup guide
- `GUI_QUICK_START.md` - GUI usage guide
- `DESKTOP_APP_QUICK_START.md` - Desktop app guide

## Installation Location

After installation:
- **Executable**: `%LOCALAPPDATA%\DIREWOLF\direwolf_gui.exe`
- **QML Files**: `%LOCALAPPDATA%\DIREWOLF\qml\`
- **Config**: `%LOCALAPPDATA%\DIREWOLF\config\`
- **Python XAI**: `%LOCALAPPDATA%\DIREWOLF\python\`

---

**Ready to install?** Run `INSTALL_GUI.bat` now!
