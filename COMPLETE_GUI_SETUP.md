# Complete DIREWOLF GUI Setup Guide

## Current Situation
You've been running the CLI version of DIREWOLF. The GUI version requires Qt6 to be installed.

## Option 1: Quick Install (Recommended)
Run the automated installer that will handle everything:

```powershell
cd N:\CPPfiles\DRLHSS
.\install_complete_gui.ps1
```

This script will:
1. Check for Qt6 installation
2. Guide you through Qt installation if needed
3. Build the GUI version
4. Install to %LOCALAPPDATA%\DIREWOLF
5. Create Start Menu and Desktop shortcuts

## Option 2: Manual Installation

### Step 1: Install Qt6
1. Download Qt Online Installer from: https://www.qt.io/download-qt-installer
2. Run the installer
3. Select Qt 6.5.0 or later
4. Choose MSVC 2019 64-bit component
5. Install to default location (C:\Qt)

### Step 2: Build GUI
```batch
cd N:\CPPfiles\DRLHSS
build_and_install_gui.bat
```

### Step 3: Launch
- Find "DIREWOLF" in your Start Menu
- Or use the Desktop shortcut
- Or run: %LOCALAPPDATA%\DIREWOLF\direwolf_gui.exe

## What You'll Get

The GUI includes:
- Dashboard with real-time threat monitoring
- Network visualization
- System status overview
- Settings panel
- XAI chat interface (DIREWOLF assistant)
- Permission management
- Scan controls

## Troubleshooting

### "Qt not found" error
- Install Qt6 from qt.io
- Make sure MSVC 2019 64-bit component is selected
- Add Qt bin directory to PATH

### "QML module not found" error
- Make sure qml folder was copied to installation directory
- Check that Qt Quick modules are installed

### GUI doesn't start
- Check if all DLLs are in %LOCALAPPDATA%\DIREWOLF
- Run from command line to see error messages:
  ```
  %LOCALAPPDATA%\DIREWOLF\direwolf_gui.exe
  ```

## Current Installation Location
After installation, DIREWOLF will be at:
- Executable: %LOCALAPPDATA%\DIREWOLF\direwolf_gui.exe
- QML files: %LOCALAPPDATA%\DIREWOLF\qml\
- Config: %LOCALAPPDATA%\DIREWOLF\config\
- Python XAI: %LOCALAPPDATA%\DIREWOLF\python\

## Next Steps
Once installed, you can:
1. Launch DIREWOLF from Start Menu
2. Configure your security preferences
3. Start real-time monitoring
4. Chat with DIREWOLF XAI assistant
