# DIREWOLF Desktop Application - Quick Start Guide

## ✅ Build Complete!

Your DIREWOLF desktop application has been successfully built!

### 📍 Executable Location

**Build Directory:** `n:\CPPfiles\DRLHSS\build_desktop\Release\direwolf.exe`

## 🚀 Quick Start Options

### Option 1: Run from Build Directory (Development Mode - RECOMMENDED)

This is the fastest way to develop and test:

```powershell
# Run directly
n:\CPPfiles\DRLHSS\build_desktop\Release\direwolf.exe

# Or navigate and run
cd n:\CPPfiles\DRLHSS\build_desktop\Release
.\direwolf.exe
```

**Development Workflow:**
```powershell
# 1. Make code changes in: n:\CPPfiles\DRLHSS\src\desktop_main.cpp

# 2. Rebuild
cd n:\CPPfiles\DRLHSS\build_desktop
cmake --build . --config Release

# 3. Run updated version
.\Release\direwolf.exe
```

### Option 2: Install to C:\DIREWOLF (Production Mode)

For permanent installation with admin privileges:

```powershell
# Run as Administrator
cd n:\CPPfiles\DRLHSS\build_desktop
cmake --install . --config Release

# Then run from anywhere
C:\DIREWOLF\bin\direwolf.exe
```

## 🔐 Admin Authentication Setup

### First Time Setup

On first run, DIREWOLF will automatically prompt you to create an admin account:

1. **Username:** Enter your admin username (min 3 characters)
2. **Password:** Enter secure password (min 8 characters)
3. **Confirm Password:** Re-enter password

### Manual Setup

```powershell
# Setup admin account explicitly
direwolf.exe --setup-admin
```

### Skip Authentication (Development Only)

```powershell
# Skip auth for testing
direwolf.exe --no-auth
```

## 📋 Available Commands

Once DIREWOLF is running, you can use these commands:

- `status` - Show system status
- `scan` - Run security scan
- `update` - Check for updates
- `help` - Show available commands
- `exit` - Exit DIREWOLF

## 🔧 Command Line Options

```powershell
direwolf.exe [options]

Options:
  --setup-admin    Setup administrator account
  --no-auth        Skip authentication (development only)
  --help, -h       Show help message
```

## 🎯 Current Features

- ✅ Admin username/password authentication
- ✅ Command-line interface
- ✅ System status monitoring
- ✅ Security scanning
- ✅ Update checking
- ⏳ Voice biometric (coming soon)
- ⏳ GUI dashboard (coming soon)

## 🔄 Development Workflow

### Quick Rebuild After Code Changes

```powershell
# Navigate to build directory
cd n:\CPPfiles\DRLHSS\build_desktop

# Rebuild (fast - only recompiles changed files)
cmake --build . --config Release

# Run updated app
.\Release\direwolf.exe
```

### Clean Rebuild

```powershell
# Clean and rebuild everything
cd n:\CPPfiles\DRLHSS
rmdir /s /q build_desktop
.\build_desktop_simple.bat
```

## 📁 Project Structure

```
n:\CPPfiles\DRLHSS\
├── src\
│   └── desktop_main.cpp          # Main desktop application
├── build_desktop\                # Build output directory
│   └── Release\
│       └── direwolf.exe          # Executable
├── build_desktop_simple.bat      # Build script
└── DESKTOP_APP_QUICK_START.md    # This file
```

## 🎨 Next Steps

### Add Voice Biometric Authentication

1. Integrate Windows Speech API
2. Record voice samples during setup
3. Verify voice during authentication

### Add GUI Dashboard

1. Integrate Qt or Windows Forms
2. Create visual dashboard
3. Add system tray icon
4. Real-time monitoring display

### Add Auto-Update System

1. Check for updates on startup
2. Download and verify updates
3. Apply updates with admin privileges

## 🐛 Troubleshooting

### Build Errors

```powershell
# Clean build directory
cd n:\CPPfiles\DRLHSS
rmdir /s /q build_desktop

# Rebuild
.\build_desktop_simple.bat
```

### Can't Run - Missing DLLs

The application is statically linked, so this shouldn't happen. If it does:

```powershell
# Check dependencies
dumpbin /dependents direwolf.exe
```

### Admin Privileges Required

Some features require admin privileges. Right-click and "Run as Administrator":

```powershell
# Check if running as admin
net session
```

## 📞 Support

For issues or questions:
1. Check this guide
2. Review build output
3. Check `DESKTOP_APP_BUILD_GUIDE.md` for detailed setup

## 🎉 Success!

You now have a working DIREWOLF desktop application with admin authentication!

**Quick Test:**
```powershell
cd n:\CPPfiles\DRLHSS\build_desktop\Release
.\direwolf.exe --no-auth
```

This will launch DIREWOLF without authentication for quick testing.
