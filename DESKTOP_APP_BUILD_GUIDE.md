# DIREWOLF Desktop Application - Build & Installation Guide

## Quick Start for Windows Desktop Application

### Prerequisites

Before building DIREWOLF as a desktop application, you need:

1. **Visual Studio 2022** (already installed)
2. **CMake** (already installed)
3. **SQLite3** - Download from: https://www.sqlite.org/download.html
4. **OpenSSL** - Download from: https://slproweb.com/products/Win32OpenSSL.html

### Step 1: Install Dependencies

#### Install SQLite3:
```powershell
# Download SQLite3 precompiled binaries for Windows
# Extract to: C:\sqlite3
# Add C:\sqlite3 to your PATH
```

#### Install OpenSSL:
```powershell
# Download Win64 OpenSSL v3.x.x
# Install to: C:\Program Files\OpenSSL-Win64
# Installer will add to PATH automatically
```

### Step 2: Build DIREWOLF

```powershell
# Navigate to DRLHSS directory
cd n:\CPPfiles\DRLHSS

# Create and enter build directory
mkdir build
cd build

# Configure with CMake
cmake .. -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_INSTALL_PREFIX="C:\DIREWOLF" `
  -DSQLite3_INCLUDE_DIR="C:\sqlite3" `
  -DSQLite3_LIBRARY="C:\sqlite3\sqlite3.lib"

# Build the project
cmake --build . --config Release --parallel

# Install to C:\DIREWOLF (requires admin)
cmake --install . --config Release
```

### Step 3: Development Mode Workflow

After initial build, for quick updates during development:

```powershell
# Make your code changes in n:\CPPfiles\DRLHSS\src

# Rebuild
cd n:\CPPfiles\DRLHSS\build
cmake --build . --config Release

# Run directly from build directory
.\Release\direwolf.exe

# Or reinstall to C:\DIREWOLF
cmake --install . --config Release
C:\DIREWOLF\bin\direwolf.exe
```

### Step 4: Admin Authentication Setup

On first run, DIREWOLF will prompt you to:
1. Create admin username
2. Set secure password (min 8 chars, uppercase, lowercase, number, special char)
3. Record voice biometric samples (3 recordings)

### Step 5: Running DIREWOLF

```powershell
# Option 1: From installation directory
C:\DIREWOLF\bin\direwolf.exe

# Option 2: From build directory (development)
n:\CPPfiles\DRLHSS\build\Release\direwolf.exe

# Option 3: Setup admin account only
direwolf.exe --setup-admin

# Option 4: Skip auth (development only)
direwolf.exe --no-auth
```

## Troubleshooting

### SQLite3 Not Found
```powershell
# Set environment variable
$env:SQLite3_ROOT = "C:\sqlite3"
```

### OpenSSL Not Found
```powershell
# Set environment variable
$env:OPENSSL_ROOT_DIR = "C:\Program Files\OpenSSL-Win64"
```

### Build Errors
```powershell
# Clean build directory
cd n:\CPPfiles\DRLHSS\build
Remove-Item * -Recurse -Force

# Reconfigure
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
```

## Next Steps

1. Build the application following steps above
2. Install to C:\DIREWOLF
3. Setup admin authentication
4. Start developing with quick rebuild workflow

For updates during development, just rebuild and run from build directory - no need to reinstall each time!
