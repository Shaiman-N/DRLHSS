# DIREWOLF GUI Dashboard - Quick Start Guide

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Qt
Download and install Qt 6.5 or later from [qt.io](https://www.qt.io/download)

**Windows**: Install to `C:\Qt\6.5.3\msvc2019_64`
**Linux**: `sudo apt install qt6-base-dev qt6-declarative-dev`
**macOS**: `brew install qt@6`

### Step 2: Configure Build Script

**Windows** - Edit `build_gui.bat`:
```batch
set QT_PATH=C:\Qt\6.5.3\msvc2019_64
```

**Linux/macOS** - Edit `build_gui.sh`:
```bash
export QT_PATH="/usr/lib/qt6"
```

### Step 3: Build

**Windows**:
```batch
build_gui.bat
```

**Linux/macOS**:
```bash
chmod +x build_gui.sh
./build_gui.sh
```

### Step 4: Run

**Windows**:
```batch
run_gui.bat
```

**Linux/macOS**:
```bash
./build_gui/bin/direwolf_gui
```

## 🎯 What You'll See

### Main Dashboard
- Security status overview
- System performance metrics
- Quick action buttons
- Recent activity log

### Navigation Menu
- 🏠 Dashboard - Main overview
- 🛡️ Security - Threat management
- 🎤 Voice Assistant - Voice interaction
- 📊 System Monitor - Resource usage
- ⚙️ Settings - Configuration

## 🎤 Try Voice Commands

1. Click **Voice Assistant** in sidebar
2. Click the microphone icon or "Start Listening"
3. Say: "Scan for threats"
4. Watch the response appear

**Quick Commands**:
- "Scan for threats"
- "Check security status"
- "Show performance"
- "Find duplicate files"

## 🛡️ Run a Security Scan

1. Go to **Security** page
2. Click "Run Full Scan" button
3. Watch scan progress
4. View results in scan history

## 📊 Monitor System

1. Go to **System Monitor** page
2. View real-time metrics:
   - CPU usage with graph
   - Memory breakdown
   - Disk capacity
   - Top processes

## ⚙️ Configure Settings

1. Go to **Settings** page
2. Adjust preferences:
   - Enable/disable auto-start
   - Change theme (Dark/Light)
   - Configure voice settings
   - Set performance limits

## 🎨 UI Features

### Dark Theme
Modern dark interface with blue accents

### Real-time Updates
Metrics refresh every 2 seconds

### Smooth Animations
Pulsing microphone, smooth transitions

### Responsive Layout
Adapts to window size

## 🔧 Troubleshooting

### "Qt not found"
```bash
# Set Qt path
export CMAKE_PREFIX_PATH=/path/to/qt6
```

### "Build failed"
```bash
# Clean and rebuild
rm -rf build_gui
./build_gui.sh
```

### "Application won't start"
```bash
# Check Qt libraries are in PATH
# Windows: Add Qt bin to PATH
# Linux: sudo ldconfig
```

## 📱 Interface Overview

```
┌─────────────────────────────────────────────┐
│  DIREWOLF XAI                               │
├──────────┬──────────────────────────────────┤
│          │                                  │
│ 🏠 Dash  │     Main Dashboard               │
│ 🛡️ Sec   │     - Security Status            │
│ 🎤 Voice │     - Performance Metrics        │
│ 📊 Sys   │     - Quick Actions              │
│ ⚙️ Set   │     - Recent Activity            │
│          │                                  │
│ ● Online │                                  │
└──────────┴──────────────────────────────────┘
```

## 🎯 Key Features

### Security Dashboard
- Real-time threat detection
- Scan management
- Quarantine viewer
- Threat intelligence

### Voice Assistant
- Natural language commands
- Voice feedback
- Conversation history
- Quick command buttons

### System Monitor
- CPU/Memory/Disk usage
- Process list
- Historical charts
- Multi-drive support

### Settings
- General preferences
- Security configuration
- Voice customization
- Performance tuning

## 📚 Next Steps

1. **Explore Interface**: Navigate through all pages
2. **Try Voice Commands**: Test voice interaction
3. **Run Scans**: Test security features
4. **Customize Settings**: Adjust preferences
5. **Monitor System**: Watch real-time metrics

## 🔗 Integration

To connect with DRLHSS backend:

```cpp
// In GUIBackend.cpp
#include "XAI/XAIDataAggregator.hpp"
#include "Detection/UnifiedDetectionCoordinator.hpp"

// Replace mock data with real services
auto aggregator = std::make_shared<XAIDataAggregator>();
auto detector = std::make_shared<UnifiedDetectionCoordinator>();
```

## 💡 Tips

- **Keyboard Navigation**: Tab through controls
- **Window Resize**: Interface adapts automatically
- **Theme Toggle**: Change in Settings
- **Voice Speed**: Adjust in Voice settings
- **Performance**: Limit CPU/Memory in Settings

## 🆘 Support

### Documentation
- `PHASE4_GUI_COMPLETE.md` - Full documentation
- `DIREWOLF_XAI_COMPLETE_GUIDE.md` - System guide

### Build Issues
1. Check Qt installation
2. Verify CMake version (3.16+)
3. Check compiler (C++17 support)
4. Review build logs

### Runtime Issues
1. Check Qt libraries in PATH
2. Verify QML files in resources
3. Check backend initialization
4. Review console output

## ✅ Success Checklist

- [ ] Qt installed
- [ ] Build script configured
- [ ] Project built successfully
- [ ] Application launches
- [ ] All pages accessible
- [ ] Voice commands work
- [ ] Settings save correctly
- [ ] Real-time updates working

## 🎉 You're Ready!

The DIREWOLF GUI Dashboard is now running. Explore the interface, try voice commands, and monitor your system security in real-time!

---

**Need Help?** Check `PHASE4_GUI_COMPLETE.md` for detailed documentation.
