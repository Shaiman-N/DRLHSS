# DIREWOLF Installation Guide

## Complete Installation Instructions for All Platforms

**Version**: 1.0.0  
**Last Updated**: 2024

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Pre-Installation Checklist](#pre-installation-checklist)
3. [Windows Installation](#windows-installation)
4. [Linux Installation](#linux-installation)
5. [macOS Installation](#macos-installation)
6. [Post-Installation Setup](#post-installation-setup)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

**Hardware**:
- CPU: 4-core processor (Intel Core i5 or AMD Ryzen 5)
- RAM: 8 GB
- Storage: 10 GB free space
- GPU: OpenGL 3.3 compatible
- Network: 100 Mbps connection

**Software**:
- Operating System: Windows 10+, Ubuntu 20.04+, macOS 12+
- Python: 3.8 or higher
- CMake: 3.20 or higher
- C++ Compiler: GCC 9+, Clang 10+, or MSVC 2019+

### Recommended Requirements

**Hardware**:
- CPU: 8-core processor (Intel Core i7 or AMD Ryzen 7)
- RAM: 16 GB
- Storage: 20 GB free space (SSD recommended)
- GPU: Dedicated GPU with 2GB VRAM
- Network: 1 Gbps connection

**Software**:
- Latest OS updates installed
- CUDA 11.0+ (for GPU acceleration)
- Docker (for containerized deployment)

---

## Pre-Installation Checklist

Before installing DIREWOLF, ensure:

- [ ] System meets minimum requirements
- [ ] Administrator/root access available
- [ ] Antivirus temporarily disabled (will be re-enabled)
- [ ] Firewall configured to allow DIREWOLF
- [ ] Network connectivity verified
- [ ] Backup of important data completed
- [ ] Previous security software uninstalled

---

## Windows Installation

### Method 1: Installer (Recommended)

1. **Download Installer**
   ```powershell
   # Download from official website
   Invoke-WebRequest -Uri https://direwolf.ai/download/windows -OutFile direwolf-setup.exe
   ```

2. **Run Installer**
   - Right-click `direwolf-setup.exe`
   - Select "Run as Administrator"
   - Follow installation wizard

3. **Installation Options**
   - Installation Directory: `C:\Program Files\DIREWOLF`
   - Create Desktop Shortcut: ✓
   - Add to PATH: ✓
   - Install Service: ✓
   - Configure Firewall: ✓

4. **Complete Installation**
   - Click "Install"
   - Wait for installation (5-10 minutes)
   - Click "Finish"

### Method 2: Build from Source

1. **Install Dependencies**
   ```powershell
   # Install Chocolatey (if not installed)
   Set-ExecutionPolicy Bypass -Scope Process -Force
   iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
   
   # Install build tools
   choco install cmake git python visualstudio2019buildtools -y
   choco install qt6 onnxruntime -y
   ```

2. **Clone Repository**
   ```powershell
   git clone https://github.com/direwolf/DRLHSS.git
   cd DRLHSS
   ```

3. **Build Project**
   ```powershell
   # Create build directory
   mkdir build
   cd build
   
   # Configure with CMake
   cmake .. -G "Visual Studio 16 2019" -A x64
   
   # Build
   cmake --build . --config Release
   ```

4. **Install**
   ```powershell
   cmake --install . --config Release
   ```

### Windows Service Configuration

```powershell
# Install as Windows Service
sc create DIREWOLF binPath= "C:\Program Files\DIREWOLF\direwolf.exe --service" start= auto

# Start service
sc start DIREWOLF

# Check status
sc query DIREWOLF
```

---

## Linux Installation

### Method 1: Package Manager (Ubuntu/Debian)

1. **Add Repository**
   ```bash
   # Add DIREWOLF repository
   curl -fsSL https://direwolf.ai/gpg | sudo gpg --dearmor -o /usr/share/keyrings/direwolf.gpg
   echo "deb [signed-by=/usr/share/keyrings/direwolf.gpg] https://apt.direwolf.ai stable main" | sudo tee /etc/apt/sources.list.d/direwolf.list
   
   # Update package list
   sudo apt update
   ```

2. **Install DIREWOLF**
   ```bash
   sudo apt install direwolf
   ```

3. **Enable Service**
   ```bash
   sudo systemctl enable direwolf
   sudo systemctl start direwolf
   ```

### Method 2: RPM Package (RHEL/CentOS/Fedora)

1. **Add Repository**
   ```bash
   # Add DIREWOLF repository
   sudo tee /etc/yum.repos.d/direwolf.repo <<EOF
   [direwolf]
   name=DIREWOLF Repository
   baseurl=https://yum.direwolf.ai/stable
   enabled=1
   gpgcheck=1
   gpgkey=https://direwolf.ai/gpg
   EOF
   ```

2. **Install DIREWOLF**
   ```bash
   sudo dnf install direwolf
   # or for older systems
   sudo yum install direwolf
   ```

3. **Enable Service**
   ```bash
   sudo systemctl enable direwolf
   sudo systemctl start direwolf
   ```

### Method 3: Build from Source

1. **Install Dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt install build-essential cmake git python3 python3-pip
   sudo apt install qt6-base-dev libonnxruntime-dev libsqlite3-dev
   
   # RHEL/CentOS/Fedora
   sudo dnf groupinstall "Development Tools"
   sudo dnf install cmake git python3 python3-pip
   sudo dnf install qt6-qtbase-devel onnxruntime-devel sqlite-devel
   ```

2. **Clone and Build**
   ```bash
   git clone https://github.com/direwolf/DRLHSS.git
   cd DRLHSS
   
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   sudo make install
   ```

3. **Install Systemd Service**
   ```bash
   sudo cp ../scripts/direwolf.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable direwolf
   sudo systemctl start direwolf
   ```

---

## macOS Installation

### Method 1: DMG Installer (Recommended)

1. **Download DMG**
   ```bash
   curl -O https://direwolf.ai/download/macos/direwolf.dmg
   ```

2. **Install Application**
   - Double-click `direwolf.dmg`
   - Drag DIREWOLF to Applications folder
   - Eject DMG

3. **First Launch**
   - Open Applications folder
   - Right-click DIREWOLF
   - Select "Open" (to bypass Gatekeeper)
   - Click "Open" in security dialog

### Method 2: Homebrew

1. **Install via Homebrew**
   ```bash
   # Add DIREWOLF tap
   brew tap direwolf/tap
   
   # Install DIREWOLF
   brew install direwolf
   ```

2. **Start Service**
   ```bash
   brew services start direwolf
   ```

### Method 3: Build from Source

1. **Install Dependencies**
   ```bash
   # Install Homebrew (if not installed)
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install build tools
   brew install cmake git python@3.9 qt@6 onnxruntime
   ```

2. **Clone and Build**
   ```bash
   git clone https://github.com/direwolf/DRLHSS.git
   cd DRLHSS
   
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(sysctl -n hw.ncpu)
   sudo make install
   ```

3. **Create Launch Agent**
   ```bash
   cp ../scripts/com.direwolf.agent.plist ~/Library/LaunchAgents/
   launchctl load ~/Library/LaunchAgents/com.direwolf.agent.plist
   ```

---

## Post-Installation Setup

### 1. Initial Configuration

**Launch DIREWOLF**:
- Windows: Start Menu → DIREWOLF
- Linux: `direwolf` or Applications menu
- macOS: Applications → DIREWOLF

**First-Time Wizard**:
1. Welcome screen
2. License agreement
3. User profile setup
4. Network configuration
5. Voice settings (optional)
6. Completion

### 2. Network Configuration

**Automatic Discovery**:
- DIREWOLF automatically scans your network
- Review detected devices
- Confirm or correct device types

**Manual Configuration**:
```yaml
# Edit config/network.yaml
networks:
  - name: "Primary Network"
    subnet: "192.168.1.0/24"
    gateway: "192.168.1.1"
    monitor: true
    
  - name: "Guest Network"
    subnet: "192.168.2.0/24"
    gateway: "192.168.2.1"
    monitor: false
```

### 3. Firewall Configuration

**Windows Firewall**:
```powershell
# Allow DIREWOLF through firewall
New-NetFirewallRule -DisplayName "DIREWOLF" -Direction Inbound -Program "C:\Program Files\DIREWOLF\direwolf.exe" -Action Allow
```

**Linux iptables**:
```bash
# Allow DIREWOLF ports
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8443 -j ACCEPT
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

**macOS Firewall**:
```bash
# Add DIREWOLF to firewall exceptions
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /Applications/DIREWOLF.app
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /Applications/DIREWOLF.app
```

### 4. Python Dependencies

```bash
# Install Python packages
pip3 install -r requirements.txt

# Or manually
pip3 install torch ollama whisper pyttsx3 fastapi uvicorn
```

### 5. LLM Setup (Ollama)

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull Llama 2 model
ollama pull llama2:7b

# Verify
ollama list
```

---

## Verification

### 1. Service Status

**Windows**:
```powershell
sc query DIREWOLF
```

**Linux/macOS**:
```bash
systemctl status direwolf
# or
brew services list | grep direwolf
```

### 2. Web Interface

Open browser and navigate to:
```
http://localhost:8080
```

You should see the DIREWOLF dashboard.

### 3. Command Line

```bash
# Check version
direwolf --version

# Check status
direwolf --status

# Run diagnostics
direwolf --diagnose
```

### 4. Voice Test

```bash
# Test voice interface
direwolf --test-voice
```

Expected output:
```
Testing microphone... OK
Testing speakers... OK
Testing wake word detection... OK
Testing TTS... OK

Voice interface ready!
```

---

## Troubleshooting

### Installation Fails

**Issue**: Installer crashes or fails to complete

**Solutions**:
1. Run as administrator/root
2. Disable antivirus temporarily
3. Check disk space
4. Verify system requirements
5. Check installation logs:
   - Windows: `%TEMP%\direwolf-install.log`
   - Linux: `/tmp/direwolf-install.log`
   - macOS: `~/Library/Logs/direwolf-install.log`

### Service Won't Start

**Issue**: DIREWOLF service fails to start

**Solutions**:
1. Check service logs:
   ```bash
   # Windows
   Get-EventLog -LogName Application -Source DIREWOLF -Newest 10
   
   # Linux
   journalctl -u direwolf -n 50
   
   # macOS
   log show --predicate 'process == "direwolf"' --last 10m
   ```

2. Verify permissions:
   ```bash
   # Linux/macOS
   sudo chown -R direwolf:direwolf /var/lib/direwolf
   sudo chmod 755 /usr/local/bin/direwolf
   ```

3. Check port conflicts:
   ```bash
   # Linux/macOS
   sudo netstat -tulpn | grep -E '8080|8443'
   
   # Windows
   netstat -ano | findstr "8080 8443"
   ```

### Dependencies Missing

**Issue**: Missing library errors

**Solutions**:
1. Reinstall dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt install --reinstall libqt6core6 libonnxruntime1
   
   # RHEL/Fedora
   sudo dnf reinstall qt6-qtbase onnxruntime
   
   # macOS
   brew reinstall qt@6 onnxruntime
   ```

2. Update library cache:
   ```bash
   # Linux
   sudo ldconfig
   ```

### Network Not Detected

**Issue**: DIREWOLF can't detect network devices

**Solutions**:
1. Check network permissions:
   ```bash
   # Linux - add user to netdev group
   sudo usermod -aG netdev $USER
   
   # Restart session
   ```

2. Verify network interface:
   ```bash
   # List interfaces
   ip link show  # Linux
   ipconfig      # Windows
   ifconfig      # macOS
   ```

3. Run network discovery manually:
   ```bash
   direwolf --discover-network
   ```

### Voice Not Working

**Issue**: Voice commands not recognized

**Solutions**:
1. Test audio devices:
   ```bash
   direwolf --test-audio
   ```

2. Check microphone permissions:
   - Windows: Settings → Privacy → Microphone
   - macOS: System Preferences → Security & Privacy → Microphone
   - Linux: Check PulseAudio/ALSA configuration

3. Verify Whisper model:
   ```bash
   python3 -c "import whisper; whisper.load_model('base')"
   ```

---

## Uninstallation

### Windows

```powershell
# Stop service
sc stop DIREWOLF

# Uninstall via Control Panel
# Or use installer
.\direwolf-setup.exe /uninstall

# Remove data (optional)
Remove-Item -Recurse -Force "$env:APPDATA\DIREWOLF"
```

### Linux

```bash
# Stop service
sudo systemctl stop direwolf
sudo systemctl disable direwolf

# Uninstall package
sudo apt remove direwolf  # Ubuntu/Debian
sudo dnf remove direwolf  # RHEL/Fedora

# Remove data (optional)
sudo rm -rf /var/lib/direwolf
sudo rm -rf ~/.config/direwolf
```

### macOS

```bash
# Stop service
brew services stop direwolf

# Uninstall
brew uninstall direwolf

# Or remove app
rm -rf /Applications/DIREWOLF.app

# Remove data (optional)
rm -rf ~/Library/Application\ Support/DIREWOLF
```

---

## Next Steps

After successful installation:

1. **Read Quick Start Guide**: `docs/QUICK_START_GUIDE.md`
2. **Configure Settings**: Open Settings → User Profile
3. **Test Voice Commands**: "Hey Wolf, show me network status"
4. **Review Dashboard**: Familiarize yourself with the interface
5. **Generate First Briefing**: "Hey Wolf, generate daily briefing"

---

## Support

If you encounter issues not covered in this guide:

- **Documentation**: https://docs.direwolf.ai
- **Community Forum**: https://community.direwolf.ai
- **Email Support**: support@direwolf.ai
- **Emergency**: enterprise@direwolf.ai

---

**DIREWOLF Installation Guide v1.0.0**  
*Your Intelligent Security Guardian*
