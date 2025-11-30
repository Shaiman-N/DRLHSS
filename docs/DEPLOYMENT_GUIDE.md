# DRLHSS Deployment Guide

## Overview

This guide covers deploying the DRLHSS (Deep Reinforcement Learning Hybrid Security System) with NIDPS integration and cross-platform sandboxes.

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|------------|
| CPU | 4 cores, 2.0 GHz |
| RAM | 8 GB |
| Disk | 20 GB free space |
| OS | Linux (Ubuntu 20.04+), Windows 10+, macOS 11+ |

### Recommended Requirements

| Component | Requirement |
|-----------|------------|
| CPU | 8+ cores, 3.0 GHz |
| RAM | 16 GB |
| Disk | 50 GB SSD |
| Network | 1 Gbps |

## Platform-Specific Setup

### Linux (Ubuntu/Debian)

#### 1. Install Dependencies

```bash
# Update package list
sudo apt-get update

# Install build tools
sudo apt-get install -y build-essential cmake git

# Install required libraries
sudo apt-get install -y \
    libsqlite3-dev \
    libssl-dev \
    libseccomp-dev \
    libpcap-dev \
    libpthread-stubs0-dev

# Install ONNX Runtime
cd external
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
mv onnxruntime-linux-x64-1.16.0 onnxruntime
```

#### 2. Build Project

```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Install (optional)
sudo make install
```

#### 3. Setup Permissions

```bash
# For sandbox functionality (requires root or capabilities)
sudo setcap cap_sys_admin,cap_net_admin+ep ./integrated_system_example

# Or run as root (not recommended for production)
sudo ./integrated_system_example
```

#### 4. Configure cgroups v2

```bash
# Check if cgroups v2 is enabled
mount | grep cgroup2

# If not enabled, add to kernel parameters
sudo nano /etc/default/grub
# Add: systemd.unified_cgroup_hierarchy=1

# Update grub and reboot
sudo update-grub
sudo reboot
```

### Windows

#### 1. Install Dependencies

**Visual Studio 2019 or later:**
- Download from https://visualstudio.microsoft.com/
- Install "Desktop development with C++" workload

**CMake:**
```powershell
# Using Chocolatey
choco install cmake

# Or download from https://cmake.org/download/
```

**ONNX Runtime:**
```powershell
# Download and extract
Invoke-WebRequest -Uri "https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-win-x64-1.16.0.zip" -OutFile "onnxruntime.zip"
Expand-Archive -Path "onnxruntime.zip" -DestinationPath "external\"
Rename-Item "external\onnxruntime-win-x64-1.16.0" "onnxruntime"
```

**SQLite3:**
```powershell
# Download from https://www.sqlite.org/download.html
# Extract to external/sqlite3/
```

**OpenSSL:**
```powershell
# Using Chocolatey
choco install openssl

# Or download from https://slproweb.com/products/Win32OpenSSL.html
```

#### 2. Build Project

```powershell
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -G "Visual Studio 16 2019" -A x64

# Build
cmake --build . --config Release

# Or open DRLHSS.sln in Visual Studio and build
```

#### 3. Run as Administrator

```powershell
# Right-click PowerShell and "Run as Administrator"
cd build\Release
.\integrated_system_example.exe
```

### macOS

#### 1. Install Dependencies

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake sqlite3 openssl libpcap

# Install Xcode Command Line Tools
xcode-select --install

# Download ONNX Runtime
cd external
curl -L -O https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-osx-x86_64-1.16.0.tgz
tar -xzf onnxruntime-osx-x86_64-1.16.0.tgz
mv onnxruntime-osx-x86_64-1.16.0 onnxruntime
```

#### 2. Build Project

```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl

# Build
make -j$(sysctl -n hw.ncpu)
```

#### 3. Disable SIP (for testing only)

```bash
# Reboot into Recovery Mode (Cmd+R during boot)
# Open Terminal and run:
csrutil disable
# Reboot normally

# After testing, re-enable SIP:
csrutil enable
```

## Configuration

### Database Configuration

Create `config/database.conf`:

```ini
[database]
path = /var/lib/drlhss/drlhss.db
backup_path = /var/lib/drlhss/backups/
vacuum_interval = 86400  # 24 hours in seconds
max_size_mb = 10240      # 10 GB
```

### DRL Model Configuration

Create `config/drl_model.conf`:

```ini
[model]
path = models/onnx/mtl_model.onnx
feature_dim = 16
inference_threads = 4
batch_size = 32

[learning]
replay_buffer_size = 100000
pattern_learning_enabled = true
pattern_learning_interval = 3600  # 1 hour
```

### Sandbox Configuration

Create `config/sandbox.conf`:

```ini
[sandbox]
memory_limit_mb = 512
cpu_limit_percent = 50
timeout_seconds = 30
allow_network = true
read_only_filesystem = false

[linux]
base_image_path = /var/lib/drlhss/base_images/ubuntu
use_seccomp = true
use_namespaces = true

[windows]
use_appcontainer = true
use_job_objects = true

[macos]
verify_code_signature = false  # Set to true in production
use_quarantine = true
```

### NIDPS Configuration

Create `config/nidps.conf`:

```ini
[nidps]
network_interface = eth0
capture_filter = ""
malware_threshold = 0.7

[sandboxes]
positive_sandbox_enabled = true
negative_sandbox_enabled = true
sandbox_analysis_threshold = 0.7
```

## Running the System

### Integrated System

```bash
# Linux/macOS
./build/integrated_system_example

# Windows
.\build\Release\integrated_system_example.exe
```

### DRL Integration Only

```bash
# Linux/macOS
./build/drl_integration_example

# Windows
.\build\Release\drl_integration_example.exe
```

### Platform-Specific Sandbox Tests

```bash
# Linux
./build/test_linux_sandbox

# Windows
.\build\Release\test_windows_sandbox.exe

# macOS
./build/test_macos_sandbox
```

## Monitoring

### Log Files

```bash
# Application logs
tail -f /var/log/drlhss/application.log

# Sandbox logs
tail -f /var/log/drlhss/sandbox.log

# Detection logs
tail -f /var/log/drlhss/detection.log
```

### Statistics

```bash
# Query database for statistics
sqlite3 /var/lib/drlhss/drlhss.db

# Get detection count
SELECT COUNT(*) FROM telemetry;

# Get malicious detections
SELECT COUNT(*) FROM attack_patterns;

# Get recent detections
SELECT * FROM telemetry ORDER BY timestamp DESC LIMIT 10;
```

### Performance Monitoring

```bash
# CPU and memory usage
top -p $(pgrep integrated_system)

# Network usage
iftop -i eth0

# Disk I/O
iotop -p $(pgrep integrated_system)
```

## Production Deployment

### Systemd Service (Linux)

Create `/etc/systemd/system/drlhss.service`:

```ini
[Unit]
Description=DRLHSS Integrated Detection System
After=network.target

[Service]
Type=simple
User=drlhss
Group=drlhss
WorkingDirectory=/opt/drlhss
ExecStart=/opt/drlhss/bin/integrated_system_example
Restart=always
RestartSec=10

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/drlhss /var/log/drlhss

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable drlhss
sudo systemctl start drlhss
sudo systemctl status drlhss
```

### Windows Service

Use NSSM (Non-Sucking Service Manager):

```powershell
# Download NSSM
Invoke-WebRequest -Uri "https://nssm.cc/release/nssm-2.24.zip" -OutFile "nssm.zip"
Expand-Archive -Path "nssm.zip" -DestinationPath "C:\Tools\"

# Install service
C:\Tools\nssm-2.24\win64\nssm.exe install DRLHSS "C:\Program Files\DRLHSS\integrated_system_example.exe"

# Configure service
C:\Tools\nssm-2.24\win64\nssm.exe set DRLHSS AppDirectory "C:\Program Files\DRLHSS"
C:\Tools\nssm-2.24\win64\nssm.exe set DRLHSS AppStdout "C:\ProgramData\DRLHSS\logs\stdout.log"
C:\Tools\nssm-2.24\win64\nssm.exe set DRLHSS AppStderr "C:\ProgramData\DRLHSS\logs\stderr.log"

# Start service
net start DRLHSS
```

### macOS LaunchDaemon

Create `/Library/LaunchDaemons/com.drlhss.detection.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.drlhss.detection</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/integrated_system_example</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/var/log/drlhss/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/drlhss/stderr.log</string>
</dict>
</plist>
```

Load and start:

```bash
sudo launchctl load /Library/LaunchDaemons/com.drlhss.detection.plist
sudo launchctl start com.drlhss.detection
```

## Backup and Recovery

### Database Backup

```bash
# Manual backup
sqlite3 /var/lib/drlhss/drlhss.db ".backup /var/lib/drlhss/backups/drlhss_$(date +%Y%m%d).db"

# Automated backup (cron)
0 2 * * * sqlite3 /var/lib/drlhss/drlhss.db ".backup /var/lib/drlhss/backups/drlhss_$(date +\%Y\%m\%d).db"
```

### Model Backup

```bash
# Backup models
cp -r models/ /var/lib/drlhss/backups/models_$(date +%Y%m%d)/
```

### Configuration Backup

```bash
# Backup configuration
tar -czf /var/lib/drlhss/backups/config_$(date +%Y%m%d).tar.gz config/
```

## Troubleshooting

### Common Issues

#### 1. Sandbox Initialization Fails

**Linux:**
```bash
# Check cgroups
mount | grep cgroup2

# Check seccomp
grep SECCOMP /boot/config-$(uname -r)

# Check capabilities
getcap ./integrated_system_example
```

**Windows:**
```powershell
# Check if running as Administrator
[Security.Principal.WindowsIdentity]::GetCurrent().Groups -contains 'S-1-5-32-544'

# Check AppContainer support
Get-AppxPackage
```

**macOS:**
```bash
# Check SIP status
csrutil status

# Check sandbox-exec
which sandbox-exec
```

#### 2. ONNX Model Loading Fails

```bash
# Check model file
ls -lh models/onnx/mtl_model.onnx

# Check ONNX Runtime
ldd ./integrated_system_example | grep onnx  # Linux
otool -L ./integrated_system_example | grep onnx  # macOS
```

#### 3. Database Errors

```bash
# Check permissions
ls -l /var/lib/drlhss/drlhss.db

# Check disk space
df -h /var/lib/drlhss/

# Repair database
sqlite3 /var/lib/drlhss/drlhss.db "PRAGMA integrity_check;"
```

#### 4. High Resource Usage

```bash
# Check sandbox limits
cat /sys/fs/cgroup/sandbox_*/memory.max  # Linux

# Reduce limits in config
memory_limit_mb = 256
cpu_limit_percent = 25
```

## Security Hardening

### Linux

```bash
# Enable SELinux
sudo setenforce 1

# Configure firewall
sudo ufw allow 22/tcp
sudo ufw enable

# Restrict file permissions
sudo chmod 700 /var/lib/drlhss
sudo chown -R drlhss:drlhss /var/lib/drlhss
```

### Windows

```powershell
# Enable Windows Defender
Set-MpPreference -DisableRealtimeMonitoring $false

# Configure firewall
New-NetFirewallRule -DisplayName "DRLHSS" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8080
```

### macOS

```bash
# Enable firewall
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on

# Restrict permissions
sudo chmod 700 /usr/local/var/drlhss
sudo chown -R _drlhss:_drlhss /usr/local/var/drlhss
```

## Performance Tuning

### Database Optimization

```sql
-- Vacuum database
VACUUM;

-- Analyze tables
ANALYZE;

-- Create indices
CREATE INDEX idx_telemetry_timestamp ON telemetry(timestamp);
CREATE INDEX idx_telemetry_hash ON telemetry(artifact_hash);
CREATE INDEX idx_patterns_type ON attack_patterns(attack_type);
```

### Model Optimization

```bash
# Use optimized ONNX Runtime
export ORT_TENSORRT_ENGINE_CACHE_ENABLE=1
export ORT_TENSORRT_ENGINE_CACHE_PATH=/var/cache/drlhss/tensorrt
```

### Sandbox Optimization

```ini
# Reduce sandbox overhead
[sandbox]
memory_limit_mb = 256
cpu_limit_percent = 25
timeout_seconds = 15
```

## Scaling

### Horizontal Scaling

```bash
# Run multiple instances
./integrated_system_example --instance-id 1 --port 8081 &
./integrated_system_example --instance-id 2 --port 8082 &
./integrated_system_example --instance-id 3 --port 8083 &

# Load balance with nginx
upstream drlhss {
    server localhost:8081;
    server localhost:8082;
    server localhost:8083;
}
```

### Vertical Scaling

```ini
# Increase resources
[sandbox]
memory_limit_mb = 1024
cpu_limit_percent = 75

[model]
inference_threads = 8
batch_size = 64
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-org/drlhss/issues
- Documentation: https://docs.drlhss.org
- Email: support@drlhss.org

