# DIREWOLF Troubleshooting Guide

## Complete Problem-Solution Reference

**Version**: 1.0.0  
**Last Updated**: 2024

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Service & Startup Problems](#service--startup-problems)
3. [Performance Issues](#performance-issues)
4. [Network & Connectivity](#network--connectivity)
5. [Voice & Audio Problems](#voice--audio-problems)
6. [UI & Display Issues](#ui--display-issues)
7. [Detection & Security](#detection--security)
8. [Database & Storage](#database--storage)
9. [Update Problems](#update-problems)
10. [Advanced Diagnostics](#advanced-diagnostics)

---

## Installation Issues

### Problem: Installer Fails to Run

**Symptoms**:
- Installer crashes immediately
- "Access Denied" error
- Installer hangs

**Solutions**:

1. **Run as Administrator/Root**:
   ```powershell
   # Windows
   Right-click installer → "Run as Administrator"
   
   # Linux
   sudo dpkg -i direwolf.deb
   
   # macOS
   sudo installer -pkg direwolf.pkg -target /
   ```

2. **Disable Antivirus Temporarily**:
   - Windows Defender: Settings → Virus & threat protection → Manage settings → Real-time protection OFF
   - Third-party: Consult vendor documentation
   - **Remember to re-enable after installation!**

3. **Check Disk Space**:
   ```bash
   # Windows
   Get-PSDrive C | Select-Object Used,Free
   
   # Linux/macOS
   df -h /
   ```
   Need at least 10GB free.

4. **Verify Download Integrity**:
   ```bash
   # Check SHA256 hash
   sha256sum direwolf-installer.exe  # Linux
   Get-FileHash direwolf-installer.exe  # Windows PowerShell
   shasum -a 256 direwolf.dmg  # macOS
   ```
   Compare with official hash at https://direwolf.ai/checksums

**Still Not Working?**:
- Check installation logs:
  - Windows: `%TEMP%\direwolf-install.log`
  - Linux: `/tmp/direwolf-install.log`
  - macOS: `~/Library/Logs/direwolf-install.log`
- Contact support with log file

---

### Problem: Missing Dependencies

**Symptoms**:
- "Library not found" errors
- "Cannot load module" errors
- Installer complains about missing packages

**Solutions**:

**Windows**:
```powershell
# Install Visual C++ Redistributables
choco install vcredist-all -y

# Install .NET Framework
choco install dotnetfx -y
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt update
sudo apt install -y libqt6core6 libqt6gui6 libqt6widgets6 \
                    libonnxruntime1 libsqlite3-0 python3 python3-pip
```

**Linux (RHEL/Fedora)**:
```bash
sudo dnf install -y qt6-qtbase onnxruntime sqlite python3 python3-pip
```

**macOS**:
```bash
brew install qt@6 onnxruntime sqlite python@3.9
```

---

## Service & Startup Problems

### Problem: DIREWOLF Service Won't Start

**Symptoms**:
- Service fails to start
- "Service not responding" error
- System tray icon doesn't appear

**Solutions**:

1. **Check Service Status**:
   ```bash
   # Windows
   sc query DIREWOLF
   Get-Service DIREWOLF
   
   # Linux
   systemctl status direwolf
   
   # macOS
   launchctl list | grep direwolf
   ```

2. **Check Service Logs**:
   ```bash
   # Windows
   Get-EventLog -LogName Application -Source DIREWOLF -Newest 20
   
   # Linux
   journalctl -u direwolf -n 50 --no-pager
   
   # macOS
   log show --predicate 'process == "direwolf"' --last 10m
   ```

3. **Verify Permissions**:
   ```bash
   # Linux/macOS
   sudo chown -R direwolf:direwolf /var/lib/direwolf
   sudo chmod 755 /usr/local/bin/direwolf
   
   # Check SELinux (if applicable)
   sudo setenforce 0  # Temporary
   sudo setsebool -P direwolf_can_network_connect 1
   ```

4. **Check Port Conflicts**:
   ```bash
   # Linux/macOS
   sudo netstat -tulpn | grep -E '8080|8443'
   
   # Windows
   netstat -ano | findstr "8080 8443"
   ```
   If ports are in use, either:
   - Stop conflicting service
   - Change DIREWOLF ports in config

5. **Restart Service**:
   ```bash
   # Windows
   sc stop DIREWOLF
   sc start DIREWOLF
   
   # Linux
   sudo systemctl restart direwolf
   
   # macOS
   launchctl unload ~/Library/LaunchAgents/com.direwolf.agent.plist
   launchctl load ~/Library/LaunchAgents/com.direwolf.agent.plist
   ```

---

### Problem: DIREWOLF Crashes on Startup

**Symptoms**:
- Application starts then immediately closes
- Crash dialog appears
- No UI appears

**Solutions**:

1. **Check Crash Logs**:
   ```bash
   # Windows
   %APPDATA%\DIREWOLF\logs\crash.log
   
   # Linux
   ~/.config/direwolf/logs/crash.log
   
   # macOS
   ~/Library/Application Support/DIREWOLF/logs/crash.log
   ```

2. **Reset Configuration**:
   ```bash
   # Backup current config
   cp config/direwolf.yaml config/direwolf.yaml.backup
   
   # Reset to defaults
   direwolf --reset-config
   ```

3. **Check Graphics Drivers**:
   - Update to latest GPU drivers
   - Try software rendering:
     ```bash
     direwolf --software-rendering
     ```

4. **Verify Database Integrity**:
   ```bash
   direwolf --check-db
   
   # If corrupted, restore from backup
   direwolf --restore-db backup/direwolf.db
   ```

---

## Performance Issues

### Problem: High CPU Usage

**Symptoms**:
- CPU usage >50%
- System slowdown
- Fan noise increase

**Solutions**:

1. **Check Active Scans**:
   ```bash
   direwolf --status
   ```
   If scan is running, wait for completion or:
   ```bash
   direwolf --pause-scan
   ```

2. **Reduce Visualization Quality**:
   - Settings → Performance → Visualization Quality → Low
   - Or disable 3D visualization temporarily

3. **Adjust Scan Priority**:
   - Settings → Performance → Scan Priority → Low
   - Reduces CPU usage during scans

4. **Check for Malware**:
   Ironically, high CPU could indicate actual malware:
   ```bash
   direwolf --full-scan
   ```

5. **Update DIREWOLF**:
   ```bash
   direwolf --check-updates
   direwolf --update
   ```
   Performance improvements in newer versions.

---

### Problem: High Memory Usage

**Symptoms**:
- RAM usage >2GB
- System swapping
- Out of memory errors

**Solutions**:

1. **Check Memory Usage**:
   ```bash
   direwolf --memory-stats
   ```

2. **Clear Cache**:
   ```bash
   direwolf --clear-cache
   ```

3. **Reduce Buffer Sizes**:
   Edit `config/direwolf.yaml`:
   ```yaml
   performance:
     replay_buffer_size: 10000  # Reduce from 100000
     cache_size_mb: 100  # Reduce from 500
   ```

4. **Disable Video Generation**:
   - Settings → Video → Auto-generate → OFF
   - Videos consume significant memory

5. **Restart Service**:
   ```bash
   direwolf --restart
   ```
   Clears memory leaks (if any).

---

### Problem: Slow Response Time

**Symptoms**:
- UI lag
- Delayed threat detection
- Slow command execution

**Solutions**:

1. **Check System Resources**:
   ```bash
   # Overall system status
   direwolf --system-health
   ```

2. **Optimize Database**:
   ```bash
   direwolf --optimize-db
   ```

3. **Reduce Monitoring Scope**:
   - Settings → Network → Monitored Subnets
   - Disable monitoring for non-critical networks

4. **Enable Performance Mode**:
   - Settings → Performance → Mode → Performance
   - Reduces accuracy slightly for speed

5. **Check Network Latency**:
   ```bash
   direwolf --network-test
   ```

---

## Network & Connectivity

### Problem: Network Devices Not Detected

**Symptoms**:
- Empty network visualization
- "No devices found" message
- Incomplete device list

**Solutions**:

1. **Check Network Permissions**:
   ```bash
   # Linux - add user to netdev group
   sudo usermod -aG netdev $USER
   # Logout and login again
   
   # macOS - grant network access
   # System Preferences → Security & Privacy → Privacy → Network
   ```

2. **Verify Network Interface**:
   ```bash
   # List interfaces
   ip link show  # Linux
   ipconfig /all  # Windows
   ifconfig  # macOS
   
   # Set correct interface in config
   direwolf --set-interface eth0
   ```

3. **Run Manual Discovery**:
   ```bash
   direwolf --discover-network
   ```

4. **Check Firewall**:
   ```bash
   # Linux - allow ICMP
   sudo iptables -A INPUT -p icmp --icmp-type echo-request -j ACCEPT
   
   # Windows - allow ICMP
   netsh advfirewall firewall add rule name="ICMP Allow" protocol=icmpv4:8,any dir=in action=allow
   ```

5. **Verify Subnet Configuration**:
   Edit `config/network.yaml`:
   ```yaml
   networks:
     - subnet: "192.168.1.0/24"  # Correct subnet
       monitor: true
   ```

---

### Problem: Cannot Connect to Web Interface

**Symptoms**:
- Browser shows "Connection refused"
- "Unable to connect" error
- Timeout errors

**Solutions**:

1. **Verify Service is Running**:
   ```bash
   direwolf --status
   ```

2. **Check Port Binding**:
   ```bash
   # Linux/macOS
   sudo netstat -tulpn | grep 8080
   
   # Windows
   netstat -ano | findstr 8080
   ```

3. **Try Different Port**:
   Edit `config/direwolf.yaml`:
   ```yaml
   web:
     port: 8081  # Change from 8080
   ```
   Then restart service.

4. **Check Firewall**:
   ```bash
   # Linux
   sudo ufw allow 8080/tcp
   
   # Windows
   netsh advfirewall firewall add rule name="DIREWOLF Web" dir=in action=allow protocol=TCP localport=8080
   ```

5. **Access via Localhost**:
   Try: `http://127.0.0.1:8080` instead of `http://localhost:8080`

---

## Voice & Audio Problems

### Problem: Voice Commands Not Recognized

**Symptoms**:
- Wake word doesn't trigger
- Commands not understood
- No response from Wolf

**Solutions**:

1. **Test Audio Devices**:
   ```bash
   direwolf --test-audio
   ```

2. **Check Microphone Permissions**:
   - **Windows**: Settings → Privacy → Microphone → Allow apps
   - **macOS**: System Preferences → Security & Privacy → Microphone
   - **Linux**: Check PulseAudio/ALSA permissions

3. **Verify Whisper Model**:
   ```bash
   python3 -c "import whisper; whisper.load_model('base')"
   ```
   If fails, reinstall:
   ```bash
   pip3 install --upgrade openai-whisper
   ```

4. **Adjust Wake Word Sensitivity**:
   - Settings → Voice → Wake Word Sensitivity → High

5. **Test with Simple Commands**:
   Start with: "Hey Wolf, hello"
   If works, gradually try more complex commands.

6. **Check Background Noise**:
   - Reduce ambient noise
   - Use better microphone
   - Adjust input volume

---

### Problem: No Voice Output (TTS)

**Symptoms**:
- Wolf doesn't speak
- Text appears but no audio
- Silent responses

**Solutions**:

1. **Check TTS Settings**:
   - Settings → Voice → TTS Enabled → ON
   - Settings → Voice → Volume → 80%

2. **Test TTS**:
   ```bash
   direwolf --test-tts
   ```

3. **Verify TTS Provider**:
   ```bash
   # For Azure TTS
   echo $AZURE_SPEECH_KEY
   
   # For Google TTS
   echo $GOOGLE_APPLICATION_CREDENTIALS
   
   # For local TTS
   pip3 list | grep pyttsx3
   ```

4. **Switch TTS Provider**:
   - Settings → Voice → TTS Provider → Local
   - Local TTS always works offline

5. **Check Audio Output**:
   - Verify correct output device selected
   - Check system volume
   - Test with other applications

---

## UI & Display Issues

### Problem: UI Not Displaying Correctly

**Symptoms**:
- Blank window
- Garbled graphics
- Missing elements

**Solutions**:

1. **Update Graphics Drivers**:
   - NVIDIA: https://www.nvidia.com/drivers
   - AMD: https://www.amd.com/support
   - Intel: https://www.intel.com/content/www/us/en/support/detect.html

2. **Try Software Rendering**:
   ```bash
   direwolf --software-rendering
   ```

3. **Reset UI Settings**:
   ```bash
   direwolf --reset-ui
   ```

4. **Check Display Scaling**:
   - Windows: Settings → Display → Scale → 100%
   - macOS: System Preferences → Displays → Resolution
   - Linux: Display settings → Scale

5. **Verify Qt Installation**:
   ```bash
   # Check Qt version
   qmake --version
   
   # Reinstall if needed
   sudo apt install --reinstall qt6-base-dev  # Linux
   brew reinstall qt@6  # macOS
   ```

---

### Problem: 3D Visualization Not Working

**Symptoms**:
- Black screen in network view
- "WebGL not supported" error
- Visualization crashes

**Solutions**:

1. **Check WebGL Support**:
   Visit: https://get.webgl.org/
   Should show spinning cube.

2. **Enable Hardware Acceleration**:
   - Chrome: chrome://settings → Advanced → System → Use hardware acceleration
   - Firefox: about:config → webgl.force-enabled → true

3. **Update Browser**:
   ```bash
   # Update to latest version
   # Chrome, Firefox, or Edge
   ```

4. **Try Different Browser**:
   - Chrome (recommended)
   - Firefox
   - Edge

5. **Reduce Visualization Complexity**:
   - Settings → Visualization → Quality → Low
   - Settings → Visualization → Max Nodes → 50

---

## Detection & Security

### Problem: False Positives

**Symptoms**:
- Legitimate files flagged as threats
- Normal traffic blocked
- Excessive alerts

**Solutions**:

1. **Add to Whitelist**:
   ```bash
   direwolf --whitelist-file /path/to/file
   direwolf --whitelist-ip 192.168.1.100
   ```

2. **Adjust Detection Sensitivity**:
   - Settings → Detection → Sensitivity → Medium
   - Reduces false positives, may miss some threats

3. **Provide Feedback**:
   When denying a permission request:
   - Select "This is a false positive"
   - Wolf learns and adjusts

4. **Check Detection Rules**:
   ```bash
   direwolf --list-rules
   direwolf --disable-rule RULE_ID
   ```

5. **Update Signatures**:
   ```bash
   direwolf --update-signatures
   ```

---

### Problem: Threats Not Detected

**Symptoms**:
- Known malware not flagged
- Suspicious activity ignored
- No alerts generated

**Solutions**:

1. **Update DIREWOLF**:
   ```bash
   direwolf --check-updates
   direwolf --update
   ```

2. **Update Threat Signatures**:
   ```bash
   direwolf --update-signatures --force
   ```

3. **Increase Detection Sensitivity**:
   - Settings → Detection → Sensitivity → High
   - May increase false positives

4. **Run Full Scan**:
   ```bash
   direwolf --full-scan
   ```

5. **Check Detection Logs**:
   ```bash
   tail -f ~/.config/direwolf/logs/detection.log
   ```

6. **Verify Models**:
   ```bash
   direwolf --verify-models
   ```
   Redownload if corrupted.

---

## Database & Storage

### Problem: Database Errors

**Symptoms**:
- "Database locked" errors
- "Corruption detected" messages
- Data not saving

**Solutions**:

1. **Check Database Integrity**:
   ```bash
   direwolf --check-db
   ```

2. **Unlock Database**:
   ```bash
   # Stop all DIREWOLF processes
   direwolf --stop-all
   
   # Remove lock file
   rm ~/.config/direwolf/direwolf.db-lock
   
   # Restart
   direwolf --start
   ```

3. **Repair Database**:
   ```bash
   direwolf --repair-db
   ```

4. **Restore from Backup**:
   ```bash
   direwolf --list-backups
   direwolf --restore-db backup_20240115.db
   ```

5. **Reinitialize Database** (Last Resort):
   ```bash
   # Backup first!
   cp ~/.config/direwolf/direwolf.db ~/direwolf.db.backup
   
   # Reinitialize
   direwolf --init-db
   ```

---

### Problem: Disk Space Full

**Symptoms**:
- "No space left on device" errors
- Cannot save data
- Logs not writing

**Solutions**:

1. **Check Disk Usage**:
   ```bash
   direwolf --disk-usage
   ```

2. **Clean Old Logs**:
   ```bash
   direwolf --clean-logs --older-than 30d
   ```

3. **Delete Old Videos**:
   ```bash
   direwolf --clean-videos --older-than 90d
   ```

4. **Optimize Database**:
   ```bash
   direwolf --vacuum-db
   ```

5. **Move Data Directory**:
   ```bash
   # Stop service
   direwolf --stop
   
   # Move data
   sudo mv /var/lib/direwolf /mnt/larger-disk/direwolf
   
   # Update config
   direwolf --set-data-dir /mnt/larger-disk/direwolf
   
   # Start service
   direwolf --start
   ```

---

## Update Problems

### Problem: Update Fails

**Symptoms**:
- Update download fails
- Installation errors
- Version doesn't change

**Solutions**:

1. **Check Internet Connection**:
   ```bash
   ping update.direwolf.ai
   ```

2. **Verify Update Server**:
   ```bash
   curl -I https://update.direwolf.ai/latest
   ```

3. **Manual Update**:
   ```bash
   # Download manually
   wget https://direwolf.ai/download/latest
   
   # Install
   sudo dpkg -i direwolf-latest.deb  # Linux
   # Or run installer on Windows/macOS
   ```

4. **Clear Update Cache**:
   ```bash
   direwolf --clear-update-cache
   ```

5. **Check Disk Space**:
   Need at least 2GB free for updates.

---

### Problem: Update Breaks System

**Symptoms**:
- DIREWOLF won't start after update
- Features missing
- Errors after update

**Solutions**:

1. **Rollback Update**:
   ```bash
   direwolf --rollback
   ```

2. **Restore from Backup**:
   ```bash
   direwolf --restore-backup pre-update
   ```

3. **Reinstall Previous Version**:
   ```bash
   # Download previous version
   wget https://direwolf.ai/download/v0.9.0
   
   # Install
   sudo dpkg -i direwolf-0.9.0.deb
   ```

4. **Report Bug**:
   ```bash
   direwolf --report-bug --include-logs
   ```

---

## Advanced Diagnostics

### Comprehensive System Check

```bash
direwolf --diagnose --full
```

Checks:
- Service status
- Configuration validity
- Database integrity
- Network connectivity
- Model files
- Dependencies
- Permissions
- Disk space
- Memory usage
- Log files

### Collect Debug Information

```bash
direwolf --collect-debug-info
```

Creates archive with:
- Configuration files
- Log files
- System information
- Error reports
- Performance metrics

Send to support@direwolf.ai

### Enable Debug Logging

Edit `config/direwolf.yaml`:
```yaml
logging:
  level: "DEBUG"
  console: true
  file: true
```

Restart service:
```bash
direwolf --restart
```

View logs:
```bash
tail -f ~/.config/direwolf/logs/direwolf.log
```

### Performance Profiling

```bash
direwolf --profile --duration 60s
```

Generates performance report showing:
- CPU usage by component
- Memory allocation
- I/O operations
- Network traffic
- Function call times

---

## Getting Help

### Before Contacting Support

1. Check this troubleshooting guide
2. Review FAQ: `docs/FAQ.md`
3. Search community forum
4. Check GitHub issues

### When Contacting Support

Include:
- DIREWOLF version: `direwolf --version`
- Operating system and version
- Problem description
- Steps to reproduce
- Error messages
- Log files: `direwolf --collect-debug-info`

### Support Channels

- **Community Forum**: https://community.direwolf.ai
- **Discord**: https://discord.gg/direwolf
- **Email**: support@direwolf.ai
- **Emergency**: enterprise@direwolf.ai (enterprise customers)

---

**DIREWOLF Troubleshooting Guide v1.0.0**  
*Your Intelligent Security Guardian*
