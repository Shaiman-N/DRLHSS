# Cross-Platform Sandbox Architecture

## Overview

The DRLHSS system implements a unified sandbox interface with platform-specific implementations for Linux, Windows, and macOS. This document describes the architecture, design decisions, and implementation details.

## Design Principles

1. **Platform Abstraction**: Common interface across all platforms
2. **Security First**: Maximum isolation with minimal privileges
3. **Resource Control**: Strict CPU, memory, and time limits
4. **Behavioral Monitoring**: Comprehensive activity tracking
5. **Clean Separation**: No cross-contamination between sandboxes

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│              SandboxInterface (Abstract)                │
│  + initialize(config)                                   │
│  + execute(file_path, args)                            │
│  + analyzePacket(packet_data)                          │
│  + cleanup()                                            │
│  + reset()                                              │
│  + isReady()                                            │
│  + getSandboxId()                                       │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │   SandboxFactory      │
         │  + createSandbox()    │
         │  + detectPlatform()   │
         └───────────┬───────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │              │
      ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│  Linux   │  │ Windows  │  │  macOS   │
│ Sandbox  │  │ Sandbox  │  │ Sandbox  │
└──────────┘  └──────────┘  └──────────┘
```

## Platform Implementations

### Linux Sandbox

#### Isolation Mechanisms

**1. Namespaces**
```cpp
int flags = CLONE_NEWPID |   // Process ID isolation
            CLONE_NEWNET |   // Network isolation
            CLONE_NEWNS |    // Mount point isolation
            CLONE_NEWUTS |   // Hostname isolation
            CLONE_NEWIPC;    // IPC isolation

unshare(flags);
```

**2. Overlay Filesystem**
```
/tmp/sandbox_<id>/
├── upper/      # Writable layer (changes)
├── work/       # OverlayFS working directory
└── merged/     # Combined view (lower + upper)
```

**3. cgroups v2**
```
/sys/fs/cgroup/sandbox_<id>/
├── cpu.max         # CPU quota
├── memory.max      # Memory limit
└── pids.max        # Process limit
```

**4. seccomp Filtering**
```cpp
// Block dangerous syscalls
seccomp_rule_add(ctx, SCMP_ACT_KILL, SCMP_SYS(ptrace), 0);
seccomp_rule_add(ctx, SCMP_ACT_KILL, SCMP_SYS(reboot), 0);
seccomp_rule_add(ctx, SCMP_ACT_KILL, SCMP_SYS(kexec_load), 0);
```

#### Behavioral Monitoring

- **File System**: Monitor `/upper` directory for modifications
- **Network**: Check `/proc/net/tcp` for connections
- **Processes**: Use `ps --ppid` to detect child processes
- **System Calls**: Track via seccomp audit logs

#### Privilege Dropping

```cpp
setuid(65534);  // nobody user
setgid(65534);  // nobody group
```

### Windows Sandbox

#### Isolation Mechanisms

**1. Job Objects**
```cpp
HANDLE job = CreateJobObject(NULL, L"sandbox_job");

JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits;
limits.BasicLimitInformation.LimitFlags = 
    JOB_OBJECT_LIMIT_PROCESS_MEMORY |
    JOB_OBJECT_LIMIT_PROCESS_TIME |
    JOB_OBJECT_LIMIT_ACTIVE_PROCESS |
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

SetInformationJobObject(job, JobObjectExtendedLimitInformation, 
                       &limits, sizeof(limits));
```

**2. AppContainer**
```cpp
CreateAppContainerProfile(
    L"sandbox_container",
    L"Sandbox AppContainer",
    L"Isolated execution environment",
    NULL, 0, &sid
);
```

**3. Virtual Filesystem**
```
%TEMP%\sandbox_<id>\
├── packet.bin
├── analysis_tool.exe
└── results.txt
```

**4. Registry Virtualization**
```
HKEY_CURRENT_USER\Software\Sandbox\<id>\
```

#### Behavioral Monitoring

- **File System**: `FindFirstFile` / `FindNextFile` for file enumeration
- **Registry**: `RegQueryInfoKey` for registry modifications
- **Network**: `GetExtendedTcpTable` for connection tracking
- **Processes**: `CreateToolhelp32Snapshot` for process enumeration

#### Security Features

- **Integrity Levels**: Set process to Low integrity
- **Token Restrictions**: Limited token capabilities
- **Network Filtering**: Windows Filtering Platform (WFP) integration

### macOS Sandbox

#### Isolation Mechanisms

**1. Sandbox Profile (SBPL)**
```scheme
(version 1)
(deny default)
(allow process-exec process-fork)
(allow file-read* file-write* (subpath "/tmp/sandbox_<id>"))
(deny network*)
(deny system-socket mach-lookup ipc-posix-shm)
```

**2. Sandbox Directory**
```
/tmp/sandbox_<id>/
├── packet.bin
├── analysis_tool
└── results.txt
```

**3. File Quarantine**
```bash
xattr -w com.apple.quarantine "0001;$(date +%s);Sandbox;" file.bin
```

**4. Resource Limits**
```cpp
struct rlimit limit;
limit.rlim_cur = memory_mb * 1024 * 1024;
limit.rlim_max = memory_mb * 1024 * 1024;
setrlimit(RLIMIT_AS, &limit);
```

#### Behavioral Monitoring

- **File System**: `find` command for file enumeration
- **Network**: `lsof -p <pid> -i` for network connections
- **Processes**: `ps -o pid= --ppid <pid>` for child processes
- **Code Signing**: `codesign --verify` for signature validation

#### Security Features

- **TCC Restrictions**: Transparency, Consent, and Control
- **Code Signing**: Verify executable signatures
- **Entitlements**: Check application entitlements
- **Gatekeeper**: macOS security assessment

## Common Interface

### SandboxConfig

```cpp
struct SandboxConfig {
    std::string sandbox_id;          // Unique identifier
    uint64_t memory_limit_mb;        // Memory limit (MB)
    uint32_t cpu_limit_percent;      // CPU limit (0-100)
    uint32_t timeout_seconds;        // Execution timeout
    bool allow_network;              // Network access
    bool read_only_filesystem;       // Read-only FS
    std::string base_image_path;     // Base image (Linux)
};
```

### SandboxResult

```cpp
struct SandboxResult {
    bool success;                           // Execution success
    int exit_code;                          // Process exit code
    std::chrono::milliseconds execution_time;
    
    // Behavioral indicators
    bool file_system_modified;
    bool registry_modified;                 // Windows only
    bool network_activity_detected;
    bool process_created;
    bool memory_injection_detected;
    bool suspicious_api_calls;
    
    // Detailed information
    std::vector<std::string> accessed_files;
    std::vector<std::string> network_connections;
    std::vector<std::string> api_calls;
    
    int threat_score;                       // 0-100
};
```

## Threat Scoring Algorithm

```cpp
int score = 0;

if (file_system_modified)        score += 20;
if (registry_modified)           score += 25;  // Windows
if (network_activity_detected)   score += 25;
if (process_created)             score += 20;
if (memory_injection_detected)   score += 30;
if (suspicious_api_calls)        score += 25;

threat_score = min(score, 100);
```

### Threat Levels

- **0-30**: Low threat (likely benign)
- **31-60**: Medium threat (suspicious)
- **61-80**: High threat (likely malicious)
- **81-100**: Critical threat (definitely malicious)

## Factory Pattern

### Platform Detection

```cpp
SandboxType SandboxFactory::detectPlatform() {
#ifdef _WIN32
    return SandboxType::WINDOWS;
#elif defined(__APPLE__)
    return SandboxType::MACOS;
#elif defined(__linux__)
    return SandboxType::LINUX;
#else
    return SandboxType::UNKNOWN;
#endif
}
```

### Sandbox Creation

```cpp
std::unique_ptr<SandboxInterface> 
SandboxFactory::createSandbox(SandboxType type) {
    switch (detectPlatform()) {
        case SandboxType::LINUX:
            return std::make_unique<LinuxSandbox>(type);
        case SandboxType::WINDOWS:
            return std::make_unique<WindowsSandbox>(type);
        case SandboxType::MACOS:
            return std::make_unique<MacOSSandbox>(type);
        default:
            return nullptr;
    }
}
```

## Execution Flow

### 1. Initialization

```
1. Generate unique sandbox ID
2. Create sandbox directories/resources
3. Setup isolation mechanisms
4. Configure resource limits
5. Initialize monitoring systems
```

### 2. Execution

```
1. Fork/CreateProcess
2. Apply isolation (namespaces/job/sandbox profile)
3. Drop privileges
4. Change to sandbox directory
5. Execute target
6. Monitor behavior
7. Wait for completion (with timeout)
```

### 3. Monitoring

```
1. Track file system changes
2. Monitor network activity
3. Detect process creation
4. Check for memory injection
5. Log API calls
6. Calculate threat score
```

### 4. Cleanup

```
1. Terminate any running processes
2. Unmount filesystems (Linux)
3. Delete temporary files
4. Remove cgroups (Linux)
5. Delete registry keys (Windows)
6. Remove quarantine attributes (macOS)
```

## Performance Characteristics

### Initialization Time

| Platform | Cold Start | Warm Start |
|----------|-----------|------------|
| Linux    | 100-200ms | 10-20ms    |
| Windows  | 200-400ms | 20-40ms    |
| macOS    | 150-300ms | 15-30ms    |

### Execution Overhead

| Platform | CPU Overhead | Memory Overhead |
|----------|-------------|-----------------|
| Linux    | 5-10%       | 50-100 MB       |
| Windows  | 10-15%      | 100-150 MB      |
| macOS    | 8-12%       | 75-125 MB       |

### Cleanup Time

| Platform | Cleanup Time |
|----------|-------------|
| Linux    | 50-100ms    |
| Windows  | 100-200ms   |
| macOS    | 75-150ms    |

## Security Considerations

### Attack Surface

1. **Sandbox Escape**: Multiple isolation layers prevent escape
2. **Resource Exhaustion**: Strict limits prevent DoS
3. **Information Leakage**: Isolated network and filesystem
4. **Privilege Escalation**: Dropped privileges, restricted syscalls

### Hardening Recommendations

**Linux:**
- Use SELinux or AppArmor for additional MAC
- Enable kernel hardening (KASLR, SMEP, SMAP)
- Use user namespaces for unprivileged operation
- Enable audit logging for syscalls

**Windows:**
- Enable HVCI (Hypervisor-protected Code Integrity)
- Use Windows Defender Application Control
- Enable Control Flow Guard (CFG)
- Use Credential Guard

**macOS:**
- Enable System Integrity Protection (SIP)
- Use notarization for distributed binaries
- Enable Gatekeeper
- Use hardened runtime

## Limitations

### Linux
- Requires root or CAP_SYS_ADMIN for namespaces
- cgroups v2 required for full functionality
- OverlayFS may not work on all filesystems

### Windows
- Requires Windows 8+ for AppContainer
- Administrator privileges needed for Job Objects
- Some features require Windows 10+

### macOS
- Sandbox profiles are undocumented
- Code signing may interfere with testing
- TCC permissions require user interaction

## Future Enhancements

1. **Hardware Virtualization**: KVM, Hyper-V, Hypervisor.framework
2. **Container Integration**: Docker, containerd integration
3. **GPU Isolation**: GPU resource limits and isolation
4. **Network Simulation**: Simulated network environments
5. **Snapshot/Restore**: Fast sandbox state management

## References

- [Linux Namespaces](https://man7.org/linux/man-pages/man7/namespaces.7.html)
- [cgroups v2](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html)
- [seccomp](https://www.kernel.org/doc/html/latest/userspace-api/seccomp_filter.html)
- [Windows Job Objects](https://docs.microsoft.com/en-us/windows/win32/procthread/job-objects)
- [AppContainer](https://docs.microsoft.com/en-us/windows/win32/secauthz/appcontainer-isolation)
- [macOS Sandbox](https://developer.apple.com/library/archive/documentation/Security/Conceptual/AppSandboxDesignGuide/)

