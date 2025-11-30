# DIREWOLF Demo Guide

## Quick Start

### Option 1: Easy Launch (Recommended)
```batch
RUN_DEMO.bat
```
Then select which demo you want to run.

### Option 2: Direct Commands

**Complete System Demo:**
```bash
python demo_direwolf_simulation.py
```

**Live Traffic Monitor (60 seconds):**
```bash
python demo_live_traffic_monitor.py
```

**Live Traffic Monitor (Custom duration and rate):**
```bash
python demo_live_traffic_monitor.py [duration] [rate]
```

## Demo Options

### 1. Complete System Demo
**File:** `demo_direwolf_simulation.py`

Shows all DIREWOLF components:
- System startup and initialization
- Real-time dashboard
- Malware detection scan
- Network intrusion detection
- DRL-based threat response
- Sandbox analysis
- XAI assistant interaction

**Duration:** ~2-3 minutes  
**Best for:** Full system overview, presentations

### 2. Live Traffic Monitor
**File:** `demo_live_traffic_monitor.py`

Real-time network packet analysis:
- Continuous stream of network packets
- Mix of normal and suspicious traffic
- Live threat detection
- DRL decision making
- Real-time statistics

**Duration:** Configurable (default 60 seconds)  
**Best for:** Live demonstrations, showing real-time capabilities

## Command Line Options

### Live Traffic Monitor

```bash
python demo_live_traffic_monitor.py [duration] [rate]
```

**Parameters:**
- `duration` - How long to run in seconds (0 = infinite)
- `rate` - Packets per second (default: 2.0)

**Examples:**

```bash
# Run for 60 seconds at 2 packets/second (default)
python demo_live_traffic_monitor.py

# Run for 30 seconds
python demo_live_traffic_monitor.py 30

# Run for 2 minutes at 3 packets/second
python demo_live_traffic_monitor.py 120 3.0

# Run continuously until Ctrl+C
python demo_live_traffic_monitor.py 0 2.0

# Fast demo - 5 packets/second for 30 seconds
python demo_live_traffic_monitor.py 30 5.0
```

## What You'll See

### Complete System Demo

```
============================================================
               DIREWOLF Security System v1.0      
============================================================

ℹ Initializing core components...
✓ DRL Orchestrator initialized
✓ Antivirus Engine initialized
✓ Malware Detection Service initialized
...
```

Shows:
- ✓ System initialization
- ✓ Dashboard with statistics
- ✓ File scanning with threat detection
- ✓ Network monitoring
- ✓ DRL agent decisions
- ✓ Sandbox execution
- ✓ XAI explanations

### Live Traffic Monitor

```
============================================================
        DIREWOLF LIVE NETWORK TRAFFIC MONITOR
        Real-Time Threat Detection & DRL-Based Response
============================================================

[15:26:12.345] ✓ CLEAN | 192.168.1.105   | HTTP   | Port    80 |  512B
  Activity: GET /index.html
  Status: Normal traffic (Confidence: 97.3%)
  Action: ALLOWED

[15:26:13.123] ⚠ THREAT | 203.0.113.42    | TCP    | Port     0 |  128B
  Activity: SYN flood detected
  Detection: DDoS Attack
  Threat Level: CRITICAL
  DRL Confidence: 96.8%
  DRL Decision: IMMEDIATE_BLOCK
   ACTION: BLOCKED 
```

Shows:
- ⚠ Real-time packet stream
- ⚠ Threat detection
- ⚠ DRL confidence scores
- ⚠ Automated responses
- ⚠ Live statistics

## Demo Tips

### For Presentations

1. **Start with Complete System Demo**
   - Shows all capabilities
   - Good overview for audience

2. **Follow with Live Traffic Monitor**
   - More engaging
   - Shows real-time capabilities
   - Can run while talking

### Recommended Settings

**Short demo (5 minutes):**
```bash
python demo_live_traffic_monitor.py 60 3.0
```

**Medium demo (10 minutes):**
```bash
python demo_live_traffic_monitor.py 120 2.0
```

**Long demo/Background:**
```bash
python demo_live_traffic_monitor.py 0 1.5
```

### Making It More Impressive

**Fast-paced action:**
```bash
python demo_live_traffic_monitor.py 60 5.0
```
- More packets per second
- More threats detected
- More dramatic

**Slow and detailed:**
```bash
python demo_live_traffic_monitor.py 120 1.0
```
- Easier to follow
- Can explain each packet
- Better for technical audiences

## Stopping the Demo

- **Live Traffic Monitor:** Press `Ctrl+C`
- **Complete System Demo:** Runs automatically to completion

## Output Features

### Color Coding

- 🟢 **Green** - Clean traffic, allowed
- 🔴 **Red** - Threats detected
- 🟡 **Yellow** - Warnings, flagged items
- 🔵 **Blue/Cyan** - Information, system messages

### Statistics Bar

```
[Stats] Total: 45 | Clean: 32 | Threats: 13 | Blocked: 12
```

Updates in real-time during Live Traffic Monitor.

### Threat Levels

- **CRITICAL** - Immediate danger (DDoS, C&C communication)
- **HIGH** - Serious threats (SQL injection, port scans)
- **MEDIUM** - Moderate threats (XSS attempts)
- **LOW** - Minor suspicious activity

### DRL Decisions

- **IMMEDIATE_BLOCK** - High confidence (>95%), block immediately
- **BLOCK_AND_LOG** - Medium-high confidence (85-95%), block and log
- **MONITOR** - Lower confidence (<85%), monitor closely

## Troubleshooting

**Colors not showing:**
- Windows: Should work in PowerShell and modern CMD
- If issues, colors will be replaced with plain text

**Python not found:**
```bash
# Make sure Python is installed
python --version

# Or try:
py demo_live_traffic_monitor.py
```

**Script won't run:**
```bash
# Make sure you're in the right directory
cd N:\CPPfiles\DRLHSS

# Then run
python demo_live_traffic_monitor.py
```

## Summary

You now have two powerful demos:

1. **demo_direwolf_simulation.py** - Complete system walkthrough
2. **demo_live_traffic_monitor.py** - Real-time traffic analysis

Both show DIREWOLF's capabilities and are perfect for demonstrations!

**Quick Start:** Just run `RUN_DEMO.bat` and choose your demo!
