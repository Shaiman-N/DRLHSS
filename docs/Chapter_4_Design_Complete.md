# CHAPTER 4: DESIGN

This chapter presents the comprehensive design of the Deep Reinforcement Learning Hybrid Security System (DRLHSS), covering system architecture, specifications, design decisions, and technical approaches for all integrated subsystems.

## 4.1 Overall System Architecture

DRLHSS is a multi-layered adaptive security framework integrating network intrusion detection, antivirus protection, malware analysis, web application firewall capabilities, and intelligent threat response through deep reinforcement learning.

### 4.1.1 High-Level System Design

The system comprises seven primary subsystems operating in a coordinated manner:

**Detection Layers:**
- Network Intrusion Detection and Prevention System (NIDPS)
- Antivirus Detection System (AV)  
- Malware Detection System (MD)
- Web Application Firewall (WAF) - Template Design

**Intelligence & Support Layers:**
- Deep Reinforcement Learning Framework (DRL)
- Database Management System (DB)
- Explainable AI Interface (XAI)

### 4.1.2 System Architecture Diagram

```
┌────────────────────────────────────────────────────────────┐
│                  DRLHSS Security System                     │
├────────────────────────────────────────────────────────────┤
│  Input Streams: Network | Files | System Behavior          │
│         ↓              ↓              ↓                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐              │
│  │  NIDPS   │   │    AV    │   │    MD    │   [WAF]      │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘              │
│       └──────────────┼──────────────┘                      │
│                      ↓                                      │
│           ┌──────────────────┐                             │
│           │    Unified       │                             │
│           │   Coordinator    │                             │
│           └────────┬─────────┘                             │
│                    ↓                                        │
│       ┌────────────┼────────────┐                          │
│       ↓            ↓            ↓                          │
│   ┌───────┐   ┌────────┐   ┌────────┐                    │
│   │  DRL  │   │Sandbox │   │   DB   │                    │
│   └───────┘   └────────┘   └────────┘                    │
│                                                             │
│  Cross-Platform: Windows | Linux | macOS                   │
└────────────────────────────────────────────────────────────┘
```

### 4.1.3 Design Principles

**Modularity**: Independent operation with well-defined interfaces
**Scalability**: Horizontal scaling through distributed deployment  
**Adaptability**: Continuous learning from new threats via DRL
**Real-Time**: Low-latency detection (< 100ms)
**Cross-Platform**: Windows, Linux, macOS support
**Explainability**: Human-understandable threat analysis

### 4.1.4 Technology Stack

**Languages**: C++17 (core), Python (training/utilities)
**ML Framework**: ONNX Runtime for inference
**Database**: SQLite3 for persistence
**Build System**: CMake for cross-platform compilation
**Libraries**: libpcap (network), OpenSSL (crypto), platform-specific APIs



## 4.2 Input Stream Processing Design

### 4.2.1 Multi-Stream Architecture

The system processes three concurrent input streams:

**Network Stream**: Raw packets via libpcap → Protocol parsing → Flow aggregation
**File Stream**: File system events → Binary analysis → Feature extraction  
**Behavior Stream**: API calls → Registry changes → Process monitoring

### 4.2.2 Data Processing Pipeline

```
Raw Input → Feature Extraction → Normalization → Encoding → Detection
```

**Feature Dimensions:**
- Network: 41 statistical features per flow
- Files: 2381 PE features (EMBER-compatible)
- Behavior: 500 API call patterns

**Normalization Techniques:**
- Min-max scaling for numerical features
- One-hot encoding for categorical data
- Z-score normalization for statistical features

## 4.3 Detection Layer A: NIDPS Design

### 4.3.1 Functional Specifications

**Requirements:**
- Real-time packet capture and analysis
- Multi-protocol support (TCP, UDP, ICMP, HTTP, DNS)
- Attack classification (DoS, DDoS, Port Scan, SQL Injection, XSS)
- Latency < 10ms, Throughput 1000-5000 packets/sec
- False Positive Rate < 5%, True Positive Rate > 90%

### 4.3.2 NIDPS Architecture

```
Packet Capture (libpcap)
    ↓
Protocol Parser (TCP/UDP/ICMP/HTTP/DNS)
    ↓
Flow Aggregator (Session Tracking)
    ↓
Feature Extractor (41 Statistical Features)
    ↓
MTL Model Inference (ONNX)
    ↓
DRL Decision Engine
```

### 4.3.3 Multi-Task Learning Model Design

**Architecture:**
- Input: 41 flow features
- Shared layers: Dense(128) → ReLU → Dropout(0.3)
- Task 1: Binary classification (Benign/Malicious)
- Task 2: Attack type (7 classes)

**Attack Types**: Benign, DoS/DDoS, Port Scan, Brute Force, Web Attack, Infiltration, Botnet

### 4.3.4 Design Justification

**libpcap Selection**: Cross-platform, low-level access, proven performance
**Flow-based Analysis**: Reduces overhead while capturing session context
**MTL Approach**: Shared representations improve generalization with single model



## 4.4 Detection Layer B: Malware Detection System Design

### 4.4.1 Specifications

**Requirements:**
- Multi-stage detection pipeline
- Static and dynamic analysis
- Visual malware analysis (MalImg)
- Real-time monitoring
- Throughput: 100-500 files/minute

### 4.4.2 Multi-Stage Pipeline Architecture

```
File Input → Initial Detection → Positive Sandbox → Negative Sandbox → DRL → Action
     ↓              ↓                  ↓                  ↓           ↓        ↓
  Hash/Meta    ML Classify      Behavior Anal      FN Detect    Q-Values  Response
```

**Stages:**
1. **Initial Detection**: DCNN-based classification
2. **Positive Sandbox**: False positive reduction
3. **Negative Sandbox**: False negative detection  
4. **DRL Decision**: Intelligent action selection

### 4.4.3 Detection Models

**DCNN Classifier**: Deep CNN for binary classification
**MalImg CNN**: Visual malware analysis from binary visualization
**Feature Set**: PE headers, sections, imports, exports, entropy

### 4.4.4 Real-Time Monitoring Design

**Monitors:**
- File system events (create, modify, delete)
- Registry modifications
- Startup folder changes
- Network connections
- Process creation

**Design Choice**: Event-driven architecture for minimal overhead (< 5% CPU)

## 4.5 Detection Layer C: Antivirus System Design

### 4.5.1 Specifications

**Requirements:**
- Static PE analysis (2381 features)
- Dynamic behavior monitoring (500 API patterns)
- Hybrid ML prediction
- Scan time: 50-100ms per file
- False Positive Rate < 2%

### 4.5.2 Hybrid Detection Architecture

```
File Input
    ↓
    ├→ Static Analysis (PE Features) → Static Model → Score₁
    │
    └→ Dynamic Analysis (API Calls) → Dynamic Model → Score₂
                                            ↓
                            Hybrid Score = 0.6×Score₁ + 0.4×Score₂
                                            ↓
                                    DRL Decision Engine
```

### 4.5.3 Feature Extraction Design

**Static Features (2381 dimensions)**:
- Byte histogram (256)
- Entropy features (256)
- String features (104)
- PE header features (62)
- Section features (255)
- Import/Export features (1448)

**Dynamic Features (500 dimensions)**:
- API call patterns
- Network activity
- File operations
- Registry modifications
- Process behavior

### 4.5.4 Design Justification

**Hybrid Approach**: Combines speed of static analysis with accuracy of dynamic analysis
**EMBER Compatibility**: Industry-standard feature set for reproducibility
**Weighted Fusion**: 60/40 split optimized through validation experiments



## 4.6 Detection Layer D: Web Application Firewall (WAF) - Template Design

### 4.6.1 Specifications

**Requirements:**
- HTTP/HTTPS traffic inspection
- OWASP Top 30 attack detection
- Request/Response analysis
- Latency < 50ms per request

### 4.6.2 WAF Architecture Template

```
HTTP/HTTPS Traffic
    ↓
Request Parser
    ↓
    ├→ Header Analysis
    ├→ URL Analysis  
    ├→ Parameter Analysis
    └→ Body Analysis
         ↓
Pattern Matching Engine
         ↓
ML-Based Anomaly Detection
         ↓
DRL Decision Engine
```

### 4.6.3 OWASP Attack Coverage (Template)

**Injection Attacks (10)**:
1. SQL Injection
2. NoSQL Injection
3. LDAP Injection
4. XML Injection
5. Command Injection
6. Code Injection
7. XPath Injection
8. SSI Injection
9. Template Injection
10. Expression Language Injection

**Cross-Site Attacks (5)**:
11. Cross-Site Scripting (XSS) - Reflected
12. XSS - Stored
13. XSS - DOM-based
14. Cross-Site Request Forgery (CSRF)
15. Clickjacking

**Authentication & Session (5)**:
16. Broken Authentication
17. Session Fixation
18. Session Hijacking
19. Credential Stuffing
20. Brute Force

**Access Control (3)**:
21. Broken Access Control
22. Insecure Direct Object References
23. Path Traversal

**Data Exposure (4)**:
24. Sensitive Data Exposure
25. XML External Entities (XXE)
26. Server-Side Request Forgery (SSRF)
27. Information Disclosure

**Misconfiguration & Other (3)**:
28. Security Misconfiguration
29. Insecure Deserialization
30. Using Components with Known Vulnerabilities

### 4.6.4 Detection Approach

**Pattern-Based**: Regex patterns for known attack signatures
**ML-Based**: Anomaly detection for zero-day attacks
**Behavioral**: Request frequency and pattern analysis

## 4.7 Deep Reinforcement Learning Framework Design

### 4.7.1 Specifications

**Requirements:**
- Real-time inference (< 30ms)
- Continuous learning capability
- Experience replay for training
- Model hot-reload support

### 4.7.2 DRL Architecture

```
┌─────────────────────────────────────┐
│        DRL Orchestrator              │
├─────────────────────────────────────┤
│  ┌──────────┐  ┌──────────────┐    │
│  │  ONNX    │  │ Environment  │    │
│  │Inference │  │   Adapter    │    │
│  └──────────┘  └──────────────┘    │
│  ┌──────────┐  ┌──────────────┐    │
│  │  Replay  │  │   Database   │    │
│  │  Buffer  │  │   Manager    │    │
│  └──────────┘  └──────────────┘    │
└─────────────────────────────────────┘
```

### 4.7.3 DQN Model Design

**State Space (16 dimensions)**:
- System call count
- File I/O operations
- Network connections
- CPU/Memory usage
- Registry modifications
- Privilege escalation attempts
- Code injection indicators
- Derived behavioral features

**Action Space (4 actions)**:
- 0: Allow (benign)
- 1: Block (malicious)
- 2: Quarantine (suspicious)
- 3: DeepScan (sandbox analysis)

**Network Architecture**:
```
Input(16) → Dense(128) → ReLU → Dense(64) → ReLU → Dense(4) → Q-Values
```

**Training Parameters**:
- Learning Rate: 0.0001
- Discount Factor (γ): 0.99
- Epsilon Decay: 0.995
- Batch Size: 64
- Replay Buffer: 100,000 experiences

### 4.7.4 Reward Function Design

```
Reward = {
    +10  : Correct threat detection
    +5   : Correct benign classification
    -10  : False positive (benign marked malicious)
    -20  : False negative (malicious marked benign)
    +2   : Appropriate quarantine decision
}
```

### 4.7.5 Design Justification

**DQN Selection**: Proven performance in discrete action spaces
**State Representation**: Captures essential behavioral indicators
**Reward Structure**: Heavily penalizes false negatives (security priority)



## 4.8 Database Management System Design

### 4.8.1 Specifications

**Requirements:**
- Thread-safe operations
- Write latency < 10ms
- Support millions of records
- Automatic backup and optimization

### 4.8.2 Database Schema Design

**Telemetry Table**:
```sql
CREATE TABLE telemetry (
    id INTEGER PRIMARY KEY,
    sandbox_id TEXT,
    timestamp INTEGER,
    syscall_count INTEGER,
    file_operations INTEGER,
    network_connections INTEGER,
    cpu_usage REAL,
    memory_usage REAL,
    threat_indicators INTEGER,
    artifact_hash TEXT
);
```

**Experiences Table**:
```sql
CREATE TABLE experiences (
    id INTEGER PRIMARY KEY,
    episode_id TEXT,
    state_vector TEXT,
    action INTEGER,
    reward REAL,
    next_state_vector TEXT,
    done INTEGER,
    timestamp INTEGER
);
```

**Attack Patterns Table**:
```sql
CREATE TABLE attack_patterns (
    id INTEGER PRIMARY KEY,
    pattern_features TEXT,
    action_taken INTEGER,
    confidence_score REAL,
    attack_type TEXT,
    timestamp INTEGER
);
```

### 4.8.3 Design Decisions

**SQLite Selection**: Embedded, zero-configuration, cross-platform
**Indexing Strategy**: Composite indices on timestamp + artifact_hash
**Optimization**: Periodic VACUUM operations for performance

## 4.9 Explainable AI (XAI) System Design

### 4.9.1 Specifications

**Requirements:**
- Human-readable threat explanations
- Decision transparency
- Feature importance visualization
- Natural language generation

### 4.9.2 XAI Architecture

```
Detection Event
    ↓
┌─────────────────────────────┐
│   XAI Data Aggregator       │
│  (Collect detection data)   │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│  Explanation Generator      │
│  - Feature importance       │
│  - Decision reasoning       │
│  - Attack pattern matching  │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│   NLP Engine                │
│  (Generate human text)      │
└──────────┬──────────────────┘
           ↓
User Interface / Reports
```

### 4.9.3 Explanation Components

**Feature Importance**: SHAP values for ML model decisions
**Decision Path**: Trace through detection pipeline
**Attack Context**: Historical pattern matching
**Confidence Metrics**: Probability distributions

### 4.9.4 Natural Language Generation

**Template-Based**: Structured templates for common scenarios
**Context-Aware**: Adapts explanation detail to user expertise
**Multi-Level**: Technical and non-technical explanations

## 4.10 Cross-Platform Sandbox Design

### 4.10.1 Specifications

**Requirements:**
- Process isolation
- Resource limiting
- Behavioral monitoring
- Escape prevention

### 4.10.2 Platform-Specific Designs

**Linux Sandbox**:
- Isolation: Namespaces (PID, NET, MNT, IPC, UTS)
- Resource Limits: cgroups
- System Call Filtering: seccomp-bpf
- Monitoring: ptrace

**Windows Sandbox**:
- Isolation: Job Objects, AppContainer
- Resource Limits: Job Object limits
- Monitoring: API hooking
- Security: Low integrity level

**macOS Sandbox**:
- Isolation: Sandbox Profile Language
- Resource Limits: launchd
- Monitoring: FSEvents, kqueue
- Security: Restricted entitlements

### 4.10.3 Design Justification

**Platform-Native APIs**: Maximum security and performance
**Layered Defense**: Multiple isolation mechanisms
**Monitoring Integration**: Real-time behavioral data collection

## 4.11 Unified Detection Coordinator Design

### 4.11.1 Specifications

**Requirements:**
- Multi-source event processing
- Cross-system correlation
- Priority-based queuing
- Real-time statistics

### 4.11.2 Coordinator Architecture

```
Detection Events (NIDPS, AV, MD, WAF)
    ↓
Event Queue (Priority-based)
    ↓
Correlation Engine
    ↓
    ├→ Temporal Correlation
    ├→ Spatial Correlation
    └→ Behavioral Correlation
         ↓
Unified Threat Assessment
         ↓
Response Coordination
```

### 4.11.3 Correlation Algorithms

**Temporal**: Events within time window (configurable, default 60s)
**Spatial**: Events from same source IP/file hash
**Behavioral**: Similar attack patterns across systems

### 4.11.4 Design Justification

**Event-Driven**: Asynchronous processing for scalability
**Priority Queue**: Critical threats processed first
**Correlation**: Reduces false positives through multi-source validation

---

**Chapter 4 Summary**: This chapter presented the comprehensive design of DRLHSS, covering system architecture, all detection layers (NIDPS, AV, MD, WAF template), the DRL framework, database system, XAI interface, cross-platform sandboxes, and unified coordination. Design decisions were justified based on performance, security, and scalability requirements.

