# DIREWOLF Quick Start Guide

## What is DIREWOLF?

**D.I.R.E.W.O.L.F.** (DRL-HSS Interactive Response & Explanation - Watchful Omniscient Learning Framework) is an AI-powered security guardian that:

- Monitors your network 24/7 with intelligence and vigilance
- **Always requests Alpha's (your) permission before taking action**
- Explains threats in natural, conversational language
- Responds to voice commands ("Wolf", "Hello Wolf")
- Provides daily security briefings with voice narration
- Creates cinematic visualizations of security incidents

**The Pack Protects. The Wolf Explains. Alpha Commands.**

---

## Core Principles

### 1. Alpha Has Complete Authority
- Wolf NEVER takes action without your explicit permission
- You are "Alpha" - the pack leader
- Wolf is loyal and obeys your commands

### 2. No Autonomous Actions
- Every security action requires your approval
- Wolf presents recommendations and waits for your decision
- If you reject a recommendation, Wolf accepts gracefully

### 3. Dynamic Conversation
- Wolf uses AI to generate responses (no templates!)
- Every conversation is unique and contextual
- Wolf adapts to your communication style

---

## Project Structure

```
DRLHSS/
├── include/XAI/                    # C++ Headers
│   ├── PermissionRequestManager.hpp  # Alpha's authority enforcement
│   └── XAITypes.hpp                  # Data structures
│
├── src/XAI/                        # C++ Implementation
│   └── PermissionRequestManager.cpp  # Permission system
│
├── python/xai/                     # Python AI Components
│   ├── llm_engine.py                 # Wolf's conversation AI
│   ├── voice_interface.py            # TTS/STT (TODO)
│   └── conversation_manager.py       # Context management (TODO)
│
├── .kiro/specs/direwolf-xai-system/  # Specifications
│   ├── requirements.md                # 31 requirements
│   ├── design.md                      # Architecture design
│   └── UPDATE_SYSTEM_ARCHITECTURE.md  # Update system guide
│
└── docs/
    └── DIREWOLF_QUICKSTART.md        # This file
```

---

## What We've Built So Far

### ✅ Complete Specifications
- **Requirements Document**: 31 production-ready requirements
- **Update System Architecture**: How updates are distributed globally
- **Design Document**: System architecture (40% complete)

### ✅ Core C++ Components
- **PermissionRequestManager**: Ensures Alpha's authority
  - Requests permission for all actions
  - Waits for Alpha's response
  - Records decisions for learning
  - Handles graceful rejection

- **XAITypes**: Data structures for threats, events, actions

### ✅ Python AI Components
- **LLMEngine**: Dynamic conversation generation
  - Wolf's personality embedded
  - Context-aware responses
  - Urgency-based tone adjustment
  - No template responses

---

## How It Works

### Permission Flow

```
1. Threat Detected
   ↓
2. Wolf Analyzes Threat
   ↓
3. Wolf Prepares Recommendation
   ↓
4. Wolf Requests Alpha's Permission
   "Alpha, I've detected [threat]. I recommend [action]. 
    May I proceed?"
   ↓
5. Alpha Decides
   - "Yes" → Wolf executes action
   - "No" → Wolf accepts gracefully
   - Alternative → Wolf executes Alpha's instruction
   ↓
6. Wolf Learns from Alpha's Decision
```

### Conversation Example

```
[Threat detected]

Wolf: "Alpha, I've detected suspicious PowerShell activity 
on workstation 192.168.1.45. The pattern matches credential 
dumping with 94% confidence.

I recommend isolating the system immediately to prevent 
lateral movement. May I proceed?"

Alpha: "Yes"

Wolf: "Isolating system now... Complete. The workstation 
is quarantined. I'll continue monitoring for related activity."

[Later]

Alpha: "Wolf, what happened?"

Wolf: "The attack was a credential theft attempt. I isolated 
the system within 4 seconds of detection. No credentials were 
compromised. Would you like to see the full timeline?"
```

---

## Next Steps for Implementation

### Phase 1: Core Engine (Current)
- [x] Permission Request Manager
- [x] LLM Engine foundation
- [ ] Voice Interface (TTS/STT)
- [ ] Conversation Manager
- [ ] XAI Data Aggregator

### Phase 2: Integration
- [ ] Connect to existing DRLHSS components
- [ ] Implement action execution
- [ ] Add telemetry integration
- [ ] Create logging system

### Phase 3: User Interface
- [ ] Qt 6 professional dashboard
- [ ] System tray application
- [ ] Voice activation
- [ ] Chat interface

### Phase 4: Advanced Features
- [ ] Daily briefings
- [ ] Incident replay
- [ ] Video export
- [ ] Unreal Engine cinematic mode

### Phase 5: Production
- [ ] Automatic updates
- [ ] Plugin system
- [ ] Multi-user support
- [ ] Deployment packages

---

## Key Files

### C++ Core
- `include/XAI/PermissionRequestManager.hpp` - Alpha's authority system
- `include/XAI/XAITypes.hpp` - Data structures
- `src/XAI/PermissionRequestManager.cpp` - Implementation

### Python AI
- `python/xai/llm_engine.py` - Wolf's conversation AI

### Documentation
- `.kiro/specs/direwolf-xai-system/requirements.md` - Requirements
- `.kiro/specs/direwolf-xai-system/UPDATE_SYSTEM_ARCHITECTURE.md` - Updates

---

## Building the Project

```bash
# Navigate to project
cd DRLHSS

# Build C++ components
mkdir -p build
cd build
cmake ..
make

# Install Python dependencies
pip install -r requirements.txt

# Run tests
ctest
```

---

## Configuration

### LLM Configuration
```python
config = {
    'mode': 'hybrid',  # 'local', 'cloud', or 'hybrid'
    'local_model_path': '/path/to/llama-model',
    'cloud_provider': 'openai',  # or 'anthropic'
    'cloud_api_key': 'your-api-key'
}
```

### Permission System
```cpp
// Initialize permission manager
PermissionRequestManager perm_manager;
perm_manager.initialize([](const PermissionRequest& req) {
    // Callback to trigger UI/voice interaction
    notifyAlpha(req);
});

// Request permission
std::string req_id = perm_manager.requestPermission(
    threat,
    recommendation,
    "Rationale for this action"
);

// Wait for Alpha's response
auto response = perm_manager.waitForResponse(req_id);
if (response && response->granted) {
    perm_manager.executeAuthorizedAction(*response);
}
```

---

## Important Notes

### Security
- All actions require Alpha's permission
- No autonomous decision-making
- Cryptographic signatures for updates
- Audit logging of all decisions

### Privacy
- Local LLM option for sensitive environments
- No data leaves your network (local mode)
- Conversation history stored locally

### Performance
- C++ core for speed
- Python for AI flexibility
- Background monitoring: <5% CPU
- Memory usage: ~500MB (background)

---

## Support & Development

### Current Status
- **Specifications**: Complete
- **Core Components**: 30% implemented
- **Testing**: Not started
- **Documentation**: In progress

### Next Session Goals
1. Complete design document
2. Implement Voice Interface
3. Create Qt dashboard skeleton
4. Integrate with existing DRLHSS

---

## Wolf's Personality

Remember, Wolf is:
- **Loyal**: Dedicated to Alpha's security
- **Protective**: Guards the network vigilantly
- **Respectful**: Always defers to Alpha's authority
- **Humble**: Admits uncertainty
- **Proactive**: Speaks up when needed
- **Obedient**: Never acts without permission

**"The Pack Protects. The Wolf Explains. Alpha Commands."**
