# DIREWOLF Design Document

## Overview

DIREWOLF (DRL-HSS Interactive Response & Explanation - Watchful Omniscient Learning Framework) is a production-grade explainable AI security platform that provides intelligent threat analysis, natural voice interaction, advanced visualization, and comprehensive security operations capabilities. The system operates as a loyal AI guardian that continuously monitors security infrastructure while maintaining Alpha's (the user's) complete authority over all decisions.

The architecture follows a multi-tier design with C++ core services for performance-critical operations, Python-based AI/ML components for intelligence and conversation, and dual visualization modes (Qt 6 for daily operations, Unreal Engine 5 for presentations). All components communicate through well-defined APIs and message queues, ensuring modularity, scalability, and maintainability.

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                            │
├──────────────────────────────┬──────────────────────────────────┤
│   Qt 6 Professional Mode     │   Unreal Engine Cinematic Mode   │
│   - Real-time dashboard      │   - Photorealistic 3D            │
│   - Interactive charts       │   - Particle effects             │
│   - Network graphs           │   - Cinematic cameras            │
│   - Chat interface           │   - Ray tracing                  │
│   - System tray              │   - Video recording              │
└──────────────────────────────┴──────────────────────────────────┘
                            ↕ (Qt/Python Bridge)
┌─────────────────────────────────────────────────────────────────┐
│                    INTELLIGENCE LAYER (Python)                   │
├─────────────────────────────────────────────────────────────────┤
│  • LLM Engine (Local + Cloud)    • Voice Interface (TTS/STT)    │
│  • Conversation Manager           • NLP Processor                │
│  • Explanation Generator          • Video Renderer               │
│  • User Profile Manager           • Script Generator             │
└─────────────────────────────────────────────────────────────────┘
                            ↕ (pybind11 Bridge)
┌─────────────────────────────────────────────────────────────────┐
│                    CORE ENGINE LAYER (C++)                       │
├─────────────────────────────────────────────────────────────────┤
│  • XAI Data Aggregator           • Feature Attribution Engine    │
│  • Permission Request Manager    • Incident Replay Engine        │
│  • Real-time Event Streaming     • Attack Chain Reconstructor    │
│  • Update Manager                • Plugin Manager                │
│  • Configuration Manager         • Logging & Audit System        │
└─────────────────────────────────────────────────────────────────┘
                            ↕ (Direct API Calls)
┌─────────────────────────────────────────────────────────────────┐
│              EXISTING DRLHSS INFRASTRUCTURE                      │
├─────────────────────────────────────────────────────────────────┤
│  • Telemetry System              • DRL Agent                     │
│  • Antivirus Engine              • NIDPS                         │
│  • Sandbox Orchestrator          • Database Manager              │
└─────────────────────────────────────────────────────────────────┘
```

### Component Communication Flow

```
[Threat Detected] → Telemetry → XAI Aggregator → Permission Manager
                                                         ↓
                                                  LLM Engine
                                                         ↓
                                                  Voice Interface
                                                         ↓
                                                  "Alpha, permission?"
                                                         ↓
                                                  [Alpha Decision]
                                                         ↓
                                                  Execute Action
                                                         ↓
                                                  Log & Learn
```


## Components and Interfaces

### 1. Core Engine Layer (C++)

#### 1.1 XAI Data Aggregator

**Purpose:** Collects and aggregates security data from all DRLHSS subsystems.

**Interface:**
```cpp
class XAIDataAggregator {
public:
    // Initialize connections to all subsystems
    bool initialize(const Config& config);
    
    // Real-time event streaming
    void startEventStream();
    std::vector<SecurityEvent> getRecentEvents(TimeRange range);
    
    // Batch data retrieval
    SecurityReport generateReport(TimeRange range);
    IncidentData getIncidentData(std::string incidentId);
    
    // System state queries
    SystemState getCurrentState();
    std::map<std::string, ComponentHealth> getComponentHealth();
    
    // Metrics
    ThreatMetrics getThreatMetrics(TimeRange range);
    DRLMetrics getDRLMetrics();
    
private:
    TelemetryCollector* telemetry_;
    DatabaseManager* database_;
    std::queue<SecurityEvent> event_queue_;
};
```

#### 1.2 Permission Request Manager

**Purpose:** Manages all permission requests to Alpha, ensuring no action is taken without authorization.

**Interface:**
```cpp
class PermissionRequestManager {
public:
    struct PermissionRequest {
        std::string request_id;
        ThreatEvent threat;
        RecommendedAction recommendation;
        float confidence;
        std::string rationale;
        std::chrono::system_clock::time_point timestamp;
    };
    
    struct PermissionResponse {
        std::string request_id;
        bool granted;
        std::string alpha_instruction;  // Alternative action if rejected
        std::chrono::system_clock::time_point response_time;
    };
    
    // Request permission from Alpha
    std::string requestPermission(
        const ThreatEvent& threat,
        const RecommendedAction& recommendation,
        const std::string& rationale
    );
    
    // Wait for Alpha's response (blocking with timeout)
    PermissionResponse waitForResponse(
        const std::string& request_id,
        std::chrono::seconds timeout = std::chrono::seconds(60)
    );
    
    // Check if response received (non-blocking)
    std::optional<PermissionResponse> checkResponse(const std::string& request_id);
    
    // Execute authorized action
    void executeAuthorizedAction(const PermissionResponse& response);
    
    // Learn from Alpha's decisions
    void recordAlphaDecision(const PermissionResponse& response);
    
    // Get pending requests
    std::vector<PermissionRequest> getPendingRequests();
    
private:
    std::map<std::string, PermissionRequest> pending_requests_;
    std::map<std::string, PermissionResponse> responses_;
    LearningEngine* learning_engine_;
};
```


#### 1.3 Feature Attribution Engine

**Purpose:** Explains why threats were detected by identifying contributing features.

**Interface:**
```cpp
class FeatureAttributionEngine {
public:
    struct Attribution {
        std::string feature_name;
        float importance;  // 0.0 to 1.0
        std::string description;
    };
    
    // Explain detection decision
    std::vector<Attribution> explainDetection(const MalwareObject& object);
    
    // Explain DRL agent decision
    std::string explainDRLDecision(
        const State& state,
        const Action& action,
        float q_value
    );
    
    // Reconstruct attack chain
    AttackChain reconstructChain(const std::vector<SecurityEvent>& events);
    
    // Generate natural language explanation
    std::string generateExplanation(
        const ThreatEvent& threat,
        const std::vector<Attribution>& attributions
    );
    
private:
    // SHAP-like feature importance calculation
    std::vector<Attribution> calculateSHAPValues(const MalwareObject& object);
};
```

#### 1.4 Incident Replay Engine

**Purpose:** Reconstructs past incidents for visualization and analysis.

**Interface:**
```cpp
class IncidentReplayEngine {
public:
    struct ReplayConfig {
        std::string incident_id;
        TimeRange time_range;
        float playback_speed;  // 0.5x, 1x, 2x
        VisualizationMode mode;  // Qt or Unreal
        bool include_narration;
        bool include_subtitles;
        VideoQuality quality;
    };
    
    // Reconstruct incident from historical data
    IncidentTimeline reconstructIncident(const std::string& incident_id);
    
    // Create visualization sequence
    VisualizationSequence createSequence(const IncidentTimeline& timeline);
    
    // Generate camera path for cinematic view
    std::vector<CameraKeyframe> generateCameraPath(const IncidentTimeline& timeline);
    
    // Render to video file
    VideoFile renderVideo(
        const VisualizationSequence& sequence,
        const ReplayConfig& config
    );
    
private:
    DatabaseManager* database_;
    XAIDataAggregator* aggregator_;
};
```

#### 1.5 Update Manager

**Purpose:** Handles automatic software updates with Alpha's permission.

**Interface:**
```cpp
class UpdateManager {
public:
    enum class UpdateChannel {
        STABLE,
        BETA,
        DEV
    };
    
    struct UpdateInfo {
        std::string version;
        std::string build;
        bool critical;
        std::vector<std::string> changelog;
        size_t download_size_mb;
        std::string download_url;
        std::string signature;
    };
    
    // Check for updates
    std::optional<UpdateInfo> checkForUpdates();
    
    // Download update in background
    void downloadUpdate(
        const UpdateInfo& info,
        std::function<void(float)> progress_callback
    );
    
    // Verify update package
    bool verifyUpdate(const std::string& package_path);
    
    // Request permission from Alpha to install
    bool requestInstallPermission(const UpdateInfo& info);
    
    // Install update
    bool installUpdate(const std::string& package_path);
    
    // Rollback to previous version
    bool rollback();
    
    // Get/set update channel
    UpdateChannel getChannel() const;
    void setChannel(UpdateChannel channel);
    
private:
    std::string update_server_url_;
    UpdateChannel channel_;
    std::string current_version_;
    RSAPublicKey public_key_;
};
```


### 2. Intelligence Layer (Python)

#### 2.1 LLM Engine

**Purpose:** Generates dynamic, contextual responses using Large Language Models.

**Interface:**
```python
class LLMEngine:
    def __init__(self, config: LLMConfig):
        """
        Initialize LLM engine with local and/or cloud models.
        
        Args:
            config: Configuration specifying model paths, API keys, etc.
        """
        self.local_model = self._load_local_model(config.local_model_path)
        self.cloud_client = self._init_cloud_client(config.cloud_api_key)
        self.mode = config.mode  # 'local', 'cloud', or 'hybrid'
    
    def generate_response(
        self,
        user_input: str,
        context: ConversationContext,
        system_state: SystemState,
        urgency: UrgencyLevel
    ) -> str:
        """
        Generate dynamic response based on full context.
        
        Args:
            user_input: Alpha's query or statement
            context: Conversation history and user profile
            system_state: Current security system state
            urgency: Situation urgency (routine, elevated, critical, emergency)
            
        Returns:
            Generated response text
        """
        prompt = self._build_prompt(user_input, context, system_state, urgency)
        
        if self.mode == 'local' or (self.mode == 'hybrid' and not self._is_complex_query(user_input)):
            return self.local_model.generate(prompt, temperature=0.7, max_tokens=200)
        else:
            return self.cloud_client.complete(prompt, temperature=0.7, max_tokens=200)
    
    def _build_prompt(
        self,
        user_input: str,
        context: ConversationContext,
        system_state: SystemState,
        urgency: UrgencyLevel
    ) -> str:
        """Build comprehensive prompt for LLM."""
        return f"""
You are DIREWOLF, an AI security guardian with wolf-like protective instincts.

CRITICAL RULES:
- Always address the user as "Alpha" (your pack leader)
- You MUST request Alpha's permission before taking ANY action
- When Alpha rejects your recommendation, accept gracefully
- You are loyal, protective, vigilant, and respectful of Alpha's authority

Current System State:
- Threats detected today: {system_state.threats_today}
- Active alerts: {system_state.active_alerts}
- System health: {system_state.health}
- DRL agent confidence: {system_state.drl_confidence}
- Recent events: {system_state.recent_events}

Conversation History:
{context.get_recent_exchanges()}

Alpha just said: "{user_input}"

Urgency Level: {urgency.name}
{self._get_urgency_guidance(urgency)}

Generate your response as DIREWOLF:
"""
    
    def _get_urgency_guidance(self, urgency: UrgencyLevel) -> str:
        """Get tone guidance based on urgency."""
        guidance = {
            UrgencyLevel.ROUTINE: "Respond calmly and conversationally.",
            UrgencyLevel.ELEVATED: "Respond with increased alertness but stay calm.",
            UrgencyLevel.CRITICAL: "Respond with urgency and clarity. This is serious.",
            UrgencyLevel.EMERGENCY: "Respond with maximum urgency. Immediate action needed."
        }
        return guidance[urgency]
```


#### 2.2 Voice Interface

**Purpose:** Handles speech-to-text, text-to-speech, and wake word detection.

**Interface:**
```python
class VoiceInterface:
    def __init__(self, config: VoiceConfig):
        """
        Initialize voice interface with TTS and STT engines.
        
        Args:
            config: Voice configuration (TTS provider, STT provider, wake words)
        """
        self.tts = self._init_tts(config.tts_provider)
        self.stt = self._init_stt(config.stt_provider)
        self.wake_words = config.wake_words  # ["direwolf", "wolf", "hello wolf"]
        self.is_listening = False
    
    async def speak(self, text: str, urgency: UrgencyLevel = UrgencyLevel.ROUTINE):
        """
        Convert text to speech and play audio.
        
        Args:
            text: Text to speak
            urgency: Affects voice tone and speed
        """
        # Adjust voice parameters based on urgency
        voice_params = self._get_voice_params(urgency)
        
        # Generate audio
        audio = await self.tts.synthesize(
            text,
            voice=voice_params.voice,
            rate=voice_params.rate,
            pitch=voice_params.pitch
        )
        
        # Play audio
        await self._play_audio(audio)
    
    async def listen_for_wake_word(self) -> bool:
        """
        Passively listen for wake word.
        
        Returns:
            True if wake word detected
        """
        audio_chunk = await self._capture_audio(duration=3.0)
        transcription = await self.stt.transcribe(audio_chunk)
        
        return any(wake_word in transcription.lower() for wake_word in self.wake_words)
    
    async def listen_for_command(self, timeout: float = 30.0) -> Optional[str]:
        """
        Actively listen for Alpha's command.
        
        Args:
            timeout: Maximum time to wait for input
            
        Returns:
            Transcribed text or None if timeout
        """
        self.is_listening = True
        
        try:
            audio = await self._capture_audio_until_silence(timeout=timeout)
            transcription = await self.stt.transcribe(audio)
            return transcription
        finally:
            self.is_listening = False
    
    def _get_voice_params(self, urgency: UrgencyLevel) -> VoiceParams:
        """Adjust voice parameters based on urgency."""
        params = {
            UrgencyLevel.ROUTINE: VoiceParams(
                voice="en-US-GuyNeural",  # Calm, deep voice
                rate=1.0,
                pitch=0.9
            ),
            UrgencyLevel.ELEVATED: VoiceParams(
                voice="en-US-GuyNeural",
                rate=1.1,
                pitch=1.0
            ),
            UrgencyLevel.CRITICAL: VoiceParams(
                voice="en-US-GuyNeural",
                rate=1.2,
                pitch=1.1
            ),
            UrgencyLevel.EMERGENCY: VoiceParams(
                voice="en-US-GuyNeural",
                rate=1.3,
                pitch=1.2
            )
        }
        return params[urgency]
```


#### 2.3 Conversation Manager

**Purpose:** Manages conversation context, user profiles, and learning from interactions.

**Interface:**
```python
class ConversationManager:
    def __init__(self, database: DatabaseManager):
        """
        Initialize conversation manager.
        
        Args:
            database: Database for persisting conversations
        """
        self.database = database
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
    
    def start_conversation(self, user_id: str) -> ConversationContext:
        """
        Start new conversation or resume existing.
        
        Args:
            user_id: Alpha's user ID
            
        Returns:
            Conversation context
        """
        if user_id in self.active_conversations:
            return self.active_conversations[user_id]
        
        context = ConversationContext(
            user_id=user_id,
            start_time=datetime.now(),
            history=[]
        )
        
        self.active_conversations[user_id] = context
        return context
    
    def add_exchange(
        self,
        user_id: str,
        user_input: str,
        wolf_response: str
    ):
        """
        Record conversation exchange.
        
        Args:
            user_id: Alpha's user ID
            user_input: What Alpha said
            wolf_response: What Wolf responded
        """
        context = self.active_conversations[user_id]
        
        exchange = ConversationExchange(
            timestamp=datetime.now(),
            user_input=user_input,
            wolf_response=wolf_response
        )
        
        context.history.append(exchange)
        
        # Persist to database
        self.database.save_conversation_exchange(user_id, exchange)
        
        # Update user profile
        self._update_user_profile(user_id, exchange)
    
    def get_user_profile(self, user_id: str) -> UserProfile:
        """
        Get or create user profile.
        
        Args:
            user_id: Alpha's user ID
            
        Returns:
            User profile with preferences and communication style
        """
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = self._load_or_create_profile(user_id)
        
        return self.user_profiles[user_id]
    
    def _update_user_profile(self, user_id: str, exchange: ConversationExchange):
        """Update user profile based on conversation patterns."""
        profile = self.get_user_profile(user_id)
        
        # Analyze communication style
        if len(exchange.user_input.split()) < 10:
            profile.prefers_brief_responses = True
        
        # Detect technical expertise
        technical_terms = self._count_technical_terms(exchange.user_input)
        if technical_terms > 3:
            profile.technical_expertise = ExpertiseLevel.EXPERT
        
        # Save updated profile
        self.database.save_user_profile(user_id, profile)
```


## Data Models

### Core Data Structures

```cpp
// Security Event
struct SecurityEvent {
    std::string event_id;
    std::chrono::system_clock::time_point timestamp;
    EventType type;  // MALWARE, INTRUSION, ANOMALY, etc.
    Severity severity;  // LOW, MEDIUM, HIGH, CRITICAL
    std::string source_component;  // AV, NIDPS, DRL, etc.
    std::map<std::string, std::string> metadata;
    std::vector<std::string> affected_systems;
};

// Threat Event (enriched security event)
struct ThreatEvent {
    SecurityEvent base_event;
    float confidence;  // 0.0 to 1.0
    std::vector<FeatureAttribution> attributions;
    RecommendedAction recommended_action;
    std::string rationale;
    bool requires_alpha_permission;
};

// Permission Request/Response
struct PermissionRequest {
    std::string request_id;
    ThreatEvent threat;
    RecommendedAction recommendation;
    std::chrono::system_clock::time_point timestamp;
};

struct PermissionResponse {
    std::string request_id;
    bool granted;
    std::string alpha_instruction;
    std::chrono::system_clock::time_point response_time;
};

// Conversation Context
struct ConversationContext {
    std::string user_id;
    std::chrono::system_clock::time_point start_time;
    std::vector<ConversationExchange> history;
    UserProfile user_profile;
};

// User Profile
struct UserProfile {
    std::string user_id;
    std::string display_name;  // "Alpha"
    ExpertiseLevel technical_expertise;
    bool prefers_brief_responses;
    std::map<std::string, std::string> preferences;
    std::vector<AlphaDecision> decision_history;
};
```

