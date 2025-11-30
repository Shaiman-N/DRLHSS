"""
DIREWOLF Conversation Manager

Manages conversation context, user profiles, and learning from Alpha's decisions.
Wolf remembers everything and adapts to Alpha's communication style.
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
import threading


class ExpertiseLevel(Enum):
    """User's technical expertise level"""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


@dataclass
class ConversationExchange:
    """Single conversation exchange"""
    timestamp: str
    user_input: str
    wolf_response: str
    context: str  # What was happening at the time
    urgency_level: str
    response_time_ms: int  # How long Wolf took to respond


@dataclass
class UserProfile:
    """Alpha's profile and preferences"""
    user_id: str
    display_name: str  # "Alpha"
    technical_expertise: ExpertiseLevel
    prefers_brief_responses: bool
    preferred_detail_level: str  # "brief", "moderate", "detailed"
    communication_style: str  # "formal", "casual", "technical"
    decision_patterns: List[str]
    last_active: str
    total_conversations: int
    avg_session_length_minutes: float
    preferred_voice_speed: float
    timezone: str


@dataclass
class AlphaDecision:
    """Record of Alpha's decision"""
    decision_id: str
    timestamp: str
    threat_type: str
    wolf_recommendation: str
    alpha_decision: str  # "approved", "rejected", "modified"
    alpha_alternative: Optional[str]  # If Alpha provided alternative
    confidence_level: float
    outcome: Optional[str]  # What happened after the decision
    learned_pattern: Optional[str]  # What Wolf learned


class ConversationManager:
    """
    Manages all conversation context and user learning.
    
    Responsibilities:
    - Track conversation history
    - Maintain user profiles
    - Learn from Alpha's decisions
    - Adapt communication style
    - Provide context for LLM
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize conversation manager.
        
        Args:
            config: Configuration dict with:
                - database_path: Path to SQLite database
                - max_history_days: How long to keep history
                - learning_enabled: Whether to learn from decisions
        """
        self.config = config
        self.db_path = config.get('database_path', 'direwolf_conversations.db')
        self.max_history_days = config.get('max_history_days', 90)
        self.learning_enabled = config.get('learning_enabled', True)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # In-memory caches
        self.active_conversations: Dict[str, List[ConversationExchange]] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # Initialize database
        self._init_database()
        
        # Load user profiles
        self._load_user_profiles()
        
        print("[Conversation] DIREWOLF Conversation Manager initialized")
    
    def start_conversation(self, user_id: str = "alpha_001") -> str:
        """
        Start new conversation session.
        
        Args:
            user_id: Alpha's user ID
            
        Returns:
            Conversation session ID
        """
        with self.lock:
            session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize conversation history
            if user_id not in self.active_conversations:
                self.active_conversations[user_id] = []
            
            # Update user profile
            profile = self.get_user_profile(user_id)
            profile.last_active = datetime.now().isoformat()
            profile.total_conversations += 1
            
            self._save_user_profile(profile)
            
            print(f"[Conversation] Started session {session_id} for {profile.display_name}")
            return session_id
    
    def add_exchange(
        self,
        user_id: str,
        user_input: str,
        wolf_response: str,
        context: str = "",
        urgency_level: str = "routine",
        response_time_ms: int = 0
    ):
        """
        Record conversation exchange.
        
        Args:
            user_id: Alpha's user ID
            user_input: What Alpha said
            wolf_response: What Wolf responded
            context: System context at the time
            urgency_level: Urgency of the situation
            response_time_ms: How long Wolf took to respond
        """
        with self.lock:
            exchange = ConversationExchange(
                timestamp=datetime.now().isoformat(),
                user_input=user_input,
                wolf_response=wolf_response,
                context=context,
                urgency_level=urgency_level,
                response_time_ms=response_time_ms
            )
            
            # Add to active conversation
            if user_id not in self.active_conversations:
                self.active_conversations[user_id] = []
            
            self.active_conversations[user_id].append(exchange)
            
            # Keep only recent exchanges in memory (last 50)
            if len(self.active_conversations[user_id]) > 50:
                self.active_conversations[user_id] = self.active_conversations[user_id][-50:]
            
            # Save to database
            self._save_exchange_to_db(user_id, exchange)
            
            # Analyze and learn from this exchange
            if self.learning_enabled:
                self._analyze_exchange(user_id, exchange)
    
    def get_conversation_context(
        self,
        user_id: str,
        last_n_exchanges: int = 5
    ) -> Dict[str, Any]:
        """
        Get conversation context for LLM.
        
        Args:
            user_id: Alpha's user ID
            last_n_exchanges: Number of recent exchanges to include
            
        Returns:
            Context dict for LLM
        """
        with self.lock:
            profile = self.get_user_profile(user_id)
            
            # Get recent exchanges
            recent_exchanges = []
            if user_id in self.active_conversations:
                recent_exchanges = self.active_conversations[user_id][-last_n_exchanges:]
            
            # Format for LLM
            formatted_exchanges = []
            for exchange in recent_exchanges:
                formatted_exchanges.append({
                    'timestamp': exchange.timestamp,
                    'user_input': exchange.user_input,
                    'wolf_response': exchange.wolf_response,
                    'urgency': exchange.urgency_level
                })
            
            return {
                'user_profile': {
                    'display_name': profile.display_name,
                    'expertise_level': profile.technical_expertise.value,
                    'prefers_brief': profile.prefers_brief_responses,
                    'detail_level': profile.preferred_detail_level,
                    'communication_style': profile.communication_style
                },
                'recent_exchanges': formatted_exchanges,
                'conversation_stats': {
                    'total_conversations': profile.total_conversations,
                    'avg_session_length': profile.avg_session_length_minutes
                },
                'decision_patterns': profile.decision_patterns
            }
    
    def get_user_profile(self, user_id: str) -> UserProfile:
        """
        Get or create user profile.
        
        Args:
            user_id: Alpha's user ID
            
        Returns:
            User profile
        """
        if user_id not in self.user_profiles:
            # Create default profile
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                display_name="Alpha",
                technical_expertise=ExpertiseLevel.INTERMEDIATE,
                prefers_brief_responses=False,
                preferred_detail_level="moderate",
                communication_style="professional",
                decision_patterns=[],
                last_active=datetime.now().isoformat(),
                total_conversations=0,
                avg_session_length_minutes=5.0,
                preferred_voice_speed=1.0,
                timezone="UTC"
            )
            
            self._save_user_profile(self.user_profiles[user_id])
        
        return self.user_profiles[user_id]
    
    def record_alpha_decision(
        self,
        user_id: str,
        threat_type: str,
        wolf_recommendation: str,
        alpha_decision: str,
        alpha_alternative: Optional[str] = None,
        confidence_level: float = 0.0
    ) -> str:
        """
        Record Alpha's decision for learning.
        
        Args:
            user_id: Alpha's user ID
            threat_type: Type of threat
            wolf_recommendation: What Wolf recommended
            alpha_decision: Alpha's decision (approved/rejected/modified)
            alpha_alternative: Alternative action if rejected
            confidence_level: Wolf's confidence in recommendation
            
        Returns:
            Decision ID
        """
        with self.lock:
            decision_id = f"DEC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
            
            decision = AlphaDecision(
                decision_id=decision_id,
                timestamp=datetime.now().isoformat(),
                threat_type=threat_type,
                wolf_recommendation=wolf_recommendation,
                alpha_decision=alpha_decision,
                alpha_alternative=alpha_alternative,
                confidence_level=confidence_level,
                outcome=None,  # Will be updated later
                learned_pattern=None  # Will be determined by analysis
            )
            
            # Save to database
            self._save_decision_to_db(decision)
            
            # Learn from this decision
            if self.learning_enabled:
                self._learn_from_decision(user_id, decision)
            
            print(f"[Conversation] Recorded Alpha decision: {alpha_decision} for {threat_type}")
            return decision_id
    
    def analyze_communication_style(self, user_id: str) -> Dict[str, Any]:
        """
        Analyze Alpha's communication style.
        
        Args:
            user_id: Alpha's user ID
            
        Returns:
            Analysis results
        """
        with self.lock:
            if user_id not in self.active_conversations:
                return {}
            
            exchanges = self.active_conversations[user_id]
            
            if not exchanges:
                return {}
            
            # Analyze patterns
            total_exchanges = len(exchanges)
            brief_responses = 0
            technical_terms = 0
            question_count = 0
            avg_response_length = 0
            
            for exchange in exchanges:
                user_input = exchange.user_input
                
                # Count brief responses (< 10 words)
                if len(user_input.split()) < 10:
                    brief_responses += 1
                
                # Count technical terms
                technical_keywords = [
                    'malware', 'vulnerability', 'exploit', 'payload',
                    'signature', 'heuristic', 'sandbox', 'quarantine',
                    'firewall', 'intrusion', 'anomaly', 'threat'
                ]
                
                for keyword in technical_keywords:
                    if keyword.lower() in user_input.lower():
                        technical_terms += 1
                        break
                
                # Count questions
                if '?' in user_input:
                    question_count += 1
                
                avg_response_length += len(user_input.split())
            
            avg_response_length /= total_exchanges
            
            # Determine characteristics
            analysis = {
                'prefers_brief_responses': brief_responses / total_exchanges > 0.6,
                'technical_expertise': (
                    ExpertiseLevel.EXPERT if technical_terms / total_exchanges > 0.3
                    else ExpertiseLevel.INTERMEDIATE if technical_terms / total_exchanges > 0.1
                    else ExpertiseLevel.NOVICE
                ),
                'asks_many_questions': question_count / total_exchanges > 0.4,
                'avg_response_length': avg_response_length,
                'communication_style': (
                    'technical' if technical_terms / total_exchanges > 0.2
                    else 'casual' if avg_response_length < 8
                    else 'professional'
                )
            }
            
            # Update user profile
            profile = self.get_user_profile(user_id)
            profile.prefers_brief_responses = analysis['prefers_brief_responses']
            profile.technical_expertise = analysis['technical_expertise']
            profile.communication_style = analysis['communication_style']
            
            if analysis['prefers_brief_responses']:
                profile.preferred_detail_level = 'brief'
            elif analysis['asks_many_questions']:
                profile.preferred_detail_level = 'detailed'
            else:
                profile.preferred_detail_level = 'moderate'
            
            self._save_user_profile(profile)
            
            return analysis
    
    def get_decision_patterns(self, user_id: str) -> List[str]:
        """
        Get Alpha's decision patterns.
        
        Args:
            user_id: Alpha's user ID
            
        Returns:
            List of decision patterns
        """
        # Query database for decision history
        patterns = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent decisions
            cursor.execute("""
                SELECT threat_type, wolf_recommendation, alpha_decision, alpha_alternative
                FROM alpha_decisions 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 50
            """, (user_id,))
            
            decisions = cursor.fetchall()
            
            # Analyze patterns
            rejection_patterns = {}
            approval_patterns = {}
            
            for threat_type, recommendation, decision, alternative in decisions:
                if decision == 'rejected':
                    key = f"{threat_type}:{recommendation}"
                    rejection_patterns[key] = rejection_patterns.get(key, 0) + 1
                elif decision == 'approved':
                    key = f"{threat_type}:{recommendation}"
                    approval_patterns[key] = approval_patterns.get(key, 0) + 1
            
            # Generate pattern descriptions
            for pattern, count in rejection_patterns.items():
                if count >= 3:  # Pattern if rejected 3+ times
                    threat_type, recommendation = pattern.split(':', 1)
                    patterns.append(f"Alpha typically rejects {recommendation} for {threat_type}")
            
            for pattern, count in approval_patterns.items():
                if count >= 5:  # Pattern if approved 5+ times
                    threat_type, recommendation = pattern.split(':', 1)
                    patterns.append(f"Alpha typically approves {recommendation} for {threat_type}")
            
            conn.close()
            
        except Exception as e:
            print(f"[Conversation] Error analyzing decision patterns: {e}")
        
        return patterns
    
    def _init_database(self):
        """Initialize SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Conversation exchanges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_exchanges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    wolf_response TEXT NOT NULL,
                    context TEXT,
                    urgency_level TEXT,
                    response_time_ms INTEGER
                )
            """)
            
            # User profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """)
            
            # Alpha decisions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alpha_decisions (
                    decision_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    threat_type TEXT NOT NULL,
                    wolf_recommendation TEXT NOT NULL,
                    alpha_decision TEXT NOT NULL,
                    alpha_alternative TEXT,
                    confidence_level REAL,
                    outcome TEXT,
                    learned_pattern TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_exchanges_user_time ON conversation_exchanges(user_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_user_time ON alpha_decisions(user_id, timestamp)")
            
            conn.commit()
            conn.close()
            
            print("[Conversation] Database initialized successfully")
            
        except Exception as e:
            print(f"[Conversation] Database initialization failed: {e}")
    
    def _load_user_profiles(self):
        """Load user profiles from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT user_id, profile_data FROM user_profiles")
            rows = cursor.fetchall()
            
            for user_id, profile_json in rows:
                profile_dict = json.loads(profile_json)
                profile_dict['technical_expertise'] = ExpertiseLevel(profile_dict['technical_expertise'])
                self.user_profiles[user_id] = UserProfile(**profile_dict)
            
            conn.close()
            
            print(f"[Conversation] Loaded {len(self.user_profiles)} user profiles")
            
        except Exception as e:
            print(f"[Conversation] Error loading user profiles: {e}")
    
    def _save_user_profile(self, profile: UserProfile):
        """Save user profile to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert to dict and handle enum
            profile_dict = asdict(profile)
            profile_dict['technical_expertise'] = profile.technical_expertise.value
            
            cursor.execute("""
                INSERT OR REPLACE INTO user_profiles (user_id, profile_data, last_updated)
                VALUES (?, ?, ?)
            """, (
                profile.user_id,
                json.dumps(profile_dict),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"[Conversation] Error saving user profile: {e}")
    
    def _save_exchange_to_db(self, user_id: str, exchange: ConversationExchange):
        """Save conversation exchange to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO conversation_exchanges 
                (user_id, timestamp, user_input, wolf_response, context, urgency_level, response_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                exchange.timestamp,
                exchange.user_input,
                exchange.wolf_response,
                exchange.context,
                exchange.urgency_level,
                exchange.response_time_ms
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"[Conversation] Error saving exchange: {e}")
    
    def _save_decision_to_db(self, decision: AlphaDecision):
        """Save Alpha's decision to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO alpha_decisions 
                (decision_id, user_id, timestamp, threat_type, wolf_recommendation, 
                 alpha_decision, alpha_alternative, confidence_level, outcome, learned_pattern)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision.decision_id,
                decision.decision_id.split('_')[-1],  # Extract user_id
                decision.timestamp,
                decision.threat_type,
                decision.wolf_recommendation,
                decision.alpha_decision,
                decision.alpha_alternative,
                decision.confidence_level,
                decision.outcome,
                decision.learned_pattern
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"[Conversation] Error saving decision: {e}")
    
    def _analyze_exchange(self, user_id: str, exchange: ConversationExchange):
        """Analyze conversation exchange for learning."""
        # Update communication style analysis periodically
        if len(self.active_conversations.get(user_id, [])) % 10 == 0:
            self.analyze_communication_style(user_id)
    
    def _learn_from_decision(self, user_id: str, decision: AlphaDecision):
        """Learn from Alpha's decision."""
        profile = self.get_user_profile(user_id)
        
        # Update decision patterns
        pattern = f"{decision.threat_type}:{decision.alpha_decision}"
        if pattern not in profile.decision_patterns:
            profile.decision_patterns.append(pattern)
        
        # Keep only recent patterns (last 20)
        if len(profile.decision_patterns) > 20:
            profile.decision_patterns = profile.decision_patterns[-20:]
        
        self._save_user_profile(profile)
        
        print(f"[Conversation] Learned from Alpha's decision: {decision.alpha_decision}")


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'database_path': 'direwolf_conversations.db',
        'max_history_days': 90,
        'learning_enabled': True
    }
    
    # Initialize
    conv_manager = ConversationManager(config)
    
    # Start conversation
    session_id = conv_manager.start_conversation("alpha_001")
    
    # Add exchange
    conv_manager.add_exchange(
        user_id="alpha_001",
        user_input="What's the security status?",
        wolf_response="Alpha, your network is secure. I've blocked 3 threats today.",
        context="routine_check",
        urgency_level="routine",
        response_time_ms=1200
    )
    
    # Record decision
    conv_manager.record_alpha_decision(
        user_id="alpha_001",
        threat_type="malware",
        wolf_recommendation="quarantine_file",
        alpha_decision="approved",
        confidence_level=0.94
    )
    
    # Get context for LLM
    context = conv_manager.get_conversation_context("alpha_001")
    print(f"Context: {json.dumps(context, indent=2)}")
