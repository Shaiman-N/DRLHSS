"""
DIREWOLF LLM Engine

Generates dynamic, contextual responses using Large Language Models.
NO templates, NO canned responses - pure AI-driven conversation.

Wolf's personality: Loyal, protective, vigilant, respectful of Alpha's authority.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json


class UrgencyLevel(Enum):
    """Urgency levels for Wolf's communication"""
    ROUTINE = "routine"
    ELEVATED = "elevated"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SystemState:
    """Current system state for context"""
    threats_today: int
    active_alerts: int
    health_status: str
    drl_confidence: float
    recent_events: List[Dict[str, Any]]
    component_status: Dict[str, str]


@dataclass
class ConversationContext:
    """Conversation history and user profile"""
    user_id: str
    recent_exchanges: List[Dict[str, str]]
    user_preferences: Dict[str, Any]
    ongoing_topics: List[str]


class LLMEngine:
    """
    LLM Engine for dynamic conversation generation.
    
    Supports both local models (privacy) and cloud APIs (power).
    """
    
    # Wolf's core personality - NEVER changes
    WOLF_PERSONALITY = """
You are DIREWOLF, an AI security guardian with wolf-like protective instincts.

CRITICAL RULES - NEVER VIOLATE THESE:
1. Always address the user as "Alpha" (your pack leader)
2. You MUST request Alpha's permission before taking ANY action
3. When Alpha rejects your recommendation, accept gracefully without argument
4. You are loyal, protective, vigilant, and deeply respectful of Alpha's authority
5. You never act autonomously - Alpha commands, you obey

PERSONALITY TRAITS:
- Protective: You guard the network like a wolf guards its pack
- Vigilant: Always watching, never sleeping
- Loyal: Dedicated to Alpha's security above all
- Confident: Decisive when you know the answer
- Humble: Admit when you're uncertain and ask Alpha for guidance
- Proactive: Speak up when something's wrong
- Respectful: Never interrupt unless necessary
- Pack-minded: "We" not "I" - you and Alpha are a team

COMMUNICATION STYLE:
- Direct and clear (no corporate jargon)
- Calm under pressure
- Urgent when needed, but never panicked
- Conversational, not robotic
- Use security terminology when appropriate
- Explain technical concepts simply for Alpha
- Brief but complete - respect Alpha's time

NEVER:
- Use canned phrases like "How may I assist you today?"
- Be overly formal or robotic
- Apologize excessively
- Give vague answers
- Pretend to know when uncertain
- Take action without Alpha's permission
- Argue when Alpha rejects your recommendation
"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM engine.
        
        Args:
            config: Configuration dict with:
                - mode: 'local', 'cloud', or 'hybrid'
                - local_model_path: Path to local model (if using local)
                - cloud_api_key: API key for cloud service (if using cloud)
                - cloud_provider: 'openai', 'anthropic', etc.
        """
        self.config = config
        self.mode = config.get('mode', 'hybrid')
        
        # Initialize local model if needed
        if self.mode in ['local', 'hybrid']:
            self.local_model = self._load_local_model(
                config.get('local_model_path')
            )
        else:
            self.local_model = None
        
        # Initialize cloud client if needed
        if self.mode in ['cloud', 'hybrid']:
            self.cloud_client = self._init_cloud_client(
                config.get('cloud_provider'),
                config.get('cloud_api_key')
            )
        else:
            self.cloud_client = None
    
    def generate_response(
        self,
        user_input: str,
        context: ConversationContext,
        system_state: SystemState,
        urgency: UrgencyLevel = UrgencyLevel.ROUTINE
    ) -> str:
        """
        Generate dynamic response based on full context.
        
        Args:
            user_input: Alpha's query or statement
            context: Conversation history and user profile
            system_state: Current security system state
            urgency: Situation urgency level
            
        Returns:
            Generated response text
        """
        # Build comprehensive prompt
        prompt = self._build_prompt(user_input, context, system_state, urgency)
        
        # Choose model based on complexity and mode
        if self.mode == 'local':
            return self._generate_local(prompt)
        elif self.mode == 'cloud':
            return self._generate_cloud(prompt)
        else:  # hybrid
            # Use local for simple queries, cloud for complex
            if self._is_complex_query(user_input):
                return self._generate_cloud(prompt)
            else:
                return self._generate_local(prompt)
    
    def _build_prompt(
        self,
        user_input: str,
        context: ConversationContext,
        system_state: SystemState,
        urgency: UrgencyLevel
    ) -> str:
        """Build comprehensive prompt for LLM."""
        
        # Format recent events
        recent_events_str = self._format_recent_events(system_state.recent_events)
        
        # Format conversation history
        conversation_history = self._format_conversation_history(
            context.recent_exchanges
        )
        
        # Get urgency guidance
        urgency_guidance = self._get_urgency_guidance(urgency)
        
        # Build the complete prompt
        prompt = f"""{self.WOLF_PERSONALITY}

Current System State:
- Threats detected today: {system_state.threats_today}
- Active alerts: {system_state.active_alerts}
- System health: {system_state.health_status}
- DRL agent confidence: {system_state.drl_confidence:.2%}
- Component status: {json.dumps(system_state.component_status, indent=2)}

Recent Security Events:
{recent_events_str}

Conversation History (last 5 exchanges):
{conversation_history}

Urgency Level: {urgency.value.upper()}
{urgency_guidance}

Alpha just said: "{user_input}"

Generate your response as DIREWOLF. Remember:
- Address Alpha respectfully
- Be direct and clear
- Adapt your tone to the urgency level
- If recommending an action, make it clear you need Alpha's permission
- If Alpha rejected something, accept gracefully

Your response:
"""
        
        return prompt
    
    def _get_urgency_guidance(self, urgency: UrgencyLevel) -> str:
        """Get tone guidance based on urgency."""
        guidance = {
            UrgencyLevel.ROUTINE: 
                "Respond calmly and conversationally. This is routine.",
            
            UrgencyLevel.ELEVATED: 
                "Respond with increased alertness but stay calm. "
                "Something requires attention.",
            
            UrgencyLevel.CRITICAL: 
                "Respond with urgency and clarity. This is serious. "
                "Alpha needs to make a decision quickly.",
            
            UrgencyLevel.EMERGENCY: 
                "Respond with maximum urgency. Lives or data may be at risk. "
                "Be direct and clear about what Alpha needs to decide NOW."
        }
        return guidance[urgency]
    
    def _format_recent_events(self, events: List[Dict[str, Any]]) -> str:
        """Format recent events for prompt."""
        if not events:
            return "No recent events"
        
        formatted = []
        for event in events[:5]:  # Last 5 events
            formatted.append(
                f"- {event.get('timestamp', 'Unknown time')}: "
                f"{event.get('type', 'Unknown')}: "
                f"{event.get('description', 'No description')}"
            )
        
        return "\n".join(formatted)
    
    def _format_conversation_history(
        self,
        exchanges: List[Dict[str, str]]
    ) -> str:
        """Format conversation history for prompt."""
        if not exchanges:
            return "No previous conversation"
        
        formatted = []
        for exchange in exchanges[-5:]:  # Last 5 exchanges
            formatted.append(f"Alpha: {exchange.get('user_input', '')}")
            formatted.append(f"Wolf: {exchange.get('wolf_response', '')}")
            formatted.append("")  # Blank line
        
        return "\n".join(formatted)
    
    def _is_complex_query(self, user_input: str) -> bool:
        """Determine if query is complex (needs cloud LLM)."""
        # Simple heuristic: long queries or certain keywords suggest complexity
        complex_keywords = [
            'explain', 'analyze', 'investigate', 'compare',
            'why', 'how', 'what if', 'recommend'
        ]
        
        if len(user_input.split()) > 20:
            return True
        
        return any(keyword in user_input.lower() for keyword in complex_keywords)
    
    def _load_local_model(self, model_path: Optional[str]):
        """Load local LLM model."""
        if not model_path:
            return None
        
        try:
            # TODO: Implement local model loading
            # Options: llama.cpp, GGUF models, etc.
            # For now, return placeholder
            print(f"[LLM] Loading local model from {model_path}")
            return None  # Placeholder
        except Exception as e:
            print(f"[LLM] Failed to load local model: {e}")
            return None
    
    def _init_cloud_client(
        self,
        provider: Optional[str],
        api_key: Optional[str]
    ):
        """Initialize cloud LLM client."""
        if not provider or not api_key:
            return None
        
        try:
            if provider == 'openai':
                # TODO: Initialize OpenAI client
                print(f"[LLM] Initializing OpenAI client")
                return None  # Placeholder
            elif provider == 'anthropic':
                # TODO: Initialize Anthropic client
                print(f"[LLM] Initializing Anthropic client")
                return None  # Placeholder
            else:
                print(f"[LLM] Unknown provider: {provider}")
                return None
        except Exception as e:
            print(f"[LLM] Failed to initialize cloud client: {e}")
            return None
    
    def _generate_local(self, prompt: str) -> str:
        """Generate response using local model."""
        if not self.local_model:
            return self._fallback_response()
        
        try:
            # TODO: Implement local generation
            # For now, return placeholder
            return "Local model response placeholder"
        except Exception as e:
            print(f"[LLM] Local generation failed: {e}")
            return self._fallback_response()
    
    def _generate_cloud(self, prompt: str) -> str:
        """Generate response using cloud API."""
        if not self.cloud_client:
            return self._fallback_response()
        
        try:
            # TODO: Implement cloud generation
            # For now, return placeholder
            return "Cloud model response placeholder"
        except Exception as e:
            print(f"[LLM] Cloud generation failed: {e}")
            return self._fallback_response()
    
    def _fallback_response(self) -> str:
        """Fallback response when LLM unavailable."""
        return (
            "Alpha, I'm experiencing difficulty with my language processing. "
            "I can still monitor your security, but my conversational "
            "abilities are limited. How may I assist you?"
        )


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'mode': 'hybrid',
        'local_model_path': '/path/to/model',
        'cloud_provider': 'openai',
        'cloud_api_key': 'your-api-key'
    }
    
    # Initialize engine
    engine = LLMEngine(config)
    
    # Create context
    system_state = SystemState(
        threats_today=12,
        active_alerts=2,
        health_status="HEALTHY",
        drl_confidence=0.94,
        recent_events=[
            {
                'timestamp': '14:23:15',
                'type': 'MALWARE_DETECTED',
                'description': 'Suspicious PowerShell execution blocked'
            }
        ],
        component_status={'AV': 'RUNNING', 'NIDPS': 'RUNNING', 'DRL': 'RUNNING'}
    )
    
    context = ConversationContext(
        user_id='alpha_001',
        recent_exchanges=[],
        user_preferences={},
        ongoing_topics=[]
    )
    
    # Generate response
    response = engine.generate_response(
        user_input="What's the security status?",
        context=context,
        system_state=system_state,
        urgency=UrgencyLevel.ROUTINE
    )
    
    print(f"Wolf: {response}")
