"""
DIREWOLF XAI NLP Engine
Intent classification, entity extraction, and conversation management
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Intent:
    """Recognized intent"""
    name: str
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Conversation context state"""
    user_id: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    current_intent: Optional[Intent] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)


class IntentClassifier:
    """Intent classification for security commands"""
    
    def __init__(self):
        self.intents = {
            'security.scan': [
                r'scan\s+(for\s+)?(threats|malware|viruses)',
                r'check\s+(for\s+)?(threats|security)',
                r'run\s+.*scan',
                r'detect\s+(threats|malware)'
            ],
            'security.status': [
                r'(what|how).*security\s+status',
                r'system\s+status',
                r'are\s+we\s+secure',
                r'any\s+threats'
            ],
            'file.find_duplicates': [
                r'find\s+duplicate\s+files',
                r'search\s+for\s+duplicates',
                r'duplicate\s+file\s+detection'
            ],
            'file.cleanup': [
                r'clean\s+up\s+files',
                r'delete\s+.*files',
                r'remove\s+duplicates'
            ],
            'system.performance': [
                r'(check|show)\s+performance',
                r'system\s+resources',
                r'cpu\s+usage',
                r'memory\s+usage'
            ],
            'help': [
                r'help',
                r'what\s+can\s+you\s+do',
                r'commands',
                r'how\s+do\s+i'
            ]
        }
        self.initialized = True
    
    def classify(self, text: str) -> Intent:
        """
        Classify intent from text
        
        Args:
            text: Input text
            
        Returns:
            Intent object with name and confidence
        """
        text = text.lower().strip()
        
        # Check each intent pattern
        for intent_name, patterns in self.intents.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return Intent(
                        name=intent_name,
                        confidence=0.95
                    )
        
        # Default intent
        return Intent(
            name='unknown',
            confidence=0.5
        )


class EntityExtractor:
    """Extract entities from text"""
    
    def __init__(self):
        self.entity_patterns = {
            'file_path': r'[A-Za-z]:\\[^\s]+|/[^\s]+',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'number': r'\b\d+\b',
            'time': r'\b\d{1,2}:\d{2}\b'
        }
        self.initialized = True
    
    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types to values
        """
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                entities[entity_type] = matches
        
        return entities


class ConversationManager:
    """Manage conversation state and context"""
    
    def __init__(self, max_history: int = 10):
        self.contexts: Dict[str, ConversationContext] = {}
        self.max_history = max_history
        self.initialized = True
    
    def get_context(self, user_id: str) -> ConversationContext:
        """Get or create conversation context"""
        if user_id not in self.contexts:
            self.contexts[user_id] = ConversationContext(user_id=user_id)
        return self.contexts[user_id]
    
    def update_context(self, user_id: str, intent: Intent, 
                      user_input: str, response: str):
        """Update conversation context"""
        context = self.get_context(user_id)
        
        # Add to history
        context.history.append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'intent': intent.name,
            'response': response
        })
        
        # Trim history
        if len(context.history) > self.max_history:
            context.history = context.history[-self.max_history:]
        
        context.current_intent = intent
        context.last_update = datetime.now()
    
    def get_history(self, user_id: str, count: int = 5) -> List[Dict]:
        """Get recent conversation history"""
        context = self.get_context(user_id)
        return context.history[-count:]
    
    def clear_context(self, user_id: str):
        """Clear conversation context"""
        if user_id in self.contexts:
            del self.contexts[user_id]


class ResponseGenerator:
    """Generate responses based on intent"""
    
    def __init__(self):
        self.templates = {
            'security.scan': [
                "Starting security scan now. I'll check for threats and malware.",
                "Initiating comprehensive security scan of your system.",
                "Running threat detection scan. This may take a moment."
            ],
            'security.status': [
                "Your system security status is: {status}. No active threats detected.",
                "Security systems are operational. All monitoring active.",
                "Current security posture: {status}. Everything looks good."
            ],
            'file.find_duplicates': [
                "Searching for duplicate files across your system.",
                "Scanning for duplicate files. I'll report what I find.",
                "Starting duplicate file detection."
            ],
            'file.cleanup': [
                "I can help clean up files. What would you like to remove?",
                "Ready to clean up files. Please confirm the operation.",
                "File cleanup initiated. Proceeding with caution."
            ],
            'system.performance': [
                "System performance: CPU at {cpu}%, Memory at {memory}%.",
                "Checking system resources now.",
                "Performance metrics: All systems nominal."
            ],
            'help': [
                "I can help with: security scans, system status, file management, and performance monitoring.",
                "Available commands: scan for threats, check status, find duplicates, show performance.",
                "I'm your AI security assistant. Ask me about threats, files, or system status."
            ],
            'unknown': [
                "I'm not sure I understand. Try asking about security, files, or system status.",
                "Could you rephrase that? I can help with security scans, file management, and system monitoring.",
                "I didn't quite catch that. Ask me about threats, performance, or files."
            ]
        }
        self.initialized = True
    
    def generate(self, intent: Intent, context: Optional[Dict] = None) -> str:
        """
        Generate response for intent
        
        Args:
            intent: Recognized intent
            context: Optional context for template variables
            
        Returns:
            Generated response text
        """
        templates = self.templates.get(intent.name, self.templates['unknown'])
        template = templates[0]  # Use first template
        
        # Fill in template variables
        if context:
            try:
                return template.format(**context)
            except KeyError:
                pass
        
        return template


class NLPEngine:
    """Complete NLP engine integrating all components"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.conversation_manager = ConversationManager()
        self.response_generator = ResponseGenerator()
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize NLP engine"""
        try:
            logger.info("NLP Engine initialized")
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize NLP engine: {e}")
            return False
    
    def process(self, text: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Process natural language input
        
        Args:
            text: Input text
            user_id: User identifier
            
        Returns:
            Processing result with intent, entities, and response
        """
        if not self.initialized:
            self.initialize()
        
        try:
            # Classify intent
            intent = self.intent_classifier.classify(text)
            
            # Extract entities
            entities = self.entity_extractor.extract(text)
            intent.entities = entities
            
            # Get conversation context
            context = self.conversation_manager.get_context(user_id)
            
            # Generate response
            response = self.response_generator.generate(intent)
            
            # Update conversation
            self.conversation_manager.update_context(
                user_id, intent, text, response
            )
            
            return {
                'success': True,
                'intent': intent.name,
                'confidence': intent.confidence,
                'entities': entities,
                'response': response,
                'context': {
                    'history_count': len(context.history),
                    'last_intent': context.current_intent.name if context.current_intent else None
                }
            }
        except Exception as e:
            logger.error(f"NLP processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': "I encountered an error processing your request."
            }
    
    def get_conversation_history(self, user_id: str = "default", count: int = 5) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_manager.get_history(user_id, count)


# Module-level instance
_nlp_engine = None


def get_nlp_engine() -> NLPEngine:
    """Get global NLP engine instance"""
    global _nlp_engine
    if _nlp_engine is None:
        _nlp_engine = NLPEngine()
        _nlp_engine.initialize()
    return _nlp_engine


if __name__ == "__main__":
    # Test NLP engine
    logging.basicConfig(level=logging.INFO)
    
    engine = get_nlp_engine()
    
    test_inputs = [
        "scan for threats",
        "what's my security status",
        "find duplicate files",
        "show system performance",
        "help me"
    ]
    
    print("Testing NLP Engine:\n")
    for text in test_inputs:
        result = engine.process(text)
        print(f"Input: {text}")
        print(f"Intent: {result['intent']} ({result['confidence']:.2f})")
        print(f"Response: {result['response']}")
        print()
