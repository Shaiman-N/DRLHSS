"""
DIREWOLF Daily Briefing System

Automated daily security briefings with scheduling, voice narration,
and export capabilities.
"""

import schedule
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from pathlib import Path
import json


class DailyBriefingSystem:
    """
    Automated daily briefing system.
    
    Generates and delivers daily security briefings on schedule.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize daily briefing system.
        
        Args:
            config: Configuration with:
                - schedule_time: Time to generate briefing (HH:MM)
                - output_dir: Directory for briefing files
                - enable_voice: Enable voice narration
                - enable_email: Enable email delivery
                - email_recipients: List of email addresses
        """
        self.config = config
        self.schedule_time = config.get('schedule_time', '08:00')
        self.output_dir = Path(config.get('output_dir', 'briefings'))
        self.enable_voice = config.get('enable_voice', False)
        self.enable_email = config.get('enable_email', False)
        
        # Components
        self.explanation_generator = None
        self.voice_interface = None
        self.bridge = None
        
        # State
        self.running = False
        self.scheduler_thread = None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[Daily Briefing] Initialized - scheduled for {self.schedule_time}")
    
    def set_components(
        self,
        explanation_generator,
        voice_interface=None,
        bridge=None
    ):
        """Set required components."""
        self.explanation_generator = explanation_generator
        self.voice_interface = voice_interface
        self.bridge = bridge
    
    def start(self):
        """Start scheduled briefing generation."""
        if self.running:
            return
        
        self.running = True
        
        # Schedule daily briefing
        schedule.every().day.at(self.schedule_time).do(self.generate_and_deliver_briefing)
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        print(f"[Daily Briefing] Started - next briefing at {self.schedule_time}")
    
    def stop(self):
        """Stop scheduled briefing generation."""
        self.running = False
        schedule.clear()
        print("[Daily Briefing] Stopped")
    
    def generate_and_deliver_briefing(self):
        """Generate and deliver daily briefing."""
        print(f"[Daily Briefing] Generating briefing for {datetime.now().date()}")
        
        try:
            # Get data from bridge
            if self.bridge:
                snapshot = self.bridge.get_system_snapshot()
                threats = self.bridge.get_recent_threats(limit=100)
                metrics = self.bridge.get_threat_metrics()
            else:
                # Fallback data
                snapshot = {}
                threats = []
                metrics = {}
            
            # Generate briefing
            if self.explanation_generator:
                briefing_text = self.explanation_generator.generate_daily_briefing(
                    date=datetime.now(),
                    threats=threats,
                    system_metrics=metrics
                )
            else:
                briefing_text = self._generate_simple_briefing(threats, metrics)
            
            # Save to file
            filename = self._save_briefing(briefing_text)
            print(f"[Daily Briefing] Saved to {filename}")
            
            # Voice narration
            if self.enable_voice and self.voice_interface:
                self._narrate_briefing(briefing_text)
            
            # Email delivery
            if self.enable_email:
                self._email_briefing(briefing_text)
            
            print("[Daily Briefing] Delivery complete")
            
        except Exception as e:
            print(f"[Daily Briefing] Error: {e}")
    
    def generate_on_demand(self) -> str:
        """Generate briefing on demand."""
        print("[Daily Briefing] Generating on-demand briefing")
        
        # Get current data
        if self.bridge:
            threats = self.bridge.get_recent_threats(limit=100)
            metrics = self.bridge.get_threat_metrics()
        else:
            threats = []
            metrics = {}
        
        # Generate
        if self.explanation_generator:
            return self.explanation_generator.generate_daily_briefing(
                date=datetime.now(),
                threats=threats,
                system_metrics=metrics
            )
        else:
            return self._generate_simple_briefing(threats, metrics)
    
    def _scheduler_loop(self):
        """Scheduler loop."""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _save_briefing(self, briefing_text: str) -> str:
        """Save briefing to file."""
        filename = f"briefing_{datetime.now().strftime('%Y%m%d')}.md"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(briefing_text)
        
        return str(filepath)
    
    def _narrate_briefing(self, briefing_text: str):
        """Narrate briefing with voice."""
        if not self.voice_interface:
            return
        
        # Extract key points for narration
        lines = briefing_text.split('\n')
        narration = []
        
        for line in lines:
            if line.startswith('#') or line.startswith('**'):
                # Skip headers
                continue
            if line.strip() and not line.startswith('-'):
                narration.append(line.strip())
        
        # Speak summary
        summary = ' '.join(narration[:5])  # First 5 sentences
        self.voice_interface.speak(summary)
    
    def _email_briefing(self, briefing_text: str):
        """Email briefing to recipients."""
        # TODO: Implement email delivery
        print("[Daily Briefing] Email delivery not yet implemented")
    
    def _generate_simple_briefing(
        self,
        threats: List[Dict],
        metrics: Dict
    ) -> str:
        """Generate simple briefing without explanation generator."""
        briefing = f"# DIREWOLF Daily Briefing - {datetime.now().strftime('%B %d, %Y')}\n\n"
        briefing += f"## Summary\n\n"
        briefing += f"Threats detected: {len(threats)}\n"
        briefing += f"System health: {metrics.get('health_status', 'UNKNOWN')}\n\n"
        briefing += "## Status\n\nAll systems operational.\n"
        return briefing


# Example usage
if __name__ == "__main__":
    config = {
        'schedule_time': '08:00',
        'output_dir': 'briefings',
        'enable_voice': False,
        'enable_email': False
    }
    
    system = DailyBriefingSystem(config)
    system.start()
    
    # Generate on demand
    briefing = system.generate_on_demand()
    print(briefing)
