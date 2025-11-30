"""
DIREWOLF Investigation Mode

Deep-dive incident investigation with forensic timeline, evidence collection,
and interactive Q&A.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import json


@dataclass
class Evidence:
    """Evidence item"""
    evidence_id: str
    evidence_type: str  # file, network, process, registry
    description: str
    source: str
    timestamp: str
    hash: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class TimelineEvent:
    """Timeline event"""
    timestamp: str
    event_type: str
    description: str
    source: str
    severity: str
    related_evidence: List[str]


class InvestigationMode:
    """
    Interactive investigation mode for deep-dive incident analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize investigation mode.
        
        Args:
            config: Configuration dict
        """
        self.config = config
        self.bridge = None
        self.llm_engine = None
        
        # Active investigation
        self.current_investigation: Optional[Dict] = None
        self.evidence_items: List[Evidence] = []
        self.timeline_events: List[TimelineEvent] = []
        
        print("[Investigation Mode] Initialized")
    
    def set_components(self, bridge, llm_engine=None):
        """Set required components."""
        self.bridge = bridge
        self.llm_engine = llm_engine
    
    def start_investigation(
        self,
        incident_id: str,
        incident_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Start new investigation.
        
        Args:
            incident_id: Incident identifier
            incident_data: Initial incident data
            
        Returns:
            Investigation session info
        """
        self.current_investigation = {
            'investigation_id': f"INV_{incident_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'incident_id': incident_id,
            'start_time': datetime.now().isoformat(),
            'status': 'ACTIVE',
            'incident_data': incident_data
        }
        
        # Clear previous data
        self.evidence_items = []
        self.timeline_events = []
        
        # Collect initial evidence
        self._collect_initial_evidence(incident_data)
        
        # Build timeline
        self._build_forensic_timeline(incident_data)
        
        print(f"[Investigation] Started: {self.current_investigation['investigation_id']}")
        
        return {
            'investigation_id': self.current_investigation['investigation_id'],
            'incident_id': incident_id,
            'evidence_count': len(self.evidence_items),
            'timeline_events': len(self.timeline_events)
        }
    
    def collect_evidence(
        self,
        evidence_type: str,
        source: str
    ) -> List[Evidence]:
        """
        Collect evidence from specified source.
        
        Args:
            evidence_type: Type of evidence to collect
            source: Evidence source
            
        Returns:
            List of collected evidence
        """
        print(f"[Investigation] Collecting {evidence_type} evidence from {source}")
        
        collected = []
        
        # Simulate evidence collection
        if evidence_type == "file":
            evidence = Evidence(
                evidence_id=f"EV_{len(self.evidence_items)+1}",
                evidence_type="file",
                description=f"Suspicious file from {source}",
                source=source,
                timestamp=datetime.now().isoformat(),
                hash="abc123def456"
            )
            collected.append(evidence)
            self.evidence_items.append(evidence)
        
        elif evidence_type == "network":
            evidence = Evidence(
                evidence_id=f"EV_{len(self.evidence_items)+1}",
                evidence_type="network",
                description=f"Network connection to {source}",
                source=source,
                timestamp=datetime.now().isoformat()
            )
            collected.append(evidence)
            self.evidence_items.append(evidence)
        
        return collected
    
    def build_forensic_timeline(self) -> List[TimelineEvent]:
        """
        Build forensic timeline of incident.
        
        Returns:
            Ordered list of timeline events
        """
        if not self.current_investigation:
            return []
        
        # Sort events by timestamp
        sorted_events = sorted(
            self.timeline_events,
            key=lambda e: e.timestamp
        )
        
        return sorted_events
    
    def ask_question(self, question: str) -> str:
        """
        Interactive Q&A about the investigation.
        
        Args:
            question: Question about the incident
            
        Returns:
            Answer based on collected evidence
        """
        if not self.current_investigation:
            return "No active investigation. Start an investigation first."
        
        # Use LLM if available
        if self.llm_engine:
            # Build context from evidence and timeline
            context = self._build_investigation_context()
            
            # Generate answer
            answer = self.llm_engine.generate_response(
                user_input=question,
                context=context,
                system_state=self.current_investigation['incident_data']
            )
            
            return answer
        else:
            # Simple rule-based responses
            return self._simple_answer(question)
    
    def generate_investigation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive investigation report.
        
        Returns:
            Investigation report
        """
        if not self.current_investigation:
            return {}
        
        report = {
            'investigation_id': self.current_investigation['investigation_id'],
            'incident_id': self.current_investigation['incident_id'],
            'start_time': self.current_investigation['start_time'],
            'end_time': datetime.now().isoformat(),
            'status': 'COMPLETED',
            'summary': self._generate_investigation_summary(),
            'evidence_collected': len(self.evidence_items),
            'timeline_events': len(self.timeline_events),
            'evidence': [self._evidence_to_dict(e) for e in self.evidence_items],
            'timeline': [self._timeline_event_to_dict(e) for e in self.timeline_events],
            'conclusions': self._generate_conclusions(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def export_report(
        self,
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """
        Export investigation report.
        
        Args:
            format: Export format (json, markdown, html)
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        report = self.generate_investigation_report()
        
        if format == "json":
            content = json.dumps(report, indent=2)
            ext = ".json"
        elif format == "markdown":
            content = self._format_report_markdown(report)
            ext = ".md"
        else:
            content = json.dumps(report, indent=2)
            ext = ".json"
        
        if not output_path:
            output_path = f"investigation_{report['investigation_id']}{ext}"
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        print(f"[Investigation] Report exported to {output_path}")
        return output_path
    
    # ========== Private Methods ==========
    
    def _collect_initial_evidence(self, incident_data: Dict[str, Any]):
        """Collect initial evidence from incident data."""
        # File evidence
        if 'file_path' in incident_data:
            self.evidence_items.append(Evidence(
                evidence_id=f"EV_001",
                evidence_type="file",
                description=f"Malicious file: {incident_data['file_path']}",
                source="AV Scanner",
                timestamp=incident_data.get('timestamp', datetime.now().isoformat()),
                hash=incident_data.get('file_hash')
            ))
        
        # Network evidence
        if 'ip_address' in incident_data:
            self.evidence_items.append(Evidence(
                evidence_id=f"EV_002",
                evidence_type="network",
                description=f"Suspicious connection to {incident_data['ip_address']}",
                source="NIDPS",
                timestamp=incident_data.get('timestamp', datetime.now().isoformat())
            ))
    
    def _build_forensic_timeline(self, incident_data: Dict[str, Any]):
        """Build initial forensic timeline."""
        # Add detection event
        self.timeline_events.append(TimelineEvent(
            timestamp=incident_data.get('timestamp', datetime.now().isoformat()),
            event_type="DETECTION",
            description=f"Threat detected: {incident_data.get('threat_type')}",
            source="DIREWOLF",
            severity=incident_data.get('severity', 'UNKNOWN'),
            related_evidence=["EV_001"]
        ))
        
        # Add response event
        if 'action_taken' in incident_data:
            self.timeline_events.append(TimelineEvent(
                timestamp=datetime.now().isoformat(),
                event_type="RESPONSE",
                description=f"Action taken: {incident_data['action_taken']}",
                source="Action Executor",
                severity="INFO",
                related_evidence=[]
            ))
    
    def _build_investigation_context(self) -> Dict:
        """Build context for LLM."""
        return {
            'investigation_id': self.current_investigation['investigation_id'],
            'evidence_count': len(self.evidence_items),
            'timeline_events': len(self.timeline_events),
            'incident_data': self.current_investigation['incident_data']
        }
    
    def _simple_answer(self, question: str) -> str:
        """Simple rule-based answer."""
        q_lower = question.lower()
        
        if 'how many' in q_lower and 'evidence' in q_lower:
            return f"I've collected {len(self.evidence_items)} pieces of evidence so far."
        elif 'timeline' in q_lower:
            return f"The timeline contains {len(self.timeline_events)} events."
        elif 'what happened' in q_lower:
            return "Based on the evidence, a security threat was detected and contained."
        else:
            return "I can provide information about the evidence, timeline, and incident details."
    
    def _generate_investigation_summary(self) -> str:
        """Generate investigation summary."""
        return f"Investigation of incident {self.current_investigation['incident_id']} completed. " \
               f"Collected {len(self.evidence_items)} pieces of evidence and reconstructed timeline with " \
               f"{len(self.timeline_events)} events."
    
    def _generate_conclusions(self) -> List[str]:
        """Generate investigation conclusions."""
        return [
            "Threat was successfully detected and contained",
            "No data loss occurred",
            "All affected systems have been secured"
        ]
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations."""
        return [
            "Update security policies",
            "Review access controls",
            "Conduct security training"
        ]
    
    def _evidence_to_dict(self, evidence: Evidence) -> Dict:
        """Convert evidence to dict."""
        return {
            'evidence_id': evidence.evidence_id,
            'type': evidence.evidence_type,
            'description': evidence.description,
            'source': evidence.source,
            'timestamp': evidence.timestamp,
            'hash': evidence.hash
        }
    
    def _timeline_event_to_dict(self, event: TimelineEvent) -> Dict:
        """Convert timeline event to dict."""
        return {
            'timestamp': event.timestamp,
            'event_type': event.event_type,
            'description': event.description,
            'source': event.source,
            'severity': event.severity
        }
    
    def _format_report_markdown(self, report: Dict) -> str:
        """Format report as Markdown."""
        md = f"# Investigation Report - {report['investigation_id']}\n\n"
        md += f"**Incident ID**: {report['incident_id']}\n"
        md += f"**Start Time**: {report['start_time']}\n"
        md += f"**End Time**: {report['end_time']}\n\n"
        md += f"## Summary\n\n{report['summary']}\n\n"
        md += f"## Evidence ({report['evidence_collected']} items)\n\n"
        for evidence in report['evidence']:
            md += f"- **{evidence['evidence_id']}**: {evidence['description']}\n"
        md += f"\n## Timeline ({report['timeline_events']} events)\n\n"
        for event in report['timeline']:
            md += f"- **{event['timestamp']}**: {event['description']}\n"
        md += "\n## Conclusions\n\n"
        for conclusion in report['conclusions']:
            md += f"- {conclusion}\n"
        md += "\n## Recommendations\n\n"
        for rec in report['recommendations']:
            md += f"- {rec}\n"
        return md


# Example usage
if __name__ == "__main__":
    investigation = InvestigationMode({})
    
    # Start investigation
    incident_data = {
        'threat_type': 'Malware',
        'file_path': '/tmp/suspicious.exe',
        'severity': 'CRITICAL',
        'timestamp': datetime.now().isoformat()
    }
    
    session = investigation.start_investigation("INC_001", incident_data)
    print(f"Started: {session}")
    
    # Ask questions
    answer = investigation.ask_question("How many pieces of evidence were collected?")
    print(f"Answer: {answer}")
    
    # Generate report
    report = investigation.generate_investigation_report()
    print(json.dumps(report, indent=2))
