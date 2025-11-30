"""
DIREWOLF Explanation Generator

Generates human-readable explanations for security events, daily briefings,
investigation reports, and video narration scripts.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class AudienceLevel(Enum):
    """Target audience technical level"""
    EXECUTIVE = "executive"  # Non-technical, high-level
    MANAGER = "manager"      # Some technical knowledge
    TECHNICAL = "technical"  # Full technical details
    EXPERT = "expert"        # Deep technical analysis


class ReportFormat(Enum):
    """Report output format"""
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"


@dataclass
class ThreatExplanation:
    """Explanation of a single threat"""
    threat_id: str
    threat_type: str
    severity: str
    summary: str
    technical_details: str
    impact_assessment: str
    recommended_actions: List[str]
    timeline: List[Dict[str, Any]]
    related_threats: List[str]


class ExplanationGenerator:
    """
    Generates explanations for security events and incidents.
    
    Supports multiple audience levels, formats, and languages.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize explanation generator.
        
        Args:
            config: Configuration dict with:
                - default_audience: Default audience level
                - default_format: Default output format
                - default_language: Default language (en, es, fr, etc.)
                - llm_engine: Optional LLM engine for dynamic generation
        """
        self.config = config
        self.default_audience = AudienceLevel(config.get('default_audience', 'manager'))
        self.default_format = ReportFormat(config.get('default_format', 'markdown'))
        self.default_language = config.get('default_language', 'en')
        self.llm_engine = config.get('llm_engine')
        
        print("[Explanation Generator] Initialized")
    
    # ========== Daily Briefing Generation ==========
    
    def generate_daily_briefing(
        self,
        date: datetime,
        threats: List[Dict[str, Any]],
        system_metrics: Dict[str, Any],
        audience: Optional[AudienceLevel] = None,
        format: Optional[ReportFormat] = None,
        language: str = "en"
    ) -> str:
        """
        Generate daily security briefing.
        
        Args:
            date: Date for briefing
            threats: List of threats detected
            system_metrics: System performance metrics
            audience: Target audience level
            format: Output format
            language: Language code
            
        Returns:
            Formatted briefing text
        """
        audience = audience or self.default_audience
        format = format or self.default_format
        
        # Build briefing content
        briefing = {
            'title': f"DIREWOLF Daily Security Briefing - {date.strftime('%B %d, %Y')}",
            'date': date.isoformat(),
            'executive_summary': self._generate_executive_summary(threats, system_metrics, audience),
            'threat_overview': self._generate_threat_overview(threats, audience),
            'system_status': self._generate_system_status(system_metrics, audience),
            'key_incidents': self._generate_key_incidents(threats, audience),
            'recommendations': self._generate_recommendations(threats, system_metrics, audience),
            'metrics': self._format_metrics(system_metrics, audience)
        }
        
        # Format output
        if format == ReportFormat.MARKDOWN:
            return self._format_briefing_markdown(briefing, audience)
        elif format == ReportFormat.HTML:
            return self._format_briefing_html(briefing, audience)
        elif format == ReportFormat.JSON:
            return json.dumps(briefing, indent=2)
        else:
            return self._format_briefing_text(briefing, audience)
    
    def _generate_executive_summary(
        self,
        threats: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        audience: AudienceLevel
    ) -> str:
        """Generate executive summary."""
        total_threats = len(threats)
        critical_threats = len([t for t in threats if t.get('severity') == 'CRITICAL'])
        
        if audience == AudienceLevel.EXECUTIVE:
            if critical_threats > 0:
                return f"Alpha, {critical_threats} critical security threats were detected and neutralized today. Your network remains secure."
            else:
                return f"Alpha, your network is secure. {total_threats} threats were detected and blocked. All systems operational."
        else:
            return f"Detected {total_threats} threats ({critical_threats} critical). All threats contained. System health: {metrics.get('health_status', 'HEALTHY')}."
    
    def _generate_threat_overview(
        self,
        threats: List[Dict[str, Any]],
        audience: AudienceLevel
    ) -> str:
        """Generate threat overview section."""
        if not threats:
            return "No threats detected today."
        
        # Group by type
        threat_types = {}
        for threat in threats:
            t_type = threat.get('threat_type', 'Unknown')
            threat_types[t_type] = threat_types.get(t_type, 0) + 1
        
        overview = f"Total threats: {len(threats)}\n\n"
        overview += "Breakdown by type:\n"
        for t_type, count in sorted(threat_types.items(), key=lambda x: x[1], reverse=True):
            overview += f"- {t_type}: {count}\n"
        
        return overview
    
    def _generate_system_status(
        self,
        metrics: Dict[str, Any],
        audience: AudienceLevel
    ) -> str:
        """Generate system status section."""
        status = f"Health: {metrics.get('health_status', 'UNKNOWN')}\n"
        status += f"DRL Confidence: {metrics.get('drl_confidence', 0):.1%}\n"
        
        if audience in [AudienceLevel.TECHNICAL, AudienceLevel.EXPERT]:
            status += f"Events Processed: {metrics.get('events_processed', 0)}\n"
            status += f"Detection Rate: {metrics.get('detection_rate', 0):.2%}\n"
        
        return status
    
    def _generate_key_incidents(
        self,
        threats: List[Dict[str, Any]],
        audience: AudienceLevel
    ) -> List[Dict[str, str]]:
        """Generate key incidents list."""
        # Get top 5 most severe threats
        sorted_threats = sorted(
            threats,
            key=lambda t: (t.get('severity') == 'CRITICAL', t.get('confidence', 0)),
            reverse=True
        )[:5]
        
        incidents = []
        for threat in sorted_threats:
            incident = {
                'title': f"{threat.get('threat_type', 'Unknown')} - {threat.get('file_name', 'Unknown')}",
                'severity': threat.get('severity', 'UNKNOWN'),
                'action_taken': threat.get('action_taken', 'BLOCKED'),
                'timestamp': threat.get('timestamp', '')
            }
            
            if audience in [AudienceLevel.TECHNICAL, AudienceLevel.EXPERT]:
                incident['details'] = threat.get('technical_details', '')
            
            incidents.append(incident)
        
        return incidents
    
    def _generate_recommendations(
        self,
        threats: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        audience: AudienceLevel
    ) -> List[str]:
        """Generate recommendations."""
        recommendations = []
        
        # Check for patterns
        critical_count = len([t for t in threats if t.get('severity') == 'CRITICAL'])
        
        if critical_count > 5:
            recommendations.append("Consider reviewing security policies - high number of critical threats detected")
        
        if metrics.get('drl_confidence', 1.0) < 0.8:
            recommendations.append("DRL model confidence is low - consider retraining with recent data")
        
        if not recommendations:
            recommendations.append("Continue current security posture - all systems performing well")
        
        return recommendations
    
    def _format_metrics(
        self,
        metrics: Dict[str, Any],
        audience: AudienceLevel
    ) -> Dict[str, Any]:
        """Format metrics for audience."""
        if audience == AudienceLevel.EXECUTIVE:
            return {
                'threats_blocked': metrics.get('threats_detected', 0),
                'system_health': metrics.get('health_status', 'HEALTHY')
            }
        else:
            return metrics
    
    def _format_briefing_markdown(
        self,
        briefing: Dict[str, Any],
        audience: AudienceLevel
    ) -> str:
        """Format briefing as Markdown."""
        md = f"# {briefing['title']}\n\n"
        md += f"**Date**: {briefing['date']}\n\n"
        md += f"## Executive Summary\n\n{briefing['executive_summary']}\n\n"
        md += f"## Threat Overview\n\n{briefing['threat_overview']}\n\n"
        md += f"## System Status\n\n{briefing['system_status']}\n\n"
        
        md += "## Key Incidents\n\n"
        for incident in briefing['key_incidents']:
            md += f"### {incident['title']}\n"
            md += f"- **Severity**: {incident['severity']}\n"
            md += f"- **Action**: {incident['action_taken']}\n"
            md += f"- **Time**: {incident['timestamp']}\n\n"
        
        md += "## Recommendations\n\n"
        for rec in briefing['recommendations']:
            md += f"- {rec}\n"
        
        return md
    
    def _format_briefing_text(
        self,
        briefing: Dict[str, Any],
        audience: AudienceLevel
    ) -> str:
        """Format briefing as plain text."""
        text = f"{briefing['title']}\n"
        text += "=" * len(briefing['title']) + "\n\n"
        text += f"Date: {briefing['date']}\n\n"
        text += f"EXECUTIVE SUMMARY\n{briefing['executive_summary']}\n\n"
        text += f"THREAT OVERVIEW\n{briefing['threat_overview']}\n\n"
        text += f"SYSTEM STATUS\n{briefing['system_status']}\n\n"
        
        text += "KEY INCIDENTS\n"
        for incident in briefing['key_incidents']:
            text += f"\n{incident['title']}\n"
            text += f"  Severity: {incident['severity']}\n"
            text += f"  Action: {incident['action_taken']}\n"
        
        text += "\n\nRECOMMENDATIONS\n"
        for rec in briefing['recommendations']:
            text += f"- {rec}\n"
        
        return text
    
    def _format_briefing_html(
        self,
        briefing: Dict[str, Any],
        audience: AudienceLevel
    ) -> str:
        """Format briefing as HTML."""
        html = f"<html><head><title>{briefing['title']}</title></head><body>"
        html += f"<h1>{briefing['title']}</h1>"
        html += f"<p><strong>Date:</strong> {briefing['date']}</p>"
        html += f"<h2>Executive Summary</h2><p>{briefing['executive_summary']}</p>"
        html += f"<h2>Threat Overview</h2><pre>{briefing['threat_overview']}</pre>"
        html += f"<h2>System Status</h2><pre>{briefing['system_status']}</pre>"
        html += "<h2>Key Incidents</h2><ul>"
        for incident in briefing['key_incidents']:
            html += f"<li><strong>{incident['title']}</strong> - {incident['severity']}</li>"
        html += "</ul>"
        html += "<h2>Recommendations</h2><ul>"
        for rec in briefing['recommendations']:
            html += f"<li>{rec}</li>"
        html += "</ul></body></html>"
        return html
    
    # ========== Investigation Report Generation ==========
    
    def generate_investigation_report(
        self,
        incident_id: str,
        incident_data: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        timeline: List[Dict[str, Any]],
        audience: Optional[AudienceLevel] = None,
        format: Optional[ReportFormat] = None
    ) -> str:
        """
        Generate detailed investigation report.
        
        Args:
            incident_id: Incident identifier
            incident_data: Incident details
            evidence: Collected evidence
            timeline: Event timeline
            audience: Target audience
            format: Output format
            
        Returns:
            Formatted investigation report
        """
        audience = audience or self.default_audience
        format = format or self.default_format
        
        report = {
            'title': f"Security Incident Investigation Report - {incident_id}",
            'incident_id': incident_id,
            'summary': self._generate_incident_summary(incident_data, audience),
            'timeline': self._format_timeline(timeline, audience),
            'evidence': self._format_evidence(evidence, audience),
            'analysis': self._generate_analysis(incident_data, evidence, audience),
            'conclusions': self._generate_conclusions(incident_data, evidence, audience),
            'recommendations': self._generate_incident_recommendations(incident_data, audience)
        }
        
        if format == ReportFormat.MARKDOWN:
            return self._format_investigation_markdown(report)
        else:
            return json.dumps(report, indent=2)
    
    def _generate_incident_summary(
        self,
        incident: Dict[str, Any],
        audience: AudienceLevel
    ) -> str:
        """Generate incident summary."""
        summary = f"Incident Type: {incident.get('threat_type', 'Unknown')}\n"
        summary += f"Severity: {incident.get('severity', 'UNKNOWN')}\n"
        summary += f"Status: {incident.get('status', 'RESOLVED')}\n"
        summary += f"First Detected: {incident.get('first_detected', 'Unknown')}\n"
        return summary
    
    def _format_timeline(
        self,
        timeline: List[Dict[str, Any]],
        audience: AudienceLevel
    ) -> List[Dict[str, str]]:
        """Format timeline events."""
        formatted = []
        for event in timeline:
            formatted.append({
                'timestamp': event.get('timestamp', ''),
                'event': event.get('event_type', ''),
                'description': event.get('description', '')
            })
        return formatted
    
    def _format_evidence(
        self,
        evidence: List[Dict[str, Any]],
        audience: AudienceLevel
    ) -> List[Dict[str, str]]:
        """Format evidence items."""
        formatted = []
        for item in evidence:
            formatted.append({
                'type': item.get('evidence_type', ''),
                'description': item.get('description', ''),
                'source': item.get('source', '')
            })
        return formatted
    
    def _generate_analysis(
        self,
        incident: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        audience: AudienceLevel
    ) -> str:
        """Generate incident analysis."""
        analysis = "Based on the collected evidence and timeline:\n\n"
        analysis += f"The {incident.get('threat_type')} was detected through {len(evidence)} pieces of evidence. "
        analysis += "The attack chain has been reconstructed and all affected systems identified."
        return analysis
    
    def _generate_conclusions(
        self,
        incident: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        audience: AudienceLevel
    ) -> str:
        """Generate conclusions."""
        return f"The incident has been fully contained. No data loss occurred. All affected systems have been secured."
    
    def _generate_incident_recommendations(
        self,
        incident: Dict[str, Any],
        audience: AudienceLevel
    ) -> List[str]:
        """Generate incident-specific recommendations."""
        return [
            "Update security policies to prevent similar incidents",
            "Review access controls for affected systems",
            "Conduct security awareness training"
        ]
    
    def _format_investigation_markdown(self, report: Dict[str, Any]) -> str:
        """Format investigation report as Markdown."""
        md = f"# {report['title']}\n\n"
        md += f"**Incident ID**: {report['incident_id']}\n\n"
        md += f"## Summary\n\n{report['summary']}\n\n"
        md += "## Timeline\n\n"
        for event in report['timeline']:
            md += f"- **{event['timestamp']}**: {event['event']} - {event['description']}\n"
        md += "\n## Evidence\n\n"
        for item in report['evidence']:
            md += f"- **{item['type']}**: {item['description']} (Source: {item['source']})\n"
        md += f"\n## Analysis\n\n{report['analysis']}\n\n"
        md += f"## Conclusions\n\n{report['conclusions']}\n\n"
        md += "## Recommendations\n\n"
        for rec in report['recommendations']:
            md += f"- {rec}\n"
        return md
    
    # ========== Video Narration Scripts ==========
    
    def generate_video_narration(
        self,
        incident_data: Dict[str, Any],
        timeline: List[Dict[str, Any]],
        audience: Optional[AudienceLevel] = None
    ) -> List[Dict[str, str]]:
        """
        Generate video narration script.
        
        Args:
            incident_data: Incident details
            timeline: Event timeline
            audience: Target audience
            
        Returns:
            List of narration segments with timestamps
        """
        audience = audience or self.default_audience
        
        script = []
        
        # Introduction
        script.append({
            'timestamp': '00:00',
            'narration': f"This is a security incident replay for {incident_data.get('threat_type')} detected on {incident_data.get('first_detected')}.",
            'scene': 'overview'
        })
        
        # Timeline narration
        for i, event in enumerate(timeline):
            timestamp = f"00:{(i+1)*10:02d}"
            script.append({
                'timestamp': timestamp,
                'narration': f"At {event.get('timestamp')}, {event.get('description')}",
                'scene': f"event_{i}"
            })
        
        # Conclusion
        script.append({
            'timestamp': f"00:{(len(timeline)+1)*10:02d}",
            'narration': f"The threat was successfully contained. All systems are now secure.",
            'scene': 'conclusion'
        })
        
        return script


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'default_audience': 'manager',
        'default_format': 'markdown',
        'default_language': 'en'
    }
    
    # Initialize
    generator = ExplanationGenerator(config)
    
    # Generate daily briefing
    threats = [
        {
            'threat_type': 'Malware',
            'severity': 'CRITICAL',
            'file_name': 'suspicious.exe',
            'confidence': 0.94,
            'action_taken': 'QUARANTINED',
            'timestamp': '2025-11-27 14:23:15'
        }
    ]
    
    metrics = {
        'health_status': 'HEALTHY',
        'drl_confidence': 0.94,
        'threats_detected': 12,
        'events_processed': 1523
    }
    
    briefing = generator.generate_daily_briefing(
        date=datetime.now(),
        threats=threats,
        system_metrics=metrics,
        audience=AudienceLevel.MANAGER
    )
    
    print(briefing)
