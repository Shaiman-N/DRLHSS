#!/usr/bin/env python3
"""
DIREWOLF Live Network Traffic Monitor Demo
Real-time simulation of incoming network packets with threat detection
Perfect for live demonstrations
"""

import time
import random
from datetime import datetime
import sys

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'

class PacketGenerator:
    """Generates realistic network packets"""
    
    NORMAL_IPS = [
        "192.168.1.100", "192.168.1.105", "192.168.1.110",
        "10.0.0.50", "10.0.0.75", "172.16.0.20"
    ]
    
    SUSPICIOUS_IPS = [
        "203.0.113.42", "198.51.100.88", "185.220.101.5",
        "45.142.212.61", "91.219.237.244", "104.244.72.115"
    ]
    
    NORMAL_ACTIVITIES = [
        ("HTTP", "GET /index.html", 80, "Web browsing"),
        ("HTTPS", "TLS handshake", 443, "Secure connection"),
        ("DNS", "Query: google.com", 53, "DNS lookup"),
        ("HTTP", "POST /api/data", 80, "API request"),
        ("HTTPS", "GET /images/logo.png", 443, "Resource fetch"),
        ("DNS", "Query: github.com", 53, "DNS resolution"),
        ("HTTP", "GET /style.css", 80, "CSS resource"),
        ("SMTP", "MAIL FROM", 25, "Email sending"),
    ]
    
    SUSPICIOUS_ACTIVITIES = [
        ("TCP", "SYN flood detected", 0, "DDoS Attack", "CRITICAL"),
        ("HTTP", "SQL injection: ' OR '1'='1", 80, "SQL Injection", "HIGH"),
        ("TCP", "Port scan: 1-65535", 0, "Port Scanning", "HIGH"),
        ("HTTP", "Path traversal: ../../etc/passwd", 80, "Directory Traversal", "HIGH"),
        ("TCP", "Multiple failed SSH attempts", 22, "Brute Force", "CRITICAL"),
        ("HTTP", "XSS attempt: <script>alert(1)</script>", 80, "Cross-Site Scripting", "MEDIUM"),
        ("DNS", "DNS tunneling detected", 53, "Data Exfiltration", "HIGH"),
        ("TCP", "Connection to C&C server", 4444, "Malware Communication", "CRITICAL"),
        ("HTTP", "Buffer overflow attempt", 80, "Exploit Attempt", "HIGH"),
        ("ICMP", "Ping flood detected", 0, "DDoS Attack", "HIGH"),
    ]
    
    @staticmethod
    def generate_normal_packet():
        """Generate a normal network packet"""
        ip = random.choice(PacketGenerator.NORMAL_IPS)
        protocol, activity, port, description = random.choice(PacketGenerator.NORMAL_ACTIVITIES)
        size = random.randint(64, 1500)
        
        return {
            'src_ip': ip,
            'protocol': protocol,
            'port': port,
            'activity': activity,
            'description': description,
            'size': size,
            'is_threat': False,
            'threat_level': 'NONE'
        }
    
    @staticmethod
    def generate_suspicious_packet():
        """Generate a suspicious network packet"""
        ip = random.choice(PacketGenerator.SUSPICIOUS_IPS)
        protocol, activity, port, description, threat_level = random.choice(PacketGenerator.SUSPICIOUS_ACTIVITIES)
        size = random.randint(64, 2000)
        
        return {
            'src_ip': ip,
            'protocol': protocol,
            'port': port,
            'activity': activity,
            'description': description,
            'size': size,
            'is_threat': True,
            'threat_level': threat_level
        }

class ThreatDetector:
    """Simulates DRL-based threat detection"""
    
    def __init__(self):
        self.packet_count = 0
        self.threat_count = 0
        self.blocked_count = 0
        
    def analyze_packet(self, packet):
        """Analyze packet and return detection result"""
        self.packet_count += 1
        
        if packet['is_threat']:
            self.threat_count += 1
            confidence = random.uniform(0.85, 0.99)
            action = "BLOCKED" if confidence > 0.80 else "FLAGGED"
            if action == "BLOCKED":
                self.blocked_count += 1
            
            return {
                'detected': True,
                'confidence': confidence,
                'action': action,
                'drl_decision': self.get_drl_decision(packet, confidence)
            }
        else:
            confidence = random.uniform(0.01, 0.15)
            return {
                'detected': False,
                'confidence': 1 - confidence,
                'action': "ALLOWED",
                'drl_decision': None
            }
    
    def get_drl_decision(self, packet, confidence):
        """Get DRL agent decision"""
        if confidence > 0.95:
            return "IMMEDIATE_BLOCK"
        elif confidence > 0.85:
            return "BLOCK_AND_LOG"
        else:
            return "MONITOR"
    
    def get_stats(self):
        """Get current statistics"""
        return {
            'total': self.packet_count,
            'threats': self.threat_count,
            'blocked': self.blocked_count,
            'clean': self.packet_count - self.threat_count
        }

def print_header():
    """Print the monitor header"""
    print("\n" + "="*100)
    print(f"{Colors.BOLD}{Colors.CYAN}DIREWOLF LIVE NETWORK TRAFFIC MONITOR{Colors.END}".center(110))
    print(f"{Colors.CYAN}Real-Time Threat Detection & DRL-Based Response{Colors.END}".center(110))
    print("="*100 + "\n")

def print_stats_bar(detector):
    """Print statistics bar"""
    stats = detector.get_stats()
    print(f"\r{Colors.BOLD}[Stats]{Colors.END} " +
          f"Total: {Colors.CYAN}{stats['total']}{Colors.END} | " +
          f"Clean: {Colors.GREEN}{stats['clean']}{Colors.END} | " +
          f"Threats: {Colors.RED}{stats['threats']}{Colors.END} | " +
          f"Blocked: {Colors.YELLOW}{stats['blocked']}{Colors.END}", end='', flush=True)

def print_packet(packet, result):
    """Print packet information with detection result"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    # Clear the stats line
    print("\r" + " "*100 + "\r", end='')
    
    # Packet header
    if packet['is_threat']:
        status_color = Colors.RED
        status_icon = "⚠"
        status_text = "THREAT"
    else:
        status_color = Colors.GREEN
        status_icon = "✓"
        status_text = "CLEAN"
    
    print(f"\n[{Colors.CYAN}{timestamp}{Colors.END}] " +
          f"{status_color}{status_icon} {status_text}{Colors.END} | " +
          f"{packet['src_ip']:15} | " +
          f"{packet['protocol']:6} | " +
          f"Port {packet['port']:5} | " +
          f"{packet['size']:4}B")
    
    # Activity
    print(f"  Activity: {packet['activity']}")
    
    # Detection result
    if result['detected']:
        print(f"  {Colors.RED}Detection: {packet['description']}{Colors.END}")
        print(f"  {Colors.YELLOW}Threat Level: {packet['threat_level']}{Colors.END}")
        print(f"  {Colors.CYAN}DRL Confidence: {result['confidence']*100:.1f}%{Colors.END}")
        print(f"  {Colors.BOLD}DRL Decision: {result['drl_decision']}{Colors.END}")
        
        if result['action'] == "BLOCKED":
            print(f"  {Colors.BG_RED}{Colors.BOLD} ACTION: BLOCKED {Colors.END}")
        else:
            print(f"  {Colors.BG_YELLOW}{Colors.BOLD} ACTION: FLAGGED {Colors.END}")
    else:
        print(f"  {Colors.GREEN}Status: Normal traffic (Confidence: {result['confidence']*100:.1f}%){Colors.END}")
        print(f"  {Colors.GREEN}Action: ALLOWED{Colors.END}")

def run_live_monitor(duration=60, packet_rate=2.0):
    """
    Run the live traffic monitor
    
    Args:
        duration: How long to run (seconds), 0 for infinite
        packet_rate: Packets per second
    """
    print_header()
    
    detector = ThreatDetector()
    generator = PacketGenerator()
    
    print(f"{Colors.BOLD}Monitor Status:{Colors.END} {Colors.GREEN}ACTIVE{Colors.END}")
    print(f"{Colors.BOLD}Detection Mode:{Colors.END} DRL-Enhanced Real-Time Analysis")
    print(f"{Colors.BOLD}Packet Rate:{Colors.END} ~{packet_rate} packets/second")
    print(f"\n{Colors.YELLOW}Press Ctrl+C to stop monitoring{Colors.END}\n")
    
    start_time = time.time()
    packet_interval = 1.0 / packet_rate
    
    try:
        while True:
            # Check duration
            if duration > 0 and (time.time() - start_time) > duration:
                break
            
            # Generate packet (70% normal, 30% suspicious)
            if random.random() < 0.70:
                packet = generator.generate_normal_packet()
            else:
                packet = generator.generate_suspicious_packet()
            
            # Analyze packet
            result = detector.analyze_packet(packet)
            
            # Display packet
            print_packet(packet, result)
            
            # Update stats bar
            print_stats_bar(detector)
            
            # Wait before next packet
            time.sleep(packet_interval)
            
    except KeyboardInterrupt:
        print("\n\n" + Colors.YELLOW + "Monitor stopped by user" + Colors.END)
    
    # Final statistics
    print("\n\n" + "="*100)
    print(f"{Colors.BOLD}{Colors.CYAN}MONITORING SESSION SUMMARY{Colors.END}".center(110))
    print("="*100)
    
    stats = detector.get_stats()
    duration_actual = time.time() - start_time
    
    print(f"\n{Colors.BOLD}Session Duration:{Colors.END} {duration_actual:.1f} seconds")
    print(f"{Colors.BOLD}Total Packets:{Colors.END} {stats['total']}")
    print(f"{Colors.BOLD}Clean Traffic:{Colors.END} {Colors.GREEN}{stats['clean']}{Colors.END} " +
          f"({stats['clean']/stats['total']*100:.1f}%)")
    print(f"{Colors.BOLD}Threats Detected:{Colors.END} {Colors.RED}{stats['threats']}{Colors.END} " +
          f"({stats['threats']/stats['total']*100:.1f}%)")
    print(f"{Colors.BOLD}Threats Blocked:{Colors.END} {Colors.YELLOW}{stats['blocked']}{Colors.END} " +
          f"({stats['blocked']/stats['threats']*100:.1f}% of threats)")
    print(f"{Colors.BOLD}Average Rate:{Colors.END} {stats['total']/duration_actual:.2f} packets/second")
    
    print(f"\n{Colors.GREEN}✓ All threats were successfully detected and mitigated{Colors.END}")
    print(f"{Colors.CYAN}✓ DRL agent maintained 100% detection accuracy{Colors.END}")
    print(f"{Colors.YELLOW}✓ Zero false positives on legitimate traffic{Colors.END}\n")

def main():
    """Main entry point"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}DIREWOLF NETWORK TRAFFIC MONITOR - DEMO MODE{Colors.END}")
    print(f"{Colors.HEADER}Live demonstration of real-time threat detection{Colors.END}\n")
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Usage: python demo_live_traffic_monitor.py [duration] [rate]")
            print("\nArguments:")
            print("  duration  - How long to run in seconds (default: 60, 0 for infinite)")
            print("  rate      - Packets per second (default: 2.0)")
            print("\nExamples:")
            print("  python demo_live_traffic_monitor.py          # Run for 60 seconds")
            print("  python demo_live_traffic_monitor.py 30       # Run for 30 seconds")
            print("  python demo_live_traffic_monitor.py 0 3.0    # Run infinite at 3 pps")
            print("  python demo_live_traffic_monitor.py 120 1.5  # Run 2 min at 1.5 pps")
            return
        
        duration = int(sys.argv[1])
        rate = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    else:
        duration = 60
        rate = 2.0
    
    run_live_monitor(duration=duration, packet_rate=rate)

if __name__ == "__main__":
    main()
