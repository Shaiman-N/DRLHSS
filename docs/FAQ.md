# DIREWOLF Frequently Asked Questions (FAQ)

## Your Questions Answered

**Version**: 1.0.0  
**Last Updated**: 2024

---

## Table of Contents

1. [General Questions](#general-questions)
2. [Installation & Setup](#installation--setup)
3. [Features & Functionality](#features--functionality)
4. [Security & Privacy](#security--privacy)
5. [Performance & Resources](#performance--resources)
6. [Voice & AI](#voice--ai)
7. [Troubleshooting](#troubleshooting)
8. [Licensing & Support](#licensing--support)

---

## General Questions

### What is DIREWOLF?

DIREWOLF is an AI-powered security system that combines Deep Reinforcement Learning with Explainable AI to provide intelligent, transparent cybersecurity protection. It acts as a loyal digital guardian that never takes actions without your explicit permission.

### Why is it called DIREWOLF?

The name is inspired by the loyal direwolves from Game of Thrones. Like those creatures, DIREWOLF is fiercely loyal, protective, and always acts in your best interest while respecting your authority.

### What makes DIREWOLF different from other security software?

**Key Differentiators**:
1. **Permission-Based**: Never acts autonomously
2. **Explainable AI**: Clear explanations for all recommendations
3. **Adaptive Learning**: Learns from your decisions
4. **Multi-Modal**: Text, voice, and visual interfaces
5. **Integrated**: Coordinates multiple security systems

### Will DIREWOLF take actions without my permission?

**Never.** This is a core principle. DIREWOLF will always request permission before taking any action. Your authority is absolute.

### Can I use DIREWOLF offline?

Yes, with local LLM configuration. Some features like cloud-based TTS may require internet connectivity, but core security functions work offline.

### Is DIREWOLF open source?

DIREWOLF is released under the MIT License. Source code is available at https://github.com/direwolf/DRLHSS

---

## Installation & Setup

### What are the system requirements?

**Minimum**:
- 4-core CPU, 8GB RAM, 10GB storage
- Windows 10+, Ubuntu 20.04+, or macOS 12+
- OpenGL 3.3 compatible GPU

**Recommended**:
- 8-core CPU, 16GB RAM, 20GB SSD storage
- Dedicated GPU with 2GB VRAM
- 1 Gbps network connection

### How long does installation take?

- Installer method: 5-10 minutes
- Build from source: 20-30 minutes
- Initial setup: 5 minutes

### Do I need to uninstall my current antivirus?

No, DIREWOLF can work alongside existing security software. However, you may need to add DIREWOLF to your antivirus exclusion list.

### Can I install DIREWOLF on multiple computers?

Yes, DIREWOLF can be installed on as many computers as needed. Enterprise licenses offer centralized management.

### Does DIREWOLF work on virtual machines?

Yes, DIREWOLF works on VMs. Some features like hardware-based sandboxing may have limited functionality depending on the hypervisor.

---

## Features & Functionality

### How does the permission system work?

1. DIREWOLF detects a threat
2. Analyzes and recommends action
3. Requests your permission
4. You approve, deny, or request more info
5. If approved, DIREWOLF executes action
6. System learns from your decision

### What happens if I don't respond to a permission request?

For non-critical threats, the request remains pending. For critical emergencies, there's a 30-second timeout with a safe default action (usually isolation).

### Can DIREWOLF explain its recommendations in simple terms?

Yes! DIREWOLF adapts explanations to your expertise level (novice, intermediate, expert). You can change this in Settings → User Profile.

### How does DIREWOLF learn from my decisions?

DIREWOLF uses reinforcement learning to adjust its recommendations based on your approval/denial patterns while maintaining security standards.

### Can I automate certain actions?

Yes, you can configure auto-approval for specific low-risk actions in Settings → Automation. However, high-risk actions always require explicit approval.

### What is Investigation Mode?

Investigation Mode provides deep-dive analysis of security incidents, including timeline reconstruction, attack chain analysis, and detailed forensics.

### How do Daily Briefings work?

DIREWOLF automatically generates a security briefing every morning (configurable time) with:
- Executive summary
- Threat overview
- Network activity
- System health
- Recommendations

You can also generate briefings on-demand.

### Can I share briefings with my team?

Yes, briefings can be exported as:
- PDF reports
- Video presentations
- JSON data
- Email summaries

---

## Security & Privacy

### Is my data sent to external servers?

Only if you choose cloud-based services (TTS, LLM). Local options are available for complete privacy. All data transmission uses TLS 1.3 encryption.

### How secure is DIREWOLF itself?

DIREWOLF undergoes regular security audits, uses encrypted communications, and follows security best practices. Current status: 0 critical vulnerabilities.

### What data does DIREWOLF collect?

**Collected Locally**:
- Network traffic metadata
- Threat detection events
- User decisions and preferences
- System performance metrics

**Never Collected**:
- Personal communications content
- Passwords or credentials
- Financial information
- Browsing history (unless threat-related)

### Can DIREWOLF detect zero-day attacks?

Yes, through behavioral analysis and machine learning. However, no system is 100% effective against all zero-days.

### How does DIREWOLF handle false positives?

You can deny actions you believe are false positives. DIREWOLF learns from these decisions to improve accuracy over time.

### Is DIREWOLF compliant with regulations?

DIREWOLF supports compliance with:
- GDPR (data privacy)
- HIPAA (healthcare)
- PCI DSS (payment cards)
- SOC 2 (security controls)
- ISO 27001 (information security)

### Can DIREWOLF be hacked?

While no system is completely unhackable, DIREWOLF implements multiple security layers:
- Encrypted communications
- Secure boot verification
- Code signing
- Regular security updates
- Audit logging

---

## Performance & Resources

### How much CPU/RAM does DIREWOLF use?

**Typical Usage**:
- CPU: 4-5% (idle), 12-15% (active scan)
- RAM: 480MB (idle), 920MB (active scan)
- Disk I/O: Low (except during scans)
- Network: <1 Mbps monitoring

### Will DIREWOLF slow down my computer?

No, DIREWOLF is optimized for minimal performance impact. Most users won't notice any slowdown during normal operation.

### How much network bandwidth does DIREWOLF use?

Typically <1 Mbps for monitoring. Higher during:
- Software updates
- Cloud service usage
- Video generation

### Can I adjust resource usage?

Yes, in Settings → Performance:
- Scan priority (low/normal/high)
- Visualization quality
- Update frequency
- Cache size

### How much disk space does DIREWOLF need?

- Installation: 2-3 GB
- Database: 100MB - 1GB (grows over time)
- Logs: 50-200MB (auto-rotated)
- Videos: Variable (user-generated)

**Total**: 10-20GB recommended

---

## Voice & AI

### What voice commands does DIREWOLF understand?

DIREWOLF understands natural language. Examples:
- "Show me network status"
- "What threats have you detected?"
- "Generate daily briefing"
- "Investigate incident INC_001"

See User Manual for complete command list.

### Can I change the wake word?

Yes, in Settings → Voice → Wake Word. Default is "Hey Wolf" but you can set any phrase.

### What languages does DIREWOLF support?

Currently:
- English (complete)
- Spanish, French, German, Japanese, Chinese (framework ready, translations in progress)

### Can I change Wolf's voice?

Yes, multiple TTS voices available:
- Guy (deep, authoritative)
- Aria (female, professional)
- Davis (male, friendly)

Adjust in Settings → Voice → TTS Voice.

### Does DIREWOLF require internet for voice?

- **Speech Recognition**: Can work offline with local Whisper model
- **TTS**: Cloud providers require internet, local TTS works offline
- **LLM**: Can use local Ollama (offline) or cloud services

### How accurate is voice recognition?

Whisper achieves 95%+ accuracy in quiet environments. Accuracy decreases with:
- Background noise
- Accents (improving)
- Technical jargon (learning)

### Can multiple users talk to DIREWOLF?

Yes, DIREWOLF can recognize different users and maintain separate preference profiles (enterprise feature).

---

## Troubleshooting

### DIREWOLF won't start

**Solutions**:
1. Check system requirements
2. Run as administrator/root
3. Check antivirus exclusions
4. Verify installation integrity
5. Check logs: `%APPDATA%/DIREWOLF/logs/`

### Voice commands not working

**Solutions**:
1. Check microphone permissions
2. Test audio: Settings → Voice → Test
3. Verify wake word setting
4. Check Whisper model installation
5. Restart audio service

### High CPU usage

**Solutions**:
1. Check for active scans
2. Reduce visualization quality
3. Disable unnecessary features
4. Update to latest version
5. Check for malware (ironically!)

### Network devices not detected

**Solutions**:
1. Check network permissions
2. Verify firewall settings
3. Run manual discovery: `direwolf --discover-network`
4. Check subnet configuration
5. Restart network service

### Permission dialogs not appearing

**Solutions**:
1. Check notification settings
2. Verify UI is running
3. Check system tray icon
4. Review logs for errors
5. Restart DIREWOLF service

### Database errors

**Solutions**:
1. Check disk space
2. Verify database permissions
3. Run integrity check: `direwolf --check-db`
4. Restore from backup
5. Reinitialize database (last resort)

---

## Licensing & Support

### Is DIREWOLF free?

DIREWOLF is open source (MIT License). Free for personal and commercial use.

### What support options are available?

**Community Support** (Free):
- Documentation
- Community forum
- Discord server
- GitHub issues

**Professional Support** (Paid):
- Email support
- Priority bug fixes
- Custom features
- Training sessions

**Enterprise Support** (Paid):
- 24/7 support
- Dedicated account manager
- SLA guarantees
- On-site assistance

### How do I report a bug?

1. Check if it's a known issue: https://github.com/direwolf/DRLHSS/issues
2. Gather information:
   - DIREWOLF version
   - Operating system
   - Steps to reproduce
   - Log files
3. Submit issue on GitHub or email support@direwolf.ai

### How do I request a feature?

1. Check existing feature requests
2. Submit on GitHub: https://github.com/direwolf/DRLHSS/issues
3. Or email: features@direwolf.ai
4. Include use case and benefits

### How often is DIREWOLF updated?

- **Security updates**: As needed (immediate for critical)
- **Bug fixes**: Weekly
- **Feature updates**: Monthly
- **Major releases**: Quarterly

### Can I contribute to DIREWOLF?

Yes! We welcome contributions:
- Code contributions
- Documentation improvements
- Bug reports
- Feature suggestions
- Community support

See CONTRIBUTING.md for guidelines.

### Is training available?

Yes:
- **Free**: Video tutorials, documentation
- **Paid**: Live training sessions, workshops
- **Enterprise**: Custom training programs

Contact training@direwolf.ai

### What's the roadmap for DIREWOLF?

**Upcoming Features**:
- Phase 10: Advanced AI (Q2 2024)
- Phase 11: Enterprise Features (Q3 2024)
- Phase 12: Cloud Integration (Q4 2024)
- Phase 13: Mobile & IoT (Q1 2025)

See ROADMAP.md for details.

---

## Still Have Questions?

### Documentation
- User Manual: `docs/USER_MANUAL.md`
- Installation Guide: `docs/INSTALLATION_GUIDE.md`
- Troubleshooting: `docs/TROUBLESHOOTING.md`

### Community
- Forum: https://community.direwolf.ai
- Discord: https://discord.gg/direwolf
- Reddit: r/direwolf

### Support
- Email: support@direwolf.ai
- Enterprise: enterprise@direwolf.ai
- Security: security@direwolf.ai

### Ask Wolf
```
"Hey Wolf, I have a question about [topic]"
```

Wolf can answer many questions directly!

---

**DIREWOLF FAQ v1.0.0**  
*Your Intelligent Security Guardian*
