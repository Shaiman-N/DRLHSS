#!/usr/bin/env python3
"""
DIREWOLF Update Manifest Generator

Generates update manifests with cryptographic signatures.
"""

import json
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path
import subprocess

class ManifestGenerator:
    """Generate update manifests"""
    
    def __init__(self, version, channel, package_dir):
        self.version = version
        self.channel = channel
        self.package_dir = Path(package_dir)
        self.manifest = {
            "version": version,
            "channel": channel,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "updates": []
        }
    
    def calculate_checksum(self, file_path):
        """Calculate SHA256 checksum"""
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def get_file_size(self, file_path):
        """Get file size in bytes"""
        return os.path.getsize(file_path)
    
    def sign_file(self, file_path, private_key_path):
        """Sign file with private key"""
        signature_path = str(file_path) + ".sig"
        
        # Use OpenSSL to sign
        cmd = [
            'openssl', 'dgst', '-sha256',
            '-sign', private_key_path,
            '-out', signature_path,
            str(file_path)
        ]
        
        subprocess.run(cmd, check=True)
        
        return signature_path
    
    def add_update(self, package_file, release_notes, download_url_base,
                   requires_restart=True, is_critical=False, is_delta=False):
        """Add update to manifest"""
        
        file_path = self.package_dir / package_file
        
        if not file_path.exists():
            print(f"Warning: Package not found: {file_path}")
            return
        
        checksum = self.calculate_checksum(file_path)
        size = self.get_file_size(file_path)
        
        update_info = {
            "version": self.version,
            "channel": self.channel,
            "release_notes": release_notes,
            "download_url": f"{download_url_base}/{package_file}",
            "signature_url": f"{download_url_base}/{package_file}.sig",
            "checksum": checksum,
            "size_bytes": size,
            "release_date": datetime.utcnow().isoformat() + "Z",
            "requires_restart": requires_restart,
            "is_critical": is_critical,
            "is_delta": is_delta
        }
        
        self.manifest["updates"].append(update_info)
        
        print(f"Added update: {package_file}")
        print(f"  Size: {size:,} bytes")
        print(f"  Checksum: {checksum}")
    
    def generate(self, output_path, private_key_path=None):
        """Generate manifest file"""
        
        # Write manifest
        with open(output_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        
        print(f"\nManifest generated: {output_path}")
        
        # Sign manifest if private key provided
        if private_key_path and os.path.exists(private_key_path):
            signature_path = self.sign_file(output_path, private_key_path)
            print(f"Manifest signed: {signature_path}")
        
        return output_path


def main():
    """Main entry point"""
    
    if len(sys.argv) < 4:
        print("Usage: generate_manifest.py <version> <channel> <package_dir> [private_key]")
        print("Example: generate_manifest.py 1.0.0 stable packages keys/private.pem")
        sys.exit(1)
    
    version = sys.argv[1]
    channel = sys.argv[2]
    package_dir = sys.argv[3]
    private_key = sys.argv[4] if len(sys.argv) > 4 else None
    
    print("=" * 60)
    print("DIREWOLF Update Manifest Generator")
    print("=" * 60)
    print(f"Version: {version}")
    print(f"Channel: {channel}")
    print(f"Package Directory: {package_dir}")
    print()
    
    generator = ManifestGenerator(version, channel, package_dir)
    
    # Add packages
    download_url_base = f"https://updates.direwolf.ai/{channel}/{version}"
    
    release_notes = f"""
DIREWOLF {version} Release

New Features:
- Complete XAI system with natural language explanations
- Voice interaction with wake word detection
- 3D network visualization
- Professional video export
- Comprehensive settings panel

Improvements:
- Enhanced permission system
- Improved threat detection
- Better performance
- Updated UI

Bug Fixes:
- Various stability improvements
- Security enhancements
"""
    
    # Add platform-specific packages
    packages = [
        "direwolf-{}-linux-x86_64.deb",
        "direwolf-{}-linux-x86_64.rpm",
        "direwolf-{}-linux-x86_64.AppImage",
        "direwolf-{}-macos-x86_64.dmg",
        "direwolf-{}-macos-x86_64.pkg",
        "direwolf-{}-windows-x86_64.msi"
    ]
    
    for package_template in packages:
        package_file = package_template.format(version)
        generator.add_update(
            package_file,
            release_notes,
            download_url_base,
            requires_restart=True,
            is_critical=False,
            is_delta=False
        )
    
    # Generate manifest
    manifest_path = f"{package_dir}/manifest-{channel}.json"
    generator.generate(manifest_path, private_key)
    
    print("\n" + "=" * 60)
    print("Manifest generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
