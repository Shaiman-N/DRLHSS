#!/usr/bin/env python3
"""
DIREWOLF Update Server
Cloud-based update distribution system with CDN integration
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import boto3
from botocore.exceptions import ClientError

app = FastAPI(title="DIREWOLF Update Server", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPDATE_BUCKET = os.getenv("DIREWOLF_UPDATE_BUCKET", "direwolf-updates")
CDN_DOMAIN = os.getenv("DIREWOLF_CDN_DOMAIN", "cdn.direwolf.ai")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Initialize AWS clients
s3_client = boto3.client('s3', region_name=AWS_REGION)
cloudfront_client = boto3.client('cloudfront', region_name=AWS_REGION)

class UpdateManifest:
    """Update manifest management"""
    
    def __init__(self):
        self.manifests = self._load_manifests()
    
    def _load_manifests(self) -> Dict:
        """Load all update manifests from S3"""
        manifests = {
            "stable": {},
            "beta": {},
            "dev": {}
        }
        
        try:
            # List all manifest files
            response = s3_client.list_objects_v2(
                Bucket=UPDATE_BUCKET,
                Prefix="manifests/"
            )
            
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('.json'):
                    # Download and parse manifest
                    manifest_obj = s3_client.get_object(Bucket=UPDATE_BUCKET, Key=key)
                    manifest_data = json.loads(manifest_obj['Body'].read())
                    
                    # Categorize by channel
                    channel = manifest_data.get('channel', 'stable')
                    platform = manifest_data.get('platform', 'unknown')
                    manifests[channel][platform] = manifest_data
            
            return manifests
        except ClientError as e:
            print(f"Error loading manifests: {e}")
            return manifests
    
    def get_latest(self, channel: str, platform: str, current_version: str) -> Optional[Dict]:
        """Get latest update for platform and channel"""
        if channel not in self.manifests:
            return None
        
        if platform not in self.manifests[channel]:
            return None
        
        manifest = self.manifests[channel][platform]
        latest_version = manifest.get('version')
        
        # Compare versions
        if self._is_newer(latest_version, current_version):
            return manifest
        
        return None
    
    def _is_newer(self, version1: str, version2: str) -> bool:
        """Compare version strings"""
        v1_parts = [int(x) for x in version1.split('.')]
        v2_parts = [int(x) for x in version2.split('.')]
        
        return v1_parts > v2_parts

# Global manifest manager
manifest_manager = UpdateManifest()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "DIREWOLF Update Server",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/api/v1/check-update")
async def check_update(
    platform: str,
    version: str,
    channel: str = "stable",
    request: Request = None
):
    """
    Check for available updates
    
    Parameters:
    - platform: windows, linux, macos
    - version: current version (e.g., "1.0.0")
    - channel: stable, beta, dev
    """
    # Validate parameters
    if platform not in ["windows", "linux", "macos"]:
        raise HTTPException(status_code=400, detail="Invalid platform")
    
    if channel not in ["stable", "beta", "dev"]:
        raise HTTPException(status_code=400, detail="Invalid channel")
    
    # Get client IP for analytics
    client_ip = request.client.host if request else "unknown"
    
    # Log request
    print(f"Update check: platform={platform}, version={version}, channel={channel}, ip={client_ip}")
    
    # Check for updates
    update = manifest_manager.get_latest(channel, platform, version)
    
    if update:
        # Update available
        return {
            "update_available": True,
            "version": update['version'],
            "release_date": update['release_date'],
            "download_url": f"https://{CDN_DOMAIN}/{update['package_path']}",
            "checksum": update['checksum'],
            "checksum_algorithm": "SHA256",
            "size_bytes": update['size_bytes'],
            "release_notes_url": f"https://{CDN_DOMAIN}/{update['release_notes_path']}",
            "mandatory": update.get('mandatory', False),
            "min_version": update.get('min_version', '0.0.0')
        }
    else:
        # No update available
        return {
            "update_available": False,
            "message": "You are running the latest version"
        }

@app.get("/api/v1/manifest/{channel}/{platform}")
async def get_manifest(channel: str, platform: str):
    """Get full update manifest for platform and channel"""
    if channel not in manifest_manager.manifests:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    if platform not in manifest_manager.manifests[channel]:
        raise HTTPException(status_code=404, detail="Platform not found")
    
    return manifest_manager.manifests[channel][platform]

@app.get("/api/v1/versions")
async def list_versions():
    """List all available versions across all channels"""
    versions = {}
    
    for channel, platforms in manifest_manager.manifests.items():
        versions[channel] = {}
        for platform, manifest in platforms.items():
            versions[channel][platform] = {
                "version": manifest.get('version'),
                "release_date": manifest.get('release_date'),
                "size_bytes": manifest.get('size_bytes')
            }
    
    return versions

@app.post("/api/v1/analytics/download")
async def track_download(data: Dict):
    """Track download analytics"""
    # Log download event
    print(f"Download tracked: {json.dumps(data)}")
    
    # In production, send to analytics service
    # analytics_client.track_event('download', data)
    
    return {"status": "recorded"}

@app.post("/api/v1/analytics/install")
async def track_install(data: Dict):
    """Track installation analytics"""
    # Log install event
    print(f"Install tracked: {json.dumps(data)}")
    
    # In production, send to analytics service
    # analytics_client.track_event('install', data)
    
    return {"status": "recorded"}

@app.post("/api/v1/crash-report")
async def submit_crash_report(report: Dict):
    """Submit crash report"""
    # Generate crash ID
    crash_id = hashlib.sha256(
        f"{report.get('timestamp', '')}{report.get('stack_trace', '')}".encode()
    ).hexdigest()[:16]
    
    # Save crash report to S3
    try:
        crash_key = f"crashes/{datetime.now().strftime('%Y/%m/%d')}/{crash_id}.json"
        s3_client.put_object(
            Bucket=UPDATE_BUCKET,
            Key=crash_key,
            Body=json.dumps(report, indent=2),
            ContentType='application/json'
        )
        
        print(f"Crash report saved: {crash_id}")
        
        return {
            "status": "received",
            "crash_id": crash_id,
            "message": "Thank you for the crash report. We'll investigate this issue."
        }
    except ClientError as e:
        print(f"Error saving crash report: {e}")
        raise HTTPException(status_code=500, detail="Failed to save crash report")

@app.get("/api/v1/health")
async def health_check():
    """Detailed health check"""
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Check S3 connectivity
    try:
        s3_client.head_bucket(Bucket=UPDATE_BUCKET)
        health["services"]["s3"] = "operational"
    except ClientError:
        health["services"]["s3"] = "degraded"
        health["status"] = "degraded"
    
    # Check CloudFront connectivity
    try:
        cloudfront_client.list_distributions(MaxItems='1')
        health["services"]["cloudfront"] = "operational"
    except ClientError:
        health["services"]["cloudfront"] = "degraded"
    
    return health

@app.post("/api/v1/invalidate-cache")
async def invalidate_cache(paths: List[str]):
    """Invalidate CDN cache for specified paths"""
    # In production, require authentication
    # This is a privileged operation
    
    try:
        # Create CloudFront invalidation
        distribution_id = os.getenv("CLOUDFRONT_DISTRIBUTION_ID")
        
        if not distribution_id:
            raise HTTPException(status_code=500, detail="CloudFront not configured")
        
        response = cloudfront_client.create_invalidation(
            DistributionId=distribution_id,
            InvalidationBatch={
                'Paths': {
                    'Quantity': len(paths),
                    'Items': paths
                },
                'CallerReference': str(datetime.now().timestamp())
            }
        )
        
        return {
            "status": "invalidation_created",
            "invalidation_id": response['Invalidation']['Id']
        }
    except ClientError as e:
        print(f"Error creating invalidation: {e}")
        raise HTTPException(status_code=500, detail="Failed to invalidate cache")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
