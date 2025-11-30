# DIREWOLF Phase 5 Quick Reference

## Visualization & Video Export System

Quick reference for using Phase 5 components.

---

## 3D Network Visualization

### Basic Usage

```cpp
#include "UI/NetworkVisualization.hpp"

// Create widget
auto* viz = new NetworkVisualization(parent);

// Add node
NetworkNode node;
node.id = "server_001";
node.name = "Web Server";
node.ip_address = "192.168.1.100";
node.type = NodeType::SERVER;
node.threat_level = ThreatLevel::NONE;
viz->addNode(node);

// Add connection
NetworkConnection conn;
conn.from_node = "server_001";
conn.to_node = "workstation_001";
conn.bandwidth = 1000.0f;
viz->addConnection(conn);

// Update threat
viz->updateNodeThreat("server_001", ThreatLevel::HIGH);

// Animate attack
std::vector<std::string> path = {"attacker", "router", "server_001"};
viz->animateAttackPath(path);
```

### Layout Algorithms

```cpp
// Force-directed (automatic)
viz->applyForceDirectedLayout();

// Circular
viz->applyCircularLayout();

// Hierarchical
viz->applyHierarchicalLayout();
```

### Camera Control

```cpp
// Reset view
viz->resetCamera();

// Focus on node
viz->focusOnNode("server_001");

// Custom position
viz->setCameraPosition(
    QVector3D(0, 0, 50),  // position
    QVector3D(0, 0, 0)    // target
);
```

### Signals

```cpp
// Node clicked
connect(viz, &NetworkVisualization::nodeClicked,
    [](const QString& node_id) {
        qDebug() << "Clicked:" << node_id;
    });

// Node hovered
connect(viz, &NetworkVisualization::nodeHovered,
    [](const QString& node_id) {
        qDebug() << "Hovered:" << node_id;
    });
```

---

## Video Renderer

### Basic Usage

```python
from xai.video_renderer import VideoRenderer

# Initialize
config = {
    'ffmpeg_path': 'ffmpeg',
    'output_dir': 'videos',
    'branding': {
        'logo_path': 'assets/direwolf_logo.png'
    }
}
renderer = VideoRenderer(config)

# Render incident video
scenes = [
    {'type': 'overview', 'duration': 5},
    {'type': 'detection', 'duration': 8},
    {'type': 'response', 'duration': 5}
]

narration = [
    {'timestamp': '00:00', 'text': 'Incident detected'},
    {'timestamp': '00:05', 'text': 'Analyzing threat'},
    {'timestamp': '00:13', 'text': 'Threat contained'}
]

video_path = renderer.render_incident_video(
    incident_id="INC_001",
    scenes=scenes,
    narration=narration,
    quality='1080p',
    format='mp4'
)
```

### Daily Briefing

```python
# Render briefing
briefing_data = {
    'date': '2024-01-15',
    'executive_summary': 'All systems secure...',
    'threat_overview': '3 threats detected...'
}

video = renderer.render_daily_briefing_video(
    briefing_data=briefing_data,
    narration_audio='briefing.wav',
    quality='1080p'
)
```

### Quality Presets

```python
# Available presets
'720p'   # 1280x720, 2.5 Mbps, 30 FPS
'1080p'  # 1920x1080, 5 Mbps, 30 FPS
'4k'     # 3840x2160, 15 Mbps, 30 FPS
```

### Format Conversion

```python
# Convert format
output = renderer.convert_video_format(
    input_path='video.mp4',
    output_format='avi',
    quality='1080p'
)
```

---

## Video Library Manager

### Initialization

```cpp
#include "UI/VideoLibraryManager.hpp"

VideoLibraryManager library;
library.initialize("/path/to/library");
```

### Add Video

```cpp
VideoMetadata metadata;
metadata.title = "Incident INC_001";
metadata.description = "DDoS attack mitigation";
metadata.incident_id = "INC_001";
metadata.video_type = "incident";
metadata.quality = "1080p";
metadata.format = "mp4";
metadata.tags = {"ddos", "critical", "mitigated"};

QString video_id = library.addVideo("/path/to/video.mp4", metadata);
```

### Search & Filter

```cpp
// Search by title
auto results = library.searchByTitle("DDoS");

// Filter by type
auto incidents = library.filterByType("incident");

// Filter by date range
QDateTime start = QDateTime::currentDateTime().addDays(-7);
QDateTime end = QDateTime::currentDateTime();
auto recent = library.filterByDateRange(start, end);

// Filter by tags
QStringList tags = {"critical", "ddos"};
auto tagged = library.filterByTags(tags);

// Get all videos
auto all = library.getAllVideos();
```

### Thumbnails

```cpp
// Generate thumbnail
library.generateThumbnail(video_id, 5);  // at 5 seconds

// Get thumbnail
QPixmap thumb = library.getThumbnail(video_id);

// Regenerate all
library.regenerateAllThumbnails();
```

### Sharing & Export

```cpp
// Share video
QString share_url = library.shareVideo(video_id, "link");

// Export with metadata
library.exportVideo(video_id, "/export/path");
```

### Statistics

```cpp
// Get library stats
QVariantMap stats = library.getLibraryStatistics();
int total = stats["total_videos"].toInt();
qint64 storage = stats["total_storage"].toLongLong();

// Get storage usage
qint64 usage = library.getStorageUsage();

// Cleanup old videos
int removed = library.cleanupOldVideos(30);  // older than 30 days
```

### Signals

```cpp
// Video added
connect(&library, &VideoLibraryManager::videoAdded,
    [](const QString& video_id) {
        qDebug() << "Added:" << video_id;
    });

// Video removed
connect(&library, &VideoLibraryManager::videoRemoved,
    [](const QString& video_id) {
        qDebug() << "Removed:" << video_id;
    });

// Thumbnail generated
connect(&library, &VideoLibraryManager::thumbnailGenerated,
    [](const QString& video_id) {
        qDebug() << "Thumbnail ready:" << video_id;
    });
```

---

## Settings Panel

### Launch Settings

```qml
import QtQuick 2.15
import QtQuick.Controls 2.15

Button {
    text: "Settings"
    onClicked: {
        var component = Qt.createComponent("SettingsPanel.qml")
        var window = component.createObject()
        window.show()
    }
}
```

### Access Settings Data

```qml
// Settings are stored in the settings object
console.log("TTS Provider:", settings.voice.tts_provider)
console.log("Speaking Rate:", settings.voice.speaking_rate)
console.log("Auto Update:", settings.updates.auto_update)
console.log("Display Name:", settings.user.display_name)
```

### Save Settings

```qml
// Called automatically when Save button clicked
function saveSettings() {
    // Settings are saved to backend
    console.log("Saving:", JSON.stringify(settings))
}
```

### Reset to Defaults

```qml
// Called when Reset button clicked
function resetToDefaults() {
    // All settings reset to default values
}
```

---

## Complete Workflow Example

### Incident Video Creation

```cpp
// 1. Detect incident
ThreatEvent threat = detectThreat();

// 2. Visualize network
NetworkVisualization* viz = new NetworkVisualization();
for (const auto& host : network.hosts) {
    viz->addNode(createNode(host));
}
viz->animateAttackPath(threat.attack_path);

// 3. Generate explanation
auto explanation = explainer.explain(threat);

// 4. Create narration
std::vector<NarrationSegment> narration;
narration.push_back({
    "00:00",
    "Alpha, I detected a " + threat.type + " attack"
});
narration.push_back({
    "00:05",
    explanation.summary
});
narration.push_back({
    "00:10",
    "I recommend " + threat.recommended_action
});

// 5. Render video (Python)
video_path = renderer.render_incident_video(
    incident_id=threat.id,
    scenes=createScenes(viz),
    narration=narration,
    quality='1080p'
)

// 6. Add to library
VideoMetadata metadata;
metadata.title = "Incident " + threat.id;
metadata.description = explanation.summary;
metadata.incident_id = threat.id;
metadata.video_type = "incident";
metadata.tags = threat.tags;

QString video_id = library.addVideo(
    QString::fromStdString(video_path),
    metadata
);

// 7. Notify Alpha
QString share_url = library.shareVideo(video_id, "link");
notifyAlpha("Incident video ready: " + share_url);
```

---

## Configuration

### CMakeLists.txt

```cmake
# Network Visualization
find_package(Qt5 COMPONENTS OpenGL REQUIRED)

add_library(network_visualization
    src/UI/NetworkVisualization.cpp
)

target_link_libraries(network_visualization
    Qt5::Widgets
    Qt5::OpenGL
    ${OPENGL_LIBRARIES}
)

# Video Library
find_package(Qt5 COMPONENTS Sql REQUIRED)

add_library(video_library
    src/UI/VideoLibraryManager.cpp
)

target_link_libraries(video_library
    Qt5::Core
    Qt5::Sql
)
```

### Python Requirements

```bash
# No additional packages required
# Uses standard library + subprocess for FFmpeg

# Ensure FFmpeg is installed:
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Windows: Download from ffmpeg.org
```

---

## Troubleshooting

### Network Visualization

**Issue**: Low frame rate
```cpp
// Reduce node count or disable animations
viz->show_threat_effects_ = false;
viz->auto_layout_enabled_ = false;
```

**Issue**: Nodes overlapping
```cpp
// Adjust layout parameters
viz->layout_repulsion_strength_ = 200.0f;
viz->applyForceDirectedLayout();
```

### Video Renderer

**Issue**: FFmpeg not found
```python
# Specify full path
config['ffmpeg_path'] = '/usr/bin/ffmpeg'
```

**Issue**: Slow rendering
```python
# Use lower quality
video = renderer.render_incident_video(
    ...,
    quality='720p'  # instead of 1080p
)
```

### Video Library

**Issue**: Database locked
```cpp
// Close and reopen
library.~VideoLibraryManager();
library.initialize(library_path);
```

**Issue**: Thumbnails not generating
```cpp
// Check FFmpeg installation
// Manually regenerate
library.regenerateAllThumbnails();
```

---

## Performance Tips

### Network Visualization
- Limit to 500 nodes for smooth 60 FPS
- Disable auto-layout for static networks
- Use hierarchical layout for large networks
- Batch node updates

### Video Rendering
- Use 720p for faster rendering
- Pre-generate scenes in parallel
- Cache rendered segments
- Use hardware acceleration if available

### Video Library
- Index frequently searched fields
- Batch thumbnail generation
- Cleanup old videos regularly
- Use SSD for library storage

---

## API Reference

Full API documentation available in:
- `include/UI/NetworkVisualization.hpp`
- `include/UI/VideoLibraryManager.hpp`
- `python/xai/video_renderer.py`
- `qml/SettingsPanel.qml`

---

**DIREWOLF Phase 5 - Visualization & Video Export**  
*"See the threat. Document the response. Share the knowledge."*
