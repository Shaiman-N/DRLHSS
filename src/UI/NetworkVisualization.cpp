/**
 * @file NetworkVisualization.cpp
 * @brief 3D Network Visualization Implementation
 */

#include "UI/NetworkVisualization.hpp"
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <cmath>
#include <algorithm>

namespace ui {

NetworkVisualization::NetworkVisualization(QWidget* parent)
    : QOpenGLWidget(parent)
    , mouse_pressed_(false)
    , zoom_level_(1.0f)
    , rotation_x_(30.0f)
    , rotation_y_(45.0f)
    , animation_time_(0.0f)
    , auto_layout_enabled_(true)
    , layout_spring_strength_(0.01f)
    , layout_repulsion_strength_(100.0f)
    , layout_damping_(0.9f)
    , show_labels_(true)
    , show_connections_(true)
    , show_threat_effects_(true)
    , node_scale_(1.0f)
    , connection_scale_(1.0f)
{
    // Initialize camera
    camera_position_ = QVector3D(0.0f, 0.0f, 50.0f);
    camera_target_ = QVector3D(0.0f, 0.0f, 0.0f);
    camera_up_ = QVector3D(0.0f, 1.0f, 0.0f);
    
    // Setup animation timer
    animation_timer_ = std::make_unique<QTimer>(this);
    connect(animation_timer_.get(), &QTimer::timeout, this, &NetworkVisualization::updateAnimation);
    animation_timer_->start(16); // ~60 FPS
    
    setMouseTracking(true);
}

NetworkVisualization::~NetworkVisualization() = default;

void NetworkVisualization::initializeGL() {
    initializeOpenGLFunctions();
    
    glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
}

void NetworkVisualization::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Update view matrix
    view_matrix_.setToIdentity();
    view_matrix_.lookAt(camera_position_, camera_target_, camera_up_);
    view_matrix_.rotate(rotation_x_, 1.0f, 0.0f, 0.0f);
    view_matrix_.rotate(rotation_y_, 0.0f, 1.0f, 0.0f);
    view_matrix_.scale(zoom_level_);
    
    // Render scene
    if (show_connections_) {
        renderConnections();
    }
    
    renderNodes();
    
    if (show_threat_effects_) {
        renderThreatEffects();
    }
}

void NetworkVisualization::resizeGL(int width, int height) {
    glViewport(0, 0, width, height);
    
    projection_matrix_.setToIdentity();
    float aspect = float(width) / float(height ? height : 1);
    projection_matrix_.perspective(45.0f, aspect, 0.1f, 1000.0f);
}

void NetworkVisualization::addNode(const NetworkNode& node) {
    nodes_.push_back(node);
    update();
}

void NetworkVisualization::removeNode(const std::string& node_id) {
    nodes_.erase(
        std::remove_if(nodes_.begin(), nodes_.end(),
            [&node_id](const NetworkNode& n) { return n.id == node_id; }),
        nodes_.end()
    );
    
    // Remove connections involving this node
    connections_.erase(
        std::remove_if(connections_.begin(), connections_.end(),
            [&node_id](const NetworkConnection& c) {
                return c.from_node == node_id || c.to_node == node_id;
            }),
        connections_.end()
    );
    
    update();
}

void NetworkVisualization::updateNodeThreat(const std::string& node_id, ThreatLevel threat_level) {
    if (auto* node = findNode(node_id)) {
        node->threat_level = threat_level;
        node->color = getNodeColor(node->type, threat_level);
        
        if (threat_level >= ThreatLevel::HIGH) {
            startThreatAnimation(node_id);
        }
        
        update();
    }
}

void NetworkVisualization::highlightNode(const std::string& node_id, bool highlight) {
    if (auto* node = findNode(node_id)) {
        node->highlighted = highlight;
        update();
    }
}

void NetworkVisualization::addConnection(const NetworkConnection& connection) {
    connections_.push_back(connection);
    update();
}

void NetworkVisualization::removeConnection(const std::string& from_node, const std::string& to_node) {
    connections_.erase(
        std::remove_if(connections_.begin(), connections_.end(),
            [&](const NetworkConnection& c) {
                return c.from_node == from_node && c.to_node == to_node;
            }),
        connections_.end()
    );
    update();
}

void NetworkVisualization::updateConnectionThreat(const std::string& from_node, 
                                                   const std::string& to_node, 
                                                   bool is_threat) {
    if (auto* conn = findConnection(from_node, to_node)) {
        conn->is_threat = is_threat;
        conn->color = is_threat ? QVector3D(1.0f, 0.2f, 0.2f) : QVector3D(0.3f, 0.6f, 1.0f);
        conn->animated = is_threat;
        update();
    }
}

void NetworkVisualization::applyForceDirectedLayout() {
    auto_layout_enabled_ = true;
}

void NetworkVisualization::applyCircularLayout() {
    auto_layout_enabled_ = false;
    
    float radius = 20.0f;
    float angle_step = 2.0f * M_PI / nodes_.size();
    
    for (size_t i = 0; i < nodes_.size(); ++i) {
        float angle = i * angle_step;
        nodes_[i].position = QVector3D(
            radius * std::cos(angle),
            radius * std::sin(angle),
            0.0f
        );
    }
    
    update();
}

void NetworkVisualization::applyHierarchicalLayout() {
    auto_layout_enabled_ = false;
    
    // Simple hierarchical layout by node type
    std::map<NodeType, std::vector<NetworkNode*>> layers;
    
    for (auto& node : nodes_) {
        layers[node.type].push_back(&node);
    }
    
    float y_offset = 15.0f;
    float layer_spacing = 10.0f;
    
    for (auto& [type, layer_nodes] : layers) {
        float x_spacing = 30.0f / (layer_nodes.size() + 1);
        
        for (size_t i = 0; i < layer_nodes.size(); ++i) {
            layer_nodes[i]->position = QVector3D(
                -15.0f + (i + 1) * x_spacing,
                y_offset,
                0.0f
            );
        }
        
        y_offset -= layer_spacing;
    }
    
    update();
}

void NetworkVisualization::resetCamera() {
    camera_position_ = QVector3D(0.0f, 0.0f, 50.0f);
    camera_target_ = QVector3D(0.0f, 0.0f, 0.0f);
    zoom_level_ = 1.0f;
    rotation_x_ = 30.0f;
    rotation_y_ = 45.0f;
    update();
}

void NetworkVisualization::focusOnNode(const std::string& node_id) {
    if (auto* node = findNode(node_id)) {
        camera_target_ = node->position;
        update();
    }
}

void NetworkVisualization::setCameraPosition(const QVector3D& position, const QVector3D& target) {
    camera_position_ = position;
    camera_target_ = target;
    update();
}

void NetworkVisualization::startThreatAnimation(const std::string& node_id) {
    if (auto* node = findNode(node_id)) {
        node->is_pulsing = true;
        node->pulse_phase = 0.0f;
    }
}

void NetworkVisualization::stopThreatAnimation(const std::string& node_id) {
    if (auto* node = findNode(node_id)) {
        node->is_pulsing = false;
    }
}

void NetworkVisualization::animateAttackPath(const std::vector<std::string>& path) {
    // Highlight all nodes in path
    for (const auto& node_id : path) {
        highlightNode(node_id, true);
        startThreatAnimation(node_id);
    }
    
    // Animate connections between path nodes
    for (size_t i = 0; i < path.size() - 1; ++i) {
        updateConnectionThreat(path[i], path[i + 1], true);
    }
}

void NetworkVisualization::mousePressEvent(QMouseEvent* event) {
    mouse_pressed_ = true;
    last_mouse_pos_ = event->pos();
    
    // Check for node selection
    std::string picked = pickNode(event->x(), event->y());
    if (!picked.empty()) {
        emit nodeClicked(QString::fromStdString(picked));
    }
}

void NetworkVisualization::mouseMoveEvent(QMouseEvent* event) {
    if (mouse_pressed_) {
        QPoint delta = event->pos() - last_mouse_pos_;
        
        rotation_y_ += delta.x() * 0.5f;
        rotation_x_ += delta.y() * 0.5f;
        
        // Clamp rotation
        rotation_x_ = std::clamp(rotation_x_, -89.0f, 89.0f);
        
        last_mouse_pos_ = event->pos();
        update();
    } else {
        // Check for hover
        std::string hovered = pickNode(event->x(), event->y());
        if (!hovered.empty()) {
            emit nodeHovered(QString::fromStdString(hovered));
        }
    }
}

void NetworkVisualization::wheelEvent(QWheelEvent* event) {
    float delta = event->angleDelta().y() / 120.0f;
    zoom_level_ *= (1.0f + delta * 0.1f);
    zoom_level_ = std::clamp(zoom_level_, 0.1f, 10.0f);
    update();
}

void NetworkVisualization::updateAnimation() {
    animation_time_ += 0.016f; // ~60 FPS
    
    // Update force-directed layout
    if (auto_layout_enabled_ && !nodes_.empty()) {
        updateForceDirectedLayout();
    }
    
    // Update pulsing animations
    for (auto& node : nodes_) {
        if (node.is_pulsing) {
            node.pulse_phase += 0.1f;
        }
    }
    
    update();
}

void NetworkVisualization::renderNodes() {
    for (const auto& node : nodes_) {
        renderNode(node);
    }
}

void NetworkVisualization::renderConnections() {
    for (const auto& connection : connections_) {
        renderConnection(connection);
    }
}

void NetworkVisualization::renderNode(const NetworkNode& node) {
    // Simple sphere rendering using OpenGL immediate mode (for demonstration)
    // In production, would use VBOs and shaders
    
    glPushMatrix();
    glTranslatef(node.position.x(), node.position.y(), node.position.z());
    
    float size = node.size * node_scale_;
    
    // Apply pulsing effect
    if (node.is_pulsing) {
        float pulse = 1.0f + 0.3f * std::sin(node.pulse_phase);
        size *= pulse;
    }
    
    // Set color
    if (node.highlighted) {
        glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
    } else {
        glColor4f(node.color.x(), node.color.y(), node.color.z(), 1.0f);
    }
    
    // Draw sphere (simplified)
    glBegin(GL_TRIANGLE_FAN);
    for (int i = 0; i <= 20; ++i) {
        float angle = 2.0f * M_PI * i / 20.0f;
        glVertex3f(size * std::cos(angle), size * std::sin(angle), 0.0f);
    }
    glEnd();
    
    glPopMatrix();
}

void NetworkVisualization::renderConnection(const NetworkConnection& connection) {
    auto* from = findNode(connection.from_node);
    auto* to = findNode(connection.to_node);
    
    if (!from || !to) return;
    
    glLineWidth(connection.thickness * connection_scale_);
    glColor4f(connection.color.x(), connection.color.y(), connection.color.z(), 0.6f);
    
    glBegin(GL_LINES);
    glVertex3f(from->position.x(), from->position.y(), from->position.z());
    glVertex3f(to->position.x(), to->position.y(), to->position.z());
    glEnd();
}

void NetworkVisualization::renderThreatEffects() {
    // Render threat indicators (rings, particles, etc.)
    for (const auto& node : nodes_) {
        if (node.threat_level >= ThreatLevel::HIGH) {
            glPushMatrix();
            glTranslatef(node.position.x(), node.position.y(), node.position.z());
            
            float ring_size = node.size * 2.0f * (1.0f + 0.5f * std::sin(animation_time_ * 2.0f));
            
            glColor4f(1.0f, 0.0f, 0.0f, 0.3f);
            glBegin(GL_LINE_LOOP);
            for (int i = 0; i <= 30; ++i) {
                float angle = 2.0f * M_PI * i / 30.0f;
                glVertex3f(ring_size * std::cos(angle), ring_size * std::sin(angle), 0.0f);
            }
            glEnd();
            
            glPopMatrix();
        }
    }
}

void NetworkVisualization::updateForceDirectedLayout() {
    // Apply forces to each node
    for (auto& node : nodes_) {
        QVector3D force = calculateForces(node);
        node.velocity += force;
        node.velocity *= layout_damping_;
        node.position += node.velocity;
    }
}

QVector3D NetworkVisualization::calculateForces(const NetworkNode& node) {
    QVector3D total_force(0.0f, 0.0f, 0.0f);
    
    // Repulsion from other nodes
    for (const auto& other : nodes_) {
        if (other.id == node.id) continue;
        
        QVector3D diff = node.position - other.position;
        float dist = diff.length();
        
        if (dist > 0.1f) {
            QVector3D repulsion = diff.normalized() * (layout_repulsion_strength_ / (dist * dist));
            total_force += repulsion;
        }
    }
    
    // Spring attraction along connections
    for (const auto& conn : connections_) {
        const NetworkNode* other = nullptr;
        
        if (conn.from_node == node.id) {
            other = findNode(conn.to_node);
        } else if (conn.to_node == node.id) {
            other = findNode(conn.from_node);
        }
        
        if (other) {
            QVector3D diff = other->position - node.position;
            QVector3D attraction = diff * layout_spring_strength_;
            total_force += attraction;
        }
    }
    
    return total_force;
}

QVector3D NetworkVisualization::getNodeColor(NodeType type, ThreatLevel threat_level) {
    // Base color by type
    QVector3D base_color;
    
    switch (type) {
        case NodeType::SERVER:
            base_color = QVector3D(0.3f, 0.6f, 1.0f);
            break;
        case NodeType::WORKSTATION:
            base_color = QVector3D(0.5f, 0.8f, 0.5f);
            break;
        case NodeType::ROUTER:
            base_color = QVector3D(0.8f, 0.6f, 0.3f);
            break;
        case NodeType::FIREWALL:
            base_color = QVector3D(0.9f, 0.5f, 0.2f);
            break;
        case NodeType::THREAT:
            base_color = QVector3D(1.0f, 0.2f, 0.2f);
            break;
        default:
            base_color = QVector3D(0.5f, 0.5f, 0.5f);
    }
    
    // Modify by threat level
    if (threat_level >= ThreatLevel::CRITICAL) {
        return QVector3D(1.0f, 0.0f, 0.0f);
    } else if (threat_level >= ThreatLevel::HIGH) {
        return base_color * 0.5f + QVector3D(1.0f, 0.0f, 0.0f) * 0.5f;
    }
    
    return base_color;
}

float NetworkVisualization::getNodeSize(NodeType type) {
    switch (type) {
        case NodeType::SERVER:
            return 1.5f;
        case NodeType::ROUTER:
        case NodeType::FIREWALL:
            return 1.2f;
        case NodeType::THREAT:
            return 1.8f;
        default:
            return 1.0f;
    }
}

NetworkNode* NetworkVisualization::findNode(const std::string& node_id) {
    auto it = std::find_if(nodes_.begin(), nodes_.end(),
        [&node_id](const NetworkNode& n) { return n.id == node_id; });
    
    return it != nodes_.end() ? &(*it) : nullptr;
}

NetworkConnection* NetworkVisualization::findConnection(const std::string& from, const std::string& to) {
    auto it = std::find_if(connections_.begin(), connections_.end(),
        [&](const NetworkConnection& c) {
            return c.from_node == from && c.to_node == to;
        });
    
    return it != connections_.end() ? &(*it) : nullptr;
}

std::string NetworkVisualization::pickNode(int x, int y) {
    // Simplified ray casting - in production would use proper picking
    // For now, return empty string
    return "";
}

QVector3D NetworkVisualization::screenToWorld(int x, int y) {
    // Convert screen coordinates to world coordinates
    // Simplified implementation
    return QVector3D(0.0f, 0.0f, 0.0f);
}

} // namespace ui
