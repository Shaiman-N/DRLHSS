/**
 * Dashboard.qml
 * DIREWOLF Main Dashboard
 * 
 * Displays real-time metrics, active alerts, and component status.
 */

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    id: dashboardWindow
    width: 1000
    height: 700
    minimumWidth: 800
    minimumHeight: 600
    title: "DIREWOLF Dashboard"
    
    // Color scheme
    readonly property color bgColor: "#1a1a1a"
    readonly property color cardColor: "#2a2a2a"
    readonly property color accentColor: "#4a9eff"
    readonly property color dangerColor: "#ff4a4a"
    readonly property color warningColor: "#ffaa4a"
    readonly property color successColor: "#4aff4a"
    readonly property color textColor: "#ffffff"
    readonly property color textSecondary: "#aaaaaa"
    
    color: bgColor
    
    // Header
    header: Rectangle {
        height: 60
        color: cardColor
        
        RowLayout {
            anchors.fill: parent
            anchors.margins: 15
            spacing: 20
            
            // Logo and title
            Row {
                spacing: 10
                
                Rectangle {
                    width: 40
                    height: 40
                    radius: 20
                    color: accentColor
                    
                    Text {
                        anchors.centerIn: parent
                        text: "🐺"
                        font.pixelSize: 24
                    }
                }
                
                Column {
                    anchors.verticalCenter: parent.verticalCenter
                    
                    Text {
                        text: "DIREWOLF"
                        font.pixelSize: 20
                        font.bold: true
                        color: textColor
                    }
                    
                    Text {
                        text: "Security Guardian"
                        font.pixelSize: 12
                        color: textSecondary
                    }
                }
            }
            
            Item { Layout.fillWidth: true }
            
            // System status
            Rectangle {
                width: 150
                height: 35
                radius: 17.5
                color: successColor
                
                Text {
                    anchors.centerIn: parent
                    text: "● HEALTHY"
                    font.pixelSize: 14
                    font.bold: true
                    color: "#000000"
                }
            }
        }
    }
    
    // Main content
    ScrollView {
        anchors.fill: parent
        anchors.margins: 20
        
        ColumnLayout {
            width: parent.width
            spacing: 20
            
            // Quick stats row
            RowLayout {
                Layout.fillWidth: true
                spacing: 15
                
                // Threats today
                StatCard {
                    Layout.fillWidth: true
                    title: "Threats Today"
                    value: "12"
                    subtitle: "3 blocked"
                    iconText: "🛡️"
                    color: dangerColor
                }
                
                // System health
                StatCard {
                    Layout.fillWidth: true
                    title: "System Health"
                    value: "98%"
                    subtitle: "All systems operational"
                    iconText: "💚"
                    color: successColor
                }
                
                // DRL confidence
                StatCard {
                    Layout.fillWidth: true
                    title: "DRL Confidence"
                    value: "94%"
                    subtitle: "High accuracy"
                    iconText: "🧠"
                    color: accentColor
                }
                
                // Active alerts
                StatCard {
                    Layout.fillWidth: true
                    title: "Active Alerts"
                    value: "2"
                    subtitle: "Awaiting decision"
                    iconText: "⚠️"
                    color: warningColor
                }
            }
            
            // Component status
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 200
                color: cardColor
                radius: 10
                
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 20
                    spacing: 15
                    
                    Text {
                        text: "Component Status"
                        font.pixelSize: 18
                        font.bold: true
                        color: textColor
                    }
                    
                    GridLayout {
                        Layout.fillWidth: true
                        columns: 3
                        rowSpacing: 10
                        columnSpacing: 15
                        
                        ComponentStatus { name: "Antivirus"; status: "RUNNING" }
                        ComponentStatus { name: "NIDPS"; status: "RUNNING" }
                        ComponentStatus { name: "DRL Agent"; status: "RUNNING" }
                        ComponentStatus { name: "Sandbox"; status: "RUNNING" }
                        ComponentStatus { name: "Telemetry"; status: "RUNNING" }
                        ComponentStatus { name: "Database"; status: "RUNNING" }
                    }
                }
            }
            
            // Active alerts
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 250
                color: cardColor
                radius: 10
                
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 20
                    spacing: 15
                    
                    Text {
                        text: "Active Alerts"
                        font.pixelSize: 18
                        font.bold: true
                        color: textColor
                    }
                    
                    ListView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        spacing: 10
                        clip: true
                        
                        model: ListModel {
                            ListElement {
                                threatType: "Malware"
                                fileName: "suspicious.exe"
                                severity: "CRITICAL"
                                time: "2 minutes ago"
                            }
                            ListElement {
                                threatType: "Network Intrusion"
                                fileName: "192.168.1.100"
                                severity: "HIGH"
                                time: "5 minutes ago"
                            }
                        }
                        
                        delegate: AlertItem {
                            width: ListView.view.width
                            threatType: model.threatType
                            fileName: model.fileName
                            severity: model.severity
                            time: model.time
                        }
                    }
                }
            }
        }
    }
}

// Stat card component
component StatCard: Rectangle {
    property string title: ""
    property string value: ""
    property string subtitle: ""
    property string iconText: ""
    property color color: "#4a9eff"
    
    height: 120
    color: "#2a2a2a"
    radius: 10
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 15
        spacing: 5
        
        Row {
            spacing: 10
            
            Text {
                text: iconText
                font.pixelSize: 24
            }
            
            Text {
                text: title
                font.pixelSize: 14
                color: "#aaaaaa"
            }
        }
        
        Text {
            text: value
            font.pixelSize: 32
            font.bold: true
            color: parent.parent.parent.color
        }
        
        Text {
            text: subtitle
            font.pixelSize: 12
            color: "#aaaaaa"
        }
    }
}

// Component status component
component ComponentStatus: Row {
    property string name: ""
    property string status: "UNKNOWN"
    
    spacing: 10
    
    Rectangle {
        width: 10
        height: 10
        radius: 5
        anchors.verticalCenter: parent.verticalCenter
        color: status === "RUNNING" ? "#4aff4a" : "#ff4a4a"
    }
    
    Text {
        text: name
        font.pixelSize: 14
        color: "#ffffff"
        anchors.verticalCenter: parent.verticalCenter
    }
}

// Alert item component
component AlertItem: Rectangle {
    property string threatType: ""
    property string fileName: ""
    property string severity: ""
    property string time: ""
    
    height: 60
    color: "#333333"
    radius: 8
    
    RowLayout {
        anchors.fill: parent
        anchors.margins: 15
        spacing: 15
        
        Rectangle {
            width: 40
            height: 40
            radius: 20
            color: severity === "CRITICAL" ? "#ff4a4a" : "#ffaa4a"
            
            Text {
                anchors.centerIn: parent
                text: "⚠️"
                font.pixelSize: 20
            }
        }
        
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 5
            
            Text {
                text: threatType
                font.pixelSize: 14
                font.bold: true
                color: "#ffffff"
            }
            
            Text {
                text: fileName
                font.pixelSize: 12
                color: "#aaaaaa"
            }
        }
        
        Text {
            text: time
            font.pixelSize: 12
            color: "#aaaaaa"
        }
        
        Button {
            text: "Review"
            font.pixelSize: 12
            
            background: Rectangle {
                color: "#4a9eff"
                radius: 5
            }
            
            contentItem: Text {
                text: parent.text
                font: parent.font
                color: "#ffffff"
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
        }
    }
}
