/**
 * PermissionDialog.qml
 * DIREWOLF Permission Request Dialog
 * 
 * Displays threat details and requests Alpha's decision.
 */

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Dialog {
    id: permissionDialog
    width: 600
    height: 500
    modal: true
    title: "DIREWOLF - Permission Required"
    
    // Properties
    property string threatType: "Malware"
    property string fileName: "suspicious.exe"
    property string filePath: "/tmp/suspicious.exe"
    property string severity: "CRITICAL"
    property string recommendedAction: "QUARANTINE"
    property real confidence: 0.94
    property string explanation: "This file exhibits malicious behavior patterns..."
    
    // Color scheme
    readonly property color bgColor: "#1a1a1a"
    readonly property color cardColor: "#2a2a2a"
    readonly property color accentColor: "#4a9eff"
    readonly property color dangerColor: "#ff4a4a"
    readonly property color warningColor: "#ffaa4a"
    readonly property color successColor: "#4aff4a"
    readonly property color textColor: "#ffffff"
    readonly property color textSecondary: "#aaaaaa"
    
    background: Rectangle {
        color: bgColor
        radius: 10
    }
    
    // Content
    ColumnLayout {
        anchors.fill: parent
        spacing: 20
        
        // Header with urgency indicator
        Rectangle {
            Layout.fillWidth: true
            height: 80
            color: severity === "CRITICAL" ? dangerColor : warningColor
            radius: 8
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 20
                spacing: 15
                
                Text {
                    text: "⚠️"
                    font.pixelSize: 40
                }
                
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 5
                    
                    Text {
                        text: "Alpha, I need your decision"
                        font.pixelSize: 18
                        font.bold: true
                        color: "#000000"
                    }
                    
                    Text {
                        text: "Threat detected - " + severity + " severity"
                        font.pixelSize: 14
                        color: "#000000"
                    }
                }
            }
        }
        
        // Threat details
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 200
            color: cardColor
            radius: 8
            
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 20
                spacing: 15
                
                Text {
                    text: "Threat Details"
                    font.pixelSize: 16
                    font.bold: true
                    color: textColor
                }
                
                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    rowSpacing: 10
                    columnSpacing: 20
                    
                    Text {
                        text: "Type:"
                        font.pixelSize: 14
                        color: textSecondary
                    }
                    Text {
                        text: threatType
                        font.pixelSize: 14
                        font.bold: true
                        color: textColor
                    }
                    
                    Text {
                        text: "File:"
                        font.pixelSize: 14
                        color: textSecondary
                    }
                    Text {
                        text: fileName
                        font.pixelSize: 14
                        font.bold: true
                        color: textColor
                    }
                    
                    Text {
                        text: "Path:"
                        font.pixelSize: 14
                        color: textSecondary
                    }
                    Text {
                        text: filePath
                        font.pixelSize: 14
                        color: textColor
                        elide: Text.ElideMiddle
                        Layout.fillWidth: true
                    }
                    
                    Text {
                        text: "Confidence:"
                        font.pixelSize: 14
                        color: textSecondary
                    }
                    Row {
                        spacing: 10
                        
                        ProgressBar {
                            width: 150
                            value: confidence
                            
                            background: Rectangle {
                                color: "#444444"
                                radius: 3
                            }
                            
                            contentItem: Rectangle {
                                width: parent.visualPosition * parent.width
                                color: confidence > 0.8 ? successColor : warningColor
                                radius: 3
                            }
                        }
                        
                        Text {
                            text: Math.round(confidence * 100) + "%"
                            font.pixelSize: 14
                            font.bold: true
                            color: textColor
                        }
                    }
                }
            }
        }
        
        // Wolf's recommendation
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 100
            color: cardColor
            radius: 8
            
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 20
                spacing: 10
                
                Text {
                    text: "Wolf's Recommendation"
                    font.pixelSize: 16
                    font.bold: true
                    color: textColor
                }
                
                Row {
                    spacing: 10
                    
                    Rectangle {
                        width: 30
                        height: 30
                        radius: 15
                        color: accentColor
                        
                        Text {
                            anchors.centerIn: parent
                            text: "🐺"
                            font.pixelSize: 16
                        }
                    }
                    
                    Text {
                        text: recommendedAction
                        font.pixelSize: 18
                        font.bold: true
                        color: accentColor
                        anchors.verticalCenter: parent.verticalCenter
                    }
                }
                
                Text {
                    text: explanation
                    font.pixelSize: 12
                    color: textSecondary
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
            }
        }
        
        Item { Layout.fillHeight: true }
        
        // Action buttons
        RowLayout {
            Layout.fillWidth: true
            spacing: 15
            
            Button {
                text: "Reject"
                Layout.fillWidth: true
                Layout.preferredHeight: 50
                font.pixelSize: 16
                
                background: Rectangle {
                    color: parent.pressed ? "#555555" : "#444444"
                    radius: 8
                    border.color: "#666666"
                    border.width: 1
                }
                
                contentItem: Text {
                    text: parent.text
                    font: parent.font
                    color: textColor
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                
                onClicked: {
                    // Signal rejection
                    permissionDialog.reject()
                }
            }
            
            Button {
                text: "Approve " + recommendedAction
                Layout.fillWidth: true
                Layout.preferredHeight: 50
                font.pixelSize: 16
                font.bold: true
                
                background: Rectangle {
                    color: parent.pressed ? "#3a8eef" : accentColor
                    radius: 8
                }
                
                contentItem: Text {
                    text: parent.text
                    font: parent.font
                    color: "#ffffff"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                
                onClicked: {
                    // Signal approval
                    permissionDialog.accept()
                }
            }
        }
        
        // Alternative action
        TextField {
            Layout.fillWidth: true
            placeholderText: "Alternative action (optional)"
            font.pixelSize: 14
            color: textColor
            
            background: Rectangle {
                color: cardColor
                radius: 5
                border.color: "#444444"
                border.width: 1
            }
        }
    }
}
