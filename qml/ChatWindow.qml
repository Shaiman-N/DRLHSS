/**
 * ChatWindow.qml
 * DIREWOLF Chat Interface
 * 
 * Text conversation with Wolf, voice activation, and conversation history.
 */

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    id: chatWindow
    width: 700
    height: 600
    minimumWidth: 500
    minimumHeight: 400
    title: "Chat with Wolf"
    
    // Color scheme
    readonly property color bgColor: "#1a1a1a"
    readonly property color cardColor: "#2a2a2a"
    readonly property color accentColor: "#4a9eff"
    readonly property color wolfMessageColor: "#2a4a6a"
    readonly property color alphaMessageColor: "#3a3a3a"
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
            spacing: 15
            
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
            
            ColumnLayout {
                spacing: 2
                
                Text {
                    text: "DIREWOLF"
                    font.pixelSize: 16
                    font.bold: true
                    color: textColor
                }
                
                Row {
                    spacing: 5
                    
                    Rectangle {
                        width: 8
                        height: 8
                        radius: 4
                        color: "#4aff4a"
                        anchors.verticalCenter: parent.verticalCenter
                    }
                    
                    Text {
                        text: "Online"
                        font.pixelSize: 12
                        color: textSecondary
                    }
                }
            }
            
            Item { Layout.fillWidth: true }
            
            // Voice button
            Button {
                width: 40
                height: 40
                
                background: Rectangle {
                    color: parent.pressed ? "#3a8eef" : accentColor
                    radius: 20
                }
                
                contentItem: Text {
                    text: "🎤"
                    font.pixelSize: 20
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                
                ToolTip.visible: hovered
                ToolTip.text: "Voice input"
            }
        }
    }
    
    // Main content
    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        
        // Chat messages
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: bgColor
            
            ListView {
                id: messageList
                anchors.fill: parent
                anchors.margins: 20
                spacing: 15
                clip: true
                
                model: ListModel {
                    id: messageModel
                    
                    ListElement {
                        sender: "wolf"
                        message: "Alpha, your network is secure. I've been monitoring for 2 hours with no threats detected."
                        timestamp: "14:23"
                    }
                    ListElement {
                        sender: "alpha"
                        message: "What's the security status?"
                        timestamp: "14:25"
                    }
                    ListElement {
                        sender: "wolf"
                        message: "All systems operational, Alpha. Antivirus: RUNNING, NIDPS: RUNNING, DRL Agent: 94% confidence. 12 threats blocked today."
                        timestamp: "14:25"
                    }
                }
                
                delegate: ChatMessage {
                    width: ListView.view.width
                    sender: model.sender
                    message: model.message
                    timestamp: model.timestamp
                }
                
                // Auto-scroll to bottom
                onCountChanged: {
                    positionViewAtEnd()
                }
            }
        }
        
        // Typing indicator
        Rectangle {
            Layout.fillWidth: true
            height: wolfTyping ? 40 : 0
            color: bgColor
            visible: wolfTyping
            
            property bool wolfTyping: false
            
            Row {
                anchors.centerIn: parent
                spacing: 10
                
                Text {
                    text: "🐺"
                    font.pixelSize: 16
                }
                
                Text {
                    text: "Wolf is typing..."
                    font.pixelSize: 14
                    color: textSecondary
                    font.italic: true
                }
                
                // Animated dots
                Row {
                    spacing: 5
                    
                    Repeater {
                        model: 3
                        
                        Rectangle {
                            width: 6
                            height: 6
                            radius: 3
                            color: textSecondary
                            
                            SequentialAnimation on opacity {
                                running: parent.parent.parent.parent.wolfTyping
                                loops: Animation.Infinite
                                
                                PauseAnimation { duration: index * 200 }
                                NumberAnimation { from: 0.3; to: 1.0; duration: 400 }
                                NumberAnimation { from: 1.0; to: 0.3; duration: 400 }
                            }
                        }
                    }
                }
            }
        }
        
        // Input area
        Rectangle {
            Layout.fillWidth: true
            height: 80
            color: cardColor
            
            RowLayout {
                anchors.fill: parent
                anchors.margins: 15
                spacing: 15
                
                TextField {
                    id: messageInput
                    Layout.fillWidth: true
                    placeholderText: "Type your message to Wolf..."
                    font.pixelSize: 14
                    color: textColor
                    
                    background: Rectangle {
                        color: "#333333"
                        radius: 8
                        border.color: messageInput.activeFocus ? accentColor : "#444444"
                        border.width: 2
                    }
                    
                    Keys.onReturnPressed: {
                        sendMessage()
                    }
                }
                
                Button {
                    text: "Send"
                    Layout.preferredWidth: 80
                    Layout.preferredHeight: 40
                    font.pixelSize: 14
                    enabled: messageInput.text.length > 0
                    
                    background: Rectangle {
                        color: parent.enabled ? (parent.pressed ? "#3a8eef" : accentColor) : "#444444"
                        radius: 8
                    }
                    
                    contentItem: Text {
                        text: parent.text
                        font: parent.font
                        color: parent.enabled ? "#ffffff" : "#888888"
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    
                    onClicked: {
                        sendMessage()
                    }
                }
            }
        }
    }
    
    // Functions
    function sendMessage() {
        if (messageInput.text.trim().length === 0) {
            return
        }
        
        // Add user message
        messageModel.append({
            sender: "alpha",
            message: messageInput.text,
            timestamp: Qt.formatTime(new Date(), "hh:mm")
        })
        
        // Clear input
        var userMessage = messageInput.text
        messageInput.text = ""
        
        // Show typing indicator
        messageList.parent.wolfTyping = true
        
        // Simulate Wolf's response (in production, would call backend)
        typingTimer.start()
    }
    
    Timer {
        id: typingTimer
        interval: 1500
        onTriggered: {
            // Hide typing indicator
            messageList.parent.wolfTyping = false
            
            // Add Wolf's response
            messageModel.append({
                sender: "wolf",
                message: "I understand, Alpha. Let me check that for you...",
                timestamp: Qt.formatTime(new Date(), "hh:mm")
            })
        }
    }
}

// Chat message component
component ChatMessage: Item {
    property string sender: "wolf"
    property string message: ""
    property string timestamp: ""
    
    height: messageContent.height + 20
    
    Row {
        anchors.fill: parent
        spacing: 10
        layoutDirection: sender === "alpha" ? Qt.RightToLeft : Qt.LeftToRight
        
        // Avatar
        Rectangle {
            width: 40
            height: 40
            radius: 20
            color: sender === "wolf" ? "#2a4a6a" : "#3a3a3a"
            anchors.top: parent.top
            
            Text {
                anchors.centerIn: parent
                text: sender === "wolf" ? "🐺" : "👤"
                font.pixelSize: 20
            }
        }
        
        // Message bubble
        Rectangle {
            id: messageContent
            width: Math.min(parent.width * 0.7, messageText.implicitWidth + 30)
            height: messageText.implicitHeight + 30
            color: sender === "wolf" ? "#2a4a6a" : "#3a3a3a"
            radius: 12
            
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 15
                spacing: 5
                
                Text {
                    id: messageText
                    text: message
                    font.pixelSize: 14
                    color: "#ffffff"
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
                
                Text {
                    text: timestamp
                    font.pixelSize: 10
                    color: "#aaaaaa"
                    Layout.alignment: Qt.AlignRight
                }
            }
        }
    }
}
