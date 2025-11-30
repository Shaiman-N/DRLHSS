import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Page {
    id: settingsPage
    title: "Settings"

    property color primaryColor: "#2196F3"
    property color surfaceColor: "#2d2d2d"
    property color textColor: "#ffffff"
    property color successColor: "#4CAF50"

    ScrollView {
        anchors.fill: parent
        anchors.margins: 20

        Column {
            width: parent.width
            spacing: 20

            // General Settings
            SettingsCard {
                title: "General Settings"
                icon: "⚙️"

                Column {
                    anchors.fill: parent
                    anchors.margins: 20
                    spacing: 15

                    SettingRow {
                        label: "Start with Windows"
                        Switch {
                            checked: true
                            onToggled: console.log("Start with Windows:", checked)
                        }
                    }

                    SettingRow {
                        label: "Minimize to System Tray"
                        Switch {
                            checked: true
                            onToggled: console.log("Minimize to tray:", checked)
                        }
                    }

                    SettingRow {
                        label: "Show Notifications"
                        Switch {
                            checked: true
                            onToggled: console.log("Notifications:", checked)
                        }
                    }

                    SettingRow {
                        label: "Theme"
                        ComboBox {
                            width: 200
                            model: ["Dark", "Light", "Auto"]
                            currentIndex: 0
                        }
                    }
                }
            }

            // Security Settings
            SettingsCard {
                title: "Security Settings"
                icon: "🛡️"

                Column {
                    anchors.fill: parent
                    anchors.margins: 20
                    spacing: 15

                    SettingRow {
                        label: "Real-time Protection"
                        Switch {
                            checked: true
                            onToggled: console.log("Real-time protection:", checked)
                        }
                    }

                    SettingRow {
                        label: "Automatic Scans"
                        Switch {
                            checked: true
                            onToggled: console.log("Auto scans:", checked)
                        }
                    }

                    SettingRow {
                        label: "Scan Schedule"
                        ComboBox {
                            width: 200
                            model: ["Daily", "Weekly", "Monthly"]
                            currentIndex: 0
                        }
                    }

                    SettingRow {
                        label: "Quarantine Suspicious Files"
                        Switch {
                            checked: true
                            onToggled: console.log("Quarantine:", checked)
                        }
                    }
                }
            }

            // Voice Assistant Settings
            SettingsCard {
                title: "Voice Assistant"
                icon: "🎤"

                Column {
                    anchors.fill: parent
                    anchors.margins: 20
                    spacing: 15

                    SettingRow {
                        label: "Enable Voice Commands"
                        Switch {
                            checked: true
                            onToggled: console.log("Voice commands:", checked)
                        }
                    }

                    SettingRow {
                        label: "Wake Word"
                        TextField {
                            width: 200
                            text: "Hey DIREWOLF"
                            placeholderText: "Enter wake word"
                        }
                    }

                    SettingRow {
                        label: "Voice Speed"
                        Slider {
                            width: 200
                            from: 0.5
                            to: 2.0
                            value: 1.0
                            stepSize: 0.1
                        }
                    }

                    SettingRow {
                        label: "Voice Language"
                        ComboBox {
                            width: 200
                            model: ["English (US)", "English (UK)", "Spanish", "French"]
                            currentIndex: 0
                        }
                    }
                }
            }

            // Performance Settings
            SettingsCard {
                title: "Performance"
                icon: "⚡"

                Column {
                    anchors.fill: parent
                    anchors.margins: 20
                    spacing: 15

                    SettingRow {
                        label: "CPU Usage Limit"
                        Slider {
                            width: 200
                            from: 10
                            to: 100
                            value: 50
                            stepSize: 10
                        }
                    }

                    SettingRow {
                        label: "Memory Usage Limit"
                        Slider {
                            width: 200
                            from: 10
                            to: 100
                            value: 50
                            stepSize: 10
                        }
                    }

                    SettingRow {
                        label: "Background Scanning"
                        Switch {
                            checked: true
                            onToggled: console.log("Background scanning:", checked)
                        }
                    }
                }
            }

            // About Section
            SettingsCard {
                title: "About"
                icon: "ℹ️"

                Column {
                    anchors.fill: parent
                    anchors.margins: 20
                    spacing: 15

                    Text {
                        text: "DIREWOLF XAI Security Suite"
                        color: textColor
                        font.pixelSize: 18
                        font.bold: true
                    }

                    Text {
                        text: "Version 1.0.0"
                        color: "#888888"
                        font.pixelSize: 14
                    }

                    Text {
                        text: "AI-Powered Security Assistant"
                        color: textColor
                        font.pixelSize: 14
                    }

                    Button {
                        text: "Check for Updates"
                        background: Rectangle {
                            color: parent.pressed ? primaryColor + "CC" : primaryColor
                            radius: 6
                        }
                        contentItem: Text {
                            text: parent.text
                            color: "white"
                            font.pixelSize: 14
                            horizontalAlignment: Text.AlignHCenter
                        }
                        onClicked: console.log("Checking for updates...")
                    }
                }
            }
        }
    }

    // Settings Card Component
    component SettingsCard: Rectangle {
        property string title: ""
        property string icon: ""
        default property alias content: contentArea.children

        width: parent.width
        height: contentArea.childrenRect.height + 70
        color: surfaceColor
        radius: 12
        border.color: "#404040"
        border.width: 1

        Rectangle {
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            height: 50
            color: "transparent"

            Row {
                anchors.left: parent.left
                anchors.leftMargin: 20
                anchors.verticalCenter: parent.verticalCenter
                spacing: 10

                Text {
                    text: icon
                    font.pixelSize: 20
                    color: primaryColor
                }

                Text {
                    text: title
                    color: textColor
                    font.pixelSize: 16
                    font.bold: true
                }
            }
        }

        Item {
            id: contentArea
            anchors.top: parent.top
            anchors.topMargin: 50
            anchors.left: parent.left
            anchors.right: parent.right
            height: childrenRect.height
        }
    }

    // Setting Row Component
    component SettingRow: Row {
        property string label: ""
        default property alias control: controlArea.children

        width: parent.width
        spacing: 20

        Text {
            text: label
            color: textColor
            font.pixelSize: 14
            width: 200
            anchors.verticalCenter: parent.verticalCenter
        }

        Item {
            id: controlArea
            width: 200
            height: childrenRect.height
        }
    }
}
