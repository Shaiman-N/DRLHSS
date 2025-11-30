import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Page {
    id: systemPage
    title: "System Monitor"

    property color primaryColor: "#2196F3"
    property color surfaceColor: "#2d2d2d"
    property color textColor: "#ffffff"
    property color successColor: "#4CAF50"
    property color warningColor: "#FF9800"
    property color errorColor: "#F44336"

    ScrollView {
        anchors.fill: parent
        anchors.margins: 20

        GridLayout {
            width: parent.width
            columns: 2
            columnSpacing: 20
            rowSpacing: 20

            // System Overview
            SystemCard {
                Layout.columnSpan: 2
                Layout.fillWidth: true
                Layout.preferredHeight: 200
                title: "System Overview"
                icon: "💻"

                Row {
                    anchors.fill: parent
                    anchors.margins: 20
                    spacing: 40

                    Column {
                        width: parent.width / 4
                        spacing: 10

                        Text {
                            text: "Operating System"
                            color: "#888888"
                            font.pixelSize: 12
                        }
                        Text {
                            text: "Windows 11 Pro"
                            color: textColor
                            font.pixelSize: 14
                            font.bold: true
                        }
                        Text {
                            text: "Build 22621.2715"
                            color: "#888888"
                            font.pixelSize: 11
                        }
                    }

                    Column {
                        width: parent.width / 4
                        spacing: 10

                        Text {
                            text: "Processor"
                            color: "#888888"
                            font.pixelSize: 12
                        }
                        Text {
                            text: "Intel Core i7-12700K"
                            color: textColor
                            font.pixelSize: 14
                            font.bold: true
                        }
                        Text {
                            text: "12 Cores, 20 Threads"
                            color: "#888888"
                            font.pixelSize: 11
                        }
                    }

                    Column {
                        width: parent.width / 4
                        spacing: 10

                        Text {
                            text: "Memory"
                            color: "#888888"
                            font.pixelSize: 12
                        }
                        Text {
                            text: "32 GB DDR4"
                            color: textColor
                            font.pixelSize: 14
                            font.bold: true
                        }
                        Text {
                            text: "3200 MHz"
                            color: "#888888"
                            font.pixelSize: 11
                        }
                    }

                    Column {
                        width: parent.width / 4
                        spacing: 10

                        Text {
                            text: "Uptime"
                            color: "#888888"
                            font.pixelSize: 12
                        }
                        Text {
                            text: "2d 14h 32m"
                            color: successColor
                            font.pixelSize: 14
                            font.bold: true
                        }
                        Text {
                            text: "Since last restart"
                            color: "#888888"
                            font.pixelSize: 11
                        }
                    }
                }
            }

            // CPU Usage
            SystemCard {
                Layout.fillWidth: true
                Layout.preferredHeight: 300
                title: "CPU Usage"
                icon: "⚡"

                Column {
                    anchors.fill: parent
                    anchors.margins: 20
                    spacing: 20

                    Row {
                        width: parent.width
                        spacing: 30

                        CircularProgress {
                            value: 0.35
                            size: 120
                            color: primaryColor
                            text: "35%"
                        }

                        Column {
                            spacing: 15
                            anchors.verticalCenter: parent.verticalCenter

                            Row {
                                spacing: 10
                                Text {
                                    text: "Current:"
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Text {
                                    text: "35%"
                                    color: textColor
                                    font.pixelSize: 14
                                    font.bold: true
                                }
                            }

                            Row {
                                spacing: 10
                                Text {
                                    text: "Average:"
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Text {
                                    text: "28%"
                                    color: textColor
                                    font.pixelSize: 14
                                    font.bold: true
                                }
                            }

                            Row {
                                spacing: 10
                                Text {
                                    text: "Peak:"
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Text {
                                    text: "87%"
                                    color: warningColor
                                    font.pixelSize: 14
                                    font.bold: true
                                }
                            }
                        }
                    }

                    // CPU Usage Chart
                    Rectangle {
                        width: parent.width
                        height: 80
                        color: "#1a1a1a"
                        radius: 8
                        border.color: "#404040"
                        border.width: 1

                        Row {
                            anchors.bottom: parent.bottom
                            anchors.bottomMargin: 10
                            anchors.horizontalCenter: parent.horizontalCenter
                            spacing: 2

                            Repeater {
                                model: [25, 30, 35, 40, 35, 30, 28, 32, 35, 38, 35, 33, 30, 28, 25, 30, 35, 40, 45, 35]
                                Rectangle {
                                    width: 8
                                    height: modelData * 1.5
                                    color: primaryColor
                                    radius: 2
                                }
                            }
                        }
                    }
                }
            }

            // Memory Usage
            SystemCard {
                Layout.fillWidth: true
                Layout.preferredHeight: 300
                title: "Memory Usage"
                icon: "🧠"

                Column {
                    anchors.fill: parent
                    anchors.margins: 20
                    spacing: 20

                    Row {
                        width: parent.width
                        spacing: 30

                        CircularProgress {
                            value: 0.68
                            size: 120
                            color: warningColor
                            text: "68%"
                        }

                        Column {
                            spacing: 15
                            anchors.verticalCenter: parent.verticalCenter

                            Row {
                                spacing: 10
                                Text {
                                    text: "Used:"
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Text {
                                    text: "21.8 GB"
                                    color: textColor
                                    font.pixelSize: 14
                                    font.bold: true
                                }
                            }

                            Row {
                                spacing: 10
                                Text {
                                    text: "Available:"
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Text {
                                    text: "10.2 GB"
                                    color: successColor
                                    font.pixelSize: 14
                                    font.bold: true
                                }
                            }

                            Row {
                                spacing: 10
                                Text {
                                    text: "Total:"
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Text {
                                    text: "32.0 GB"
                                    color: textColor
                                    font.pixelSize: 14
                                    font.bold: true
                                }
                            }
                        }
                    }

                    // Memory breakdown
                    Column {
                        width: parent.width
                        spacing: 8

                        Text {
                            text: "Memory Breakdown"
                            color: textColor
                            font.pixelSize: 14
                            font.bold: true
                        }

                        MemoryBar {
                            label: "System"
                            value: 0.25
                            color: primaryColor
                            text: "8.0 GB"
                        }

                        MemoryBar {
                            label: "Applications"
                            value: 0.35
                            color: warningColor
                            text: "11.2 GB"
                        }

                        MemoryBar {
                            label: "Cache"
                            value: 0.08
                            color: successColor
                            text: "2.6 GB"
                        }
                    }
                }
            }

            // Disk Usage
            SystemCard {
                Layout.columnSpan: 2
                Layout.fillWidth: true
                Layout.preferredHeight: 250
                title: "Disk Usage"
                icon: "💾"

                Row {
                    anchors.fill: parent
                    anchors.margins: 20
                    spacing: 30

                    // C: Drive
                    Column {
                        width: parent.width / 3
                        spacing: 15

                        Text {
                            text: "C: Drive (System)"
                            color: textColor
                            font.pixelSize: 14
                            font.bold: true
                        }

                        CircularProgress {
                            value: 0.42
                            size: 100
                            color: successColor
                            text: "42%"
                        }

                        Column {
                            spacing: 5
                            Text {
                                text: "Used: 210 GB"
                                color: "#888888"
                                font.pixelSize: 12
                            }
                            Text {
                                text: "Free: 290 GB"
                                color: textColor
                                font.pixelSize: 12
                            }
                            Text {
                                text: "Total: 500 GB"
                                color: "#888888"
                                font.pixelSize: 12
                            }
                        }
                    }

                    // D: Drive
                    Column {
                        width: parent.width / 3
                        spacing: 15

                        Text {
                            text: "D: Drive (Data)"
                            color: textColor
                            font.pixelSize: 14
                            font.bold: true
                        }

                        CircularProgress {
                            value: 0.75
                            size: 100
                            color: warningColor
                            text: "75%"
                        }

                        Column {
                            spacing: 5
                            Text {
                                text: "Used: 750 GB"
                                color: "#888888"
                                font.pixelSize: 12
                            }
                            Text {
                                text: "Free: 250 GB"
                                color: textColor
                                font.pixelSize: 12
                            }
                            Text {
                                text: "Total: 1 TB"
                                color: "#888888"
                                font.pixelSize: 12
                            }
                        }
                    }

                    // Network Drive
                    Column {
                        width: parent.width / 3
                        spacing: 15

                        Text {
                            text: "N: Drive (Network)"
                            color: textColor
                            font.pixelSize: 14
                            font.bold: true
                        }

                        CircularProgress {
                            value: 0.28
                            size: 100
                            color: primaryColor
                            text: "28%"
                        }

                        Column {
                            spacing: 5
                            Text {
                                text: "Used: 560 GB"
                                color: "#888888"
                                font.pixelSize: 12
                            }
                            Text {
                                text: "Free: 1.44 TB"
                                color: textColor
                                font.pixelSize: 12
                            }
                            Text {
                                text: "Total: 2 TB"
                                color: "#888888"
                                font.pixelSize: 12
                            }
                        }
                    }
                }
            }

            // Process List
            SystemCard {
                Layout.columnSpan: 2
                Layout.fillWidth: true
                Layout.preferredHeight: 300
                title: "Top Processes"
                icon: "📋"

                ListView {
                    anchors.fill: parent
                    anchors.margins: 20
                    model: ListModel {
                        ListElement {
                            name: "direwolf_gui.exe"
                            cpu: "12.5"
                            memory: "245 MB"
                            status: "Running"
                        }
                        ListElement {
                            name: "chrome.exe"
                            cpu: "8.2"
                            memory: "1.2 GB"
                            status: "Running"
                        }
                        ListElement {
                            name: "System"
                            cpu: "5.1"
                            memory: "156 MB"
                            status: "Running"
                        }
                        ListElement {
                            name: "explorer.exe"
                            cpu: "2.3"
                            memory: "89 MB"
                            status: "Running"
                        }
                        ListElement {
                            name: "svchost.exe"
                            cpu: "1.8"
                            memory: "67 MB"
                            status: "Running"
                        }
                    }

                    header: Row {
                        width: parent.width
                        spacing: 20
                        padding: 10

                        Text {
                            text: "Process"
                            color: textColor
                            font.pixelSize: 13
                            font.bold: true
                            width: parent.width * 0.4
                        }
                        Text {
                            text: "CPU"
                            color: textColor
                            font.pixelSize: 13
                            font.bold: true
                            width: parent.width * 0.2
                        }
                        Text {
                            text: "Memory"
                            color: textColor
                            font.pixelSize: 13
                            font.bold: true
                            width: parent.width * 0.2
                        }
                        Text {
                            text: "Status"
                            color: textColor
                            font.pixelSize: 13
                            font.bold: true
                            width: parent.width * 0.2
                        }
                    }

                    delegate: Rectangle {
                        width: parent.width
                        height: 40
                        color: "transparent"
                        border.color: "#404040"
                        border.width: 1
                        radius: 4

                        Row {
                            anchors.fill: parent
                            anchors.margins: 10
                            spacing: 20

                            Text {
                                text: model.name
                                color: textColor
                                font.pixelSize: 12
                                width: parent.width * 0.4
                            }
                            Text {
                                text: model.cpu + "%"
                                color: primaryColor
                                font.pixelSize: 12
                                width: parent.width * 0.2
                            }
                            Text {
                                text: model.memory
                                color: textColor
                                font.pixelSize: 12
                                width: parent.width * 0.2
                            }
                            Text {
                                text: model.status
                                color: successColor
                                font.pixelSize: 12
                                width: parent.width * 0.2
                            }
                        }
                    }
                }
            }
        }
    }

    // System Card Component
    component SystemCard: Rectangle {
        property string title: ""
        property string icon: ""
        default property alias content: contentArea.children

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
            anchors.bottom: parent.bottom
        }
    }

    // Circular Progress Component
    component CircularProgress: Item {
        property real value: 0.0
        property int size: 100
        property color color: primaryColor
        property string text: ""

        width: size
        height: size

        Canvas {
            anchors.fill: parent
            onPaint: {
                var ctx = getContext("2d")
                ctx.clearRect(0, 0, width, height)
                
                var centerX = width / 2
                var centerY = height / 2
                var radius = Math.min(width, height) / 2 - 10
                
                // Background circle
                ctx.beginPath()
                ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI)
                ctx.strokeStyle = "#404040"
                ctx.lineWidth = 8
                ctx.stroke()
                
                // Progress arc
                ctx.beginPath()
                ctx.arc(centerX, centerY, radius, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * value)
                ctx.strokeStyle = color
                ctx.lineWidth = 8
                ctx.lineCap = "round"
                ctx.stroke()
            }
        }

        Text {
            anchors.centerIn: parent
            text: parent.text
            color: textColor
            font.pixelSize: 18
            font.bold: true
        }
    }

    // Memory Bar Component
    component MemoryBar: Row {
        property string label: ""
        property real value: 0.0
        property color color: primaryColor
        property string text: ""

        width: parent.width
        spacing: 10

        Text {
            text: label
            color: "#888888"
            font.pixelSize: 12
            width: 100
            anchors.verticalCenter: parent.verticalCenter
        }

        Rectangle {
            width: parent.width - 200
            height: 20
            color: "#404040"
            radius: 10
            anchors.verticalCenter: parent.verticalCenter

            Rectangle {
                width: parent.width * value
                height: parent.height
                color: parent.parent.color
                radius: 10
            }
        }

        Text {
            text: parent.text
            color: textColor
            font.pixelSize: 12
            width: 80
            anchors.verticalCenter: parent.verticalCenter
        }
    }
}
