#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QIcon>
#include "UI/GUIBackend.hpp"

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    
    // Set application metadata
    app.setApplicationName("DIREWOLF XAI");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("DIREWOLF Security");
    app.setOrganizationDomain("direwolf.security");
    
    // Create QML engine
    QQmlApplicationEngine engine;
    
    // Create and register backend
    DIREWOLF::UI::GUIBackend backend;
    engine.rootContext()->setContextProperty("backend", &backend);
    
    // Load main QML file
    const QUrl url(QStringLiteral("qrc:/qml/main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl)
            QCoreApplication::exit(-1);
    }, Qt::QueuedConnection);
    
    engine.load(url);
    
    if (engine.rootObjects().isEmpty())
        return -1;
    
    return app.exec();
}
