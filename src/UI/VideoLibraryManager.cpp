/**
 * @file VideoLibraryManager.cpp
 * @brief Video Library Manager Implementation
 */

#include "UI/VideoLibraryManager.hpp"
#include <QSqlQuery>
#include <QSqlError>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QCryptographicHash>
#include <QProcess>
#include <QJsonDocument>
#include <QJsonObject>
#include <QDateTime>
#include <QDebug>

namespace ui {

VideoLibraryManager::VideoLibraryManager(QObject* parent)
    : QObject(parent)
    , max_thumbnail_size_(320)
    , default_thumbnail_format_("jpg")
{
}

VideoLibraryManager::~VideoLibraryManager() {
    if (database_.isOpen()) {
        database_.close();
    }
}

bool VideoLibraryManager::initialize(const QString& library_path) {
    library_path_ = library_path;
    thumbnails_path_ = library_path + "/thumbnails";
    
    // Create directories
    QDir dir;
    if (!dir.mkpath(library_path_)) {
        qWarning() << "Failed to create library directory:" << library_path_;
        return false;
    }
    
    if (!dir.mkpath(thumbnails_path_)) {
        qWarning() << "Failed to create thumbnails directory:" << thumbnails_path_;
        return false;
    }
    
    // Initialize database
    if (!initializeDatabase()) {
        qWarning() << "Failed to initialize database";
        return false;
    }
    
    qInfo() << "Video library initialized at:" << library_path_;
    return true;
}

QString VideoLibraryManager::addVideo(const QString& video_path, const VideoMetadata& metadata) {
    QString video_id = generateVideoId();
    
    // Copy video to library
    QString library_video_path = copyVideoToLibrary(video_path, video_id);
    if (library_video_path.isEmpty()) {
        qWarning() << "Failed to copy video to library";
        return QString();
    }
    
    // Get video information
    qint64 file_size = getFileSize(library_video_path);
    int duration = getVideoDuration(library_video_path);
    
    // Insert into database
    QSqlQuery query(database_);
    query.prepare(
        "INSERT INTO videos (id, title, description, file_path, created_date, "
        "incident_id, video_type, quality, format, file_size, duration_seconds) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    );
    
    query.addBindValue(video_id);
    query.addBindValue(metadata.title);
    query.addBindValue(metadata.description);
    query.addBindValue(library_video_path);
    query.addBindValue(QDateTime::currentDateTime());
    query.addBindValue(metadata.incident_id);
    query.addBindValue(metadata.video_type);
    query.addBindValue(metadata.quality);
    query.addBindValue(metadata.format);
    query.addBindValue(file_size);
    query.addBindValue(duration);
    
    if (!query.exec()) {
        qWarning() << "Failed to insert video:" << query.lastError().text();
        return QString();
    }
    
    // Add tags
    for (const QString& tag : metadata.tags) {
        QSqlQuery tag_query(database_);
        tag_query.prepare("INSERT INTO video_tags (video_id, tag) VALUES (?, ?)");
        tag_query.addBindValue(video_id);
        tag_query.addBindValue(tag);
        tag_query.exec();
    }
    
    // Generate thumbnail
    generateThumbnail(video_id, 5);
    
    emit videoAdded(video_id);
    
    qInfo() << "Video added to library:" << video_id;
    return video_id;
}

bool VideoLibraryManager::removeVideo(const QString& video_id) {
    // Get video path
    VideoMetadata metadata = getVideoMetadata(video_id);
    
    // Delete from database
    QSqlQuery query(database_);
    query.prepare("DELETE FROM videos WHERE id = ?");
    query.addBindValue(video_id);
    
    if (!query.exec()) {
        qWarning() << "Failed to delete video from database:" << query.lastError().text();
        return false;
    }
    
    // Delete tags
    QSqlQuery tag_query(database_);
    tag_query.prepare("DELETE FROM video_tags WHERE video_id = ?");
    tag_query.addBindValue(video_id);
    tag_query.exec();
    
    // Delete files
    deleteVideoFile(video_id);
    
    emit videoRemoved(video_id);
    
    qInfo() << "Video removed from library:" << video_id;
    return true;
}

bool VideoLibraryManager::updateVideoMetadata(const QString& video_id, const VideoMetadata& metadata) {
    QSqlQuery query(database_);
    query.prepare(
        "UPDATE videos SET title = ?, description = ?, modified_date = ?, "
        "incident_id = ?, video_type = ? WHERE id = ?"
    );
    
    query.addBindValue(metadata.title);
    query.addBindValue(metadata.description);
    query.addBindValue(QDateTime::currentDateTime());
    query.addBindValue(metadata.incident_id);
    query.addBindValue(metadata.video_type);
    query.addBindValue(video_id);
    
    if (!query.exec()) {
        qWarning() << "Failed to update video metadata:" << query.lastError().text();
        return false;
    }
    
    // Update tags
    QSqlQuery delete_tags(database_);
    delete_tags.prepare("DELETE FROM video_tags WHERE video_id = ?");
    delete_tags.addBindValue(video_id);
    delete_tags.exec();
    
    for (const QString& tag : metadata.tags) {
        QSqlQuery tag_query(database_);
        tag_query.prepare("INSERT INTO video_tags (video_id, tag) VALUES (?, ?)");
        tag_query.addBindValue(video_id);
        tag_query.addBindValue(tag);
        tag_query.exec();
    }
    
    return true;
}

VideoMetadata VideoLibraryManager::getVideoMetadata(const QString& video_id) {
    VideoMetadata metadata;
    
    QSqlQuery query(database_);
    query.prepare("SELECT * FROM videos WHERE id = ?");
    query.addBindValue(video_id);
    
    if (query.exec() && query.next()) {
        metadata.id = query.value("id").toString();
        metadata.title = query.value("title").toString();
        metadata.description = query.value("description").toString();
        metadata.file_path = query.value("file_path").toString();
        metadata.thumbnail_path = query.value("thumbnail_path").toString();
        metadata.created_date = query.value("created_date").toDateTime();
        metadata.modified_date = query.value("modified_date").toDateTime();
        metadata.incident_id = query.value("incident_id").toString();
        metadata.video_type = query.value("video_type").toString();
        metadata.quality = query.value("quality").toString();
        metadata.format = query.value("format").toString();
        metadata.file_size = query.value("file_size").toLongLong();
        metadata.duration_seconds = query.value("duration_seconds").toInt();
        metadata.is_shared = query.value("is_shared").toBool();
        metadata.share_url = query.value("share_url").toString();
        
        // Get tags
        QSqlQuery tag_query(database_);
        tag_query.prepare("SELECT tag FROM video_tags WHERE video_id = ?");
        tag_query.addBindValue(video_id);
        
        if (tag_query.exec()) {
            while (tag_query.next()) {
                metadata.tags.append(tag_query.value(0).toString());
            }
        }
    }
    
    return metadata;
}

std::vector<VideoMetadata> VideoLibraryManager::getAllVideos() {
    std::vector<VideoMetadata> videos;
    
    QSqlQuery query("SELECT id FROM videos ORDER BY created_date DESC", database_);
    
    while (query.next()) {
        QString video_id = query.value(0).toString();
        videos.push_back(getVideoMetadata(video_id));
    }
    
    return videos;
}

std::vector<VideoMetadata> VideoLibraryManager::searchByTitle(const QString& query) {
    std::vector<VideoMetadata> videos;
    
    QSqlQuery sql_query(database_);
    sql_query.prepare("SELECT id FROM videos WHERE title LIKE ? ORDER BY created_date DESC");
    sql_query.addBindValue("%" + query + "%");
    
    if (sql_query.exec()) {
        while (sql_query.next()) {
            QString video_id = sql_query.value(0).toString();
            videos.push_back(getVideoMetadata(video_id));
        }
    }
    
    return videos;
}

std::vector<VideoMetadata> VideoLibraryManager::filterByType(const QString& video_type) {
    std::vector<VideoMetadata> videos;
    
    QSqlQuery query(database_);
    query.prepare("SELECT id FROM videos WHERE video_type = ? ORDER BY created_date DESC");
    query.addBindValue(video_type);
    
    if (query.exec()) {
        while (query.next()) {
            QString video_id = query.value(0).toString();
            videos.push_back(getVideoMetadata(video_id));
        }
    }
    
    return videos;
}

std::vector<VideoMetadata> VideoLibraryManager::filterByDateRange(
    const QDateTime& start_date,
    const QDateTime& end_date)
{
    std::vector<VideoMetadata> videos;
    
    QSqlQuery query(database_);
    query.prepare(
        "SELECT id FROM videos WHERE created_date BETWEEN ? AND ? "
        "ORDER BY created_date DESC"
    );
    query.addBindValue(start_date);
    query.addBindValue(end_date);
    
    if (query.exec()) {
        while (query.next()) {
            QString video_id = query.value(0).toString();
            videos.push_back(getVideoMetadata(video_id));
        }
    }
    
    return videos;
}

std::vector<VideoMetadata> VideoLibraryManager::filterByTags(const QStringList& tags) {
    std::vector<VideoMetadata> videos;
    
    if (tags.isEmpty()) {
        return videos;
    }
    
    QString placeholders = QString("?,").repeated(tags.size());
    placeholders.chop(1);
    
    QSqlQuery query(database_);
    query.prepare(
        QString("SELECT DISTINCT video_id FROM video_tags WHERE tag IN (%1)").arg(placeholders)
    );
    
    for (const QString& tag : tags) {
        query.addBindValue(tag);
    }
    
    if (query.exec()) {
        while (query.next()) {
            QString video_id = query.value(0).toString();
            videos.push_back(getVideoMetadata(video_id));
        }
    }
    
    return videos;
}

bool VideoLibraryManager::generateThumbnail(const QString& video_id, int timestamp) {
    VideoMetadata metadata = getVideoMetadata(video_id);
    
    if (metadata.file_path.isEmpty()) {
        return false;
    }
    
    QString thumbnail_path = generateThumbnailPath(video_id);
    
    if (extractThumbnailFromVideo(metadata.file_path, thumbnail_path, timestamp)) {
        // Update database
        QSqlQuery query(database_);
        query.prepare("UPDATE videos SET thumbnail_path = ? WHERE id = ?");
        query.addBindValue(thumbnail_path);
        query.addBindValue(video_id);
        query.exec();
        
        emit thumbnailGenerated(video_id);
        return true;
    }
    
    return false;
}

QPixmap VideoLibraryManager::getThumbnail(const QString& video_id) {
    VideoMetadata metadata = getVideoMetadata(video_id);
    
    if (!metadata.thumbnail_path.isEmpty() && QFile::exists(metadata.thumbnail_path)) {
        return QPixmap(metadata.thumbnail_path);
    }
    
    // Return default thumbnail
    return QPixmap();
}

void VideoLibraryManager::regenerateAllThumbnails() {
    auto videos = getAllVideos();
    
    for (const auto& video : videos) {
        generateThumbnail(video.id, 5);
    }
}

QString VideoLibraryManager::shareVideo(const QString& video_id, const QString& share_method) {
    VideoMetadata metadata = getVideoMetadata(video_id);
    
    if (share_method == "link") {
        // Generate shareable link
        QString share_url = QString("direwolf://video/%1").arg(video_id);
        
        QSqlQuery query(database_);
        query.prepare("UPDATE videos SET is_shared = 1, share_url = ? WHERE id = ?");
        query.addBindValue(share_url);
        query.addBindValue(video_id);
        query.exec();
        
        return share_url;
    }
    
    return QString();
}

bool VideoLibraryManager::exportVideo(const QString& video_id, const QString& export_path) {
    VideoMetadata metadata = getVideoMetadata(video_id);
    
    if (metadata.file_path.isEmpty()) {
        return false;
    }
    
    // Copy video file
    QString dest_video = export_path + "/" + QFileInfo(metadata.file_path).fileName();
    if (!QFile::copy(metadata.file_path, dest_video)) {
        return false;
    }
    
    // Export metadata as JSON
    QJsonObject json;
    json["id"] = metadata.id;
    json["title"] = metadata.title;
    json["description"] = metadata.description;
    json["incident_id"] = metadata.incident_id;
    json["video_type"] = metadata.video_type;
    json["quality"] = metadata.quality;
    json["format"] = metadata.format;
    json["duration_seconds"] = metadata.duration_seconds;
    json["created_date"] = metadata.created_date.toString(Qt::ISODate);
    
    QJsonDocument doc(json);
    QString metadata_path = export_path + "/" + metadata.id + "_metadata.json";
    
    QFile metadata_file(metadata_path);
    if (metadata_file.open(QIODevice::WriteOnly)) {
        metadata_file.write(doc.toJson());
        metadata_file.close();
    }
    
    return true;
}

QVariantMap VideoLibraryManager::getLibraryStatistics() {
    QVariantMap stats;
    
    QSqlQuery query(database_);
    
    // Total videos
    query.exec("SELECT COUNT(*) FROM videos");
    if (query.next()) {
        stats["total_videos"] = query.value(0).toInt();
    }
    
    // Total storage
    query.exec("SELECT SUM(file_size) FROM videos");
    if (query.next()) {
        stats["total_storage"] = query.value(0).toLongLong();
    }
    
    // Videos by type
    query.exec("SELECT video_type, COUNT(*) FROM videos GROUP BY video_type");
    QVariantMap by_type;
    while (query.next()) {
        by_type[query.value(0).toString()] = query.value(1).toInt();
    }
    stats["by_type"] = by_type;
    
    return stats;
}

qint64 VideoLibraryManager::getStorageUsage() {
    QSqlQuery query("SELECT SUM(file_size) FROM videos", database_);
    
    if (query.next()) {
        return query.value(0).toLongLong();
    }
    
    return 0;
}

int VideoLibraryManager::cleanupOldVideos(int days_old) {
    QDateTime cutoff_date = QDateTime::currentDateTime().addDays(-days_old);
    
    QSqlQuery query(database_);
    query.prepare("SELECT id FROM videos WHERE created_date < ?");
    query.addBindValue(cutoff_date);
    
    int count = 0;
    if (query.exec()) {
        while (query.next()) {
            QString video_id = query.value(0).toString();
            if (removeVideo(video_id)) {
                count++;
            }
        }
    }
    
    return count;
}

bool VideoLibraryManager::initializeDatabase() {
    database_ = QSqlDatabase::addDatabase("QSQLITE", "video_library");
    database_.setDatabaseName(library_path_ + "/library.db");
    
    if (!database_.open()) {
        qWarning() << "Failed to open database:" << database_.lastError().text();
        return false;
    }
    
    return createTables();
}

bool VideoLibraryManager::createTables() {
    QSqlQuery query(database_);
    
    // Videos table
    if (!query.exec(
        "CREATE TABLE IF NOT EXISTS videos ("
        "id TEXT PRIMARY KEY, "
        "title TEXT, "
        "description TEXT, "
        "file_path TEXT, "
        "thumbnail_path TEXT, "
        "created_date DATETIME, "
        "modified_date DATETIME, "
        "incident_id TEXT, "
        "video_type TEXT, "
        "quality TEXT, "
        "format TEXT, "
        "file_size INTEGER, "
        "duration_seconds INTEGER, "
        "is_shared INTEGER DEFAULT 0, "
        "share_url TEXT"
        ")"
    )) {
        qWarning() << "Failed to create videos table:" << query.lastError().text();
        return false;
    }
    
    // Tags table
    if (!query.exec(
        "CREATE TABLE IF NOT EXISTS video_tags ("
        "video_id TEXT, "
        "tag TEXT, "
        "FOREIGN KEY(video_id) REFERENCES videos(id)"
        ")"
    )) {
        qWarning() << "Failed to create video_tags table:" << query.lastError().text();
        return false;
    }
    
    return true;
}

QString VideoLibraryManager::copyVideoToLibrary(const QString& source_path, const QString& video_id) {
    QFileInfo source_info(source_path);
    QString dest_path = library_path_ + "/" + video_id + "." + source_info.suffix();
    
    if (QFile::copy(source_path, dest_path)) {
        return dest_path;
    }
    
    return QString();
}

bool VideoLibraryManager::deleteVideoFile(const QString& video_id) {
    VideoMetadata metadata = getVideoMetadata(video_id);
    
    // Delete video file
    if (!metadata.file_path.isEmpty()) {
        QFile::remove(metadata.file_path);
    }
    
    // Delete thumbnail
    if (!metadata.thumbnail_path.isEmpty()) {
        QFile::remove(metadata.thumbnail_path);
    }
    
    return true;
}

QString VideoLibraryManager::generateThumbnailPath(const QString& video_id) {
    return thumbnails_path_ + "/" + video_id + "." + default_thumbnail_format_;
}

bool VideoLibraryManager::extractThumbnailFromVideo(
    const QString& video_path,
    const QString& thumbnail_path,
    int timestamp)
{
    QProcess ffmpeg;
    QStringList args;
    
    args << "-i" << video_path
         << "-ss" << QString::number(timestamp)
         << "-vframes" << "1"
         << "-vf" << QString("scale=%1:-1").arg(max_thumbnail_size_)
         << "-y"
         << thumbnail_path;
    
    ffmpeg.start("ffmpeg", args);
    ffmpeg.waitForFinished(10000);
    
    return ffmpeg.exitCode() == 0 && QFile::exists(thumbnail_path);
}

QString VideoLibraryManager::generateVideoId() {
    QString timestamp = QString::number(QDateTime::currentMSecsSinceEpoch());
    QByteArray hash = QCryptographicHash::hash(timestamp.toUtf8(), QCryptographicHash::Md5);
    return hash.toHex().left(16);
}

qint64 VideoLibraryManager::getFileSize(const QString& file_path) {
    QFileInfo info(file_path);
    return info.size();
}

int VideoLibraryManager::getVideoDuration(const QString& video_path) {
    // Use ffprobe to get duration
    QProcess ffprobe;
    QStringList args;
    
    args << "-v" << "error"
         << "-show_entries" << "format=duration"
         << "-of" << "default=noprint_wrappers=1:nokey=1"
         << video_path;
    
    ffprobe.start("ffprobe", args);
    ffprobe.waitForFinished(5000);
    
    QString output = ffprobe.readAllStandardOutput();
    return output.trimmed().toInt();
}

} // namespace ui
