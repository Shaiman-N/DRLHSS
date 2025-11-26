#include "sandboxPos_FP/OverlayManager.hpp"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstdlib>

namespace sandboxPos_FP {

OverlayManager::OverlayManager() {}

std::string OverlayManager::createOverlay(const std::string& baseImage,
                                         const std::string& artifactPath,
                                         const std::map<std::string, std::string>& hostProfile) {
    namespace fs = std::filesystem;

    const std::string upper = "/tmp/sandbox-upper";
    const std::string work = "/tmp/sandbox-work";
    const std::string rootfs = "/tmp/sandbox-rootfs";

    fs::create_directories(upper);
    fs::create_directories(work);
    fs::create_directories(rootfs);

    std::string mountCmd = "mount -t overlay overlay -o lowerdir=" + baseImage +
                           ",upperdir=" + upper + ",workdir=" + work + " " + rootfs;

    if (system(mountCmd.c_str()) != 0) {
        throw std::runtime_error("[OverlayManager] Overlay mount failed");
    }

    // Copy artifact into upper layer for execution
    fs::copy_file(artifactPath, upper + "/artifact", fs::copy_options::overwrite_existing);

    std::cout << "[OverlayManager] Overlay rootfs prepared." << std::endl;

    return rootfs;
}

void OverlayManager::patchOverlay(const std::string& overlayPath,
                                  const std::map<std::string, std::string>& patches) {
    namespace fs = std::filesystem;

    for (const auto& [fileRelPath, content] : patches) {
        std::string fullPath = overlayPath + "/" + fileRelPath;
        std::ofstream ofs(fullPath);
        if (ofs.is_open()) {
            ofs << content;
            ofs.close();
            std::cout << "[OverlayManager] Patched " << fileRelPath << std::endl;
        } else {
            std::cerr << "[OverlayManager] Failed to write patch " << fileRelPath << std::endl;
        }
    }
}

void OverlayManager::cleanupOverlay(const std::string& overlayPath) {
    std::string unmountCmd = "umount " + overlayPath;
    system(unmountCmd.c_str());
    std::filesystem::remove_all("/tmp/sandbox-upper");
    std::filesystem::remove_all("/tmp/sandbox-work");
    std::filesystem::remove_all("/tmp/sandbox-rootfs");
    std::cout << "[OverlayManager] Cleaned up overlay environment." << std::endl;
}

} // namespace sandbox1