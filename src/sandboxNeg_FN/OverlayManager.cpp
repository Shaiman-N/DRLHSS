#include "sandboxNeg_FN/OverlayManager.hpp"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstdlib>

namespace sandboxNeg_FN {

OverlayManager::OverlayManager() {}

std::string OverlayManager::createOverlay(const std::string& baseImage,
                                         const std::string& artifactPath,
                                         const std::map<std::string, std::string>& /*hostProfile*/) {
    namespace fs = std::filesystem;

    const std::string upper = "/tmp/sandboxNeg_FN-upper";
    const std::string work = "/tmp/sandboxNeg_FN-work";
    const std::string rootfs = "/tmp/sandboxNeg_FN-rootfs";

    fs::create_directories(upper);
    fs::create_directories(work);
    fs::create_directories(rootfs);

    std::string mountCmd = "mount -t overlay overlay -o lowerdir=" + baseImage +
                           ",upperdir=" + upper + ",workdir=" + work + " " + rootfs;

    if(system(mountCmd.c_str()) != 0) {
        throw std::runtime_error("[sandboxNeg_FN::OverlayManager] Overlay mount failed");
    }

    fs::copy_file(artifactPath, upper + "/artifact", fs::copy_options::overwrite_existing);

    std::cout << "[sandboxNeg_FN::OverlayManager] Overlay prepared." << std::endl;
    return rootfs;
}

void OverlayManager::patchOverlay(const std::string& overlayPath,
                                  const std::map<std::string, std::string>& patches) {
    std::ofstream ofs;
    for (const auto& [filePath, content]: patches) {
        std::string fullPath = overlayPath + "/" + filePath;
        ofs.open(fullPath);
        if (ofs.is_open()) {
            ofs << content;
            ofs.close();
            std::cout << "[sandboxNeg_FN::OverlayManager] Patched: " << filePath << std::endl;
        } else {
            std::cerr << "[sandboxNeg_FN::OverlayManager] Failed to patch " << filePath << std::endl;
        }
    }
}

void OverlayManager::cleanupOverlay(const std::string& overlayPath) {
    std::string umountCommand = "umount " + overlayPath;
    system(umountCommand.c_str());
    std::filesystem::remove_all("/tmp/sandboxNeg_FN-upper");
    std::filesystem::remove_all("/tmp/sandboxNeg_FN-work");
    std::filesystem::remove_all("/tmp/sandboxNeg_FN-rootfs");
    std::cout << "[sandboxNeg_FN::OverlayManager] Cleaned sandbox overlay environment." << std::endl;
}

} // namespace sandbox2
