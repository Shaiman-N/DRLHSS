#pragma once
#include <string>
#include <map>

namespace sandboxPos_FP {

class OverlayManager {
public:
    OverlayManager();
    std::string createOverlay(const std::string& baseImage,
                              const std::string& artifactPath,
                              const std::map<std::string, std::string>& hostProfile);
    void patchOverlay(const std::string& overlayPath,
                      const std::map<std::string, std::string>& patches);
    void cleanupOverlay(const std::string& overlayPath);
};

} // namespace sandbox1