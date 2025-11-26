#include "sandboxPos_FP/HostProfiler.hpp"
#include <fstream>
#include <iostream>
#include <cstdlib>

namespace sandboxPos_FP {

HostProfiler::HostProfiler() {}

std::map<std::string, std::string> HostProfiler::collectHostProfile() {
    std::map<std::string, std::string> profile;
    // Example: read /etc/os-release
    std::ifstream osRelease("/etc/os-release");

    if (!osRelease.is_open()) {
        std::cerr << "[HostProfiler] Failed to open /etc/os-release" << std::endl;
        return profile;
    }

    std::string line;
    while (std::getline(osRelease, line)) {
        auto pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            profile[key] = value;
        }
    }
    osRelease.close();

    // Additional host info
    char buffer[128];
    FILE* pipe = popen("uname -a", "r");
    if (pipe) {
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            profile["kernel"] = std::string(buffer);
        }
        pclose(pipe);
    }

    return profile;
}

} // namespace sandbox1