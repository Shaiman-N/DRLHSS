#include "sandboxNeg_FN/HostProfiler.hpp"
#include <iostream>
#include <fstream>
#include <cstdio>

namespace sandboxNeg_FN {

HostProfiler::HostProfiler() {}

std::map<std::string, std::string> HostProfiler::collectHostProfile() {
    std::map<std::string, std::string> profile;
    std::ifstream file("/etc/os-release");
    if (!file.is_open()) {
        std::cerr << "[sandboxNeg_FN::HostProfiler] Failed to open /etc/os-release" << std::endl;
        return profile;
    }
    std::string line;
    while(std::getline(file, line)) {
        auto pos = line.find('=');
        if(pos != std::string::npos) {
            profile[line.substr(0, pos)] = line.substr(pos+1);
        }
    }
    file.close();

    char buffer[256];
    FILE* pipe = popen("uname -a", "r");
    if(pipe) {
        if(fgets(buffer, sizeof(buffer), pipe)) {
            profile["kernel"] = std::string(buffer);
        }
        pclose(pipe);
    }
    return profile;
}

} // namespace sandbox2