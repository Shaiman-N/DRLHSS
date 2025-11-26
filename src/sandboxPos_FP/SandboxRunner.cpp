#include "sandboxPos_FP/SandboxRunner.hpp"
#include <unistd.h>
#include <cstdlib>
#include <iostream>
#include <sys/types.h>

namespace sandboxPos_FP {

SandboxRunner::SandboxRunner() {}

void SandboxRunner::launch(const std::string& overlayFS,
                           const std::string& artifactPath,
                           std::function<void()> interactionSimulator) {
    if (chroot(overlayFS.c_str()) != 0) {
        perror("[SandboxRunner] chroot failed");
        throw std::runtime_error("chroot error");
    }
    chdir("/");
    setuid(65534); // nobody user drop privilege

    if (interactionSimulator) {
        interactionSimulator();
    }

    std::cout << "[SandboxRunner] Executing artifact inside sandbox." << std::endl;
    std::string cmd = "chmod +x artifact && ./artifact";
    int ret = system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "[SandboxRunner] Artifact execution failed with code: " << ret << std::endl;
    }
}

} // namespace sandbox1