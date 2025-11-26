#include "sandboxNeg_FN/SandboxRunner.hpp"
#include <iostream>
#include <unistd.h>
#include <cstdlib>

namespace sandboxNeg_FN {

SandboxRunner::SandboxRunner() {}

void SandboxRunner::launch(const std::string& overlayFS,
                           const std::string& artifactPath,
                           std::function<void()> interactionSimulator) {
    if(chroot(overlayFS.c_str()) != 0) {
        perror("[sandboxNeg_FN::SandboxRunner] chroot failed");
        throw std::runtime_error("chroot error");
    }
    chdir("/");
    setuid(65534); // Drop privilege to nobody

    if(interactionSimulator) {
        interactionSimulator();
    }

    std::cout << "[sandboxNeg_FN::SandboxRunner] Running artifact in sandbox..." << std::endl;
    std::string cmd = "chmod +x artifact && ./artifact";
    int ret = system(cmd.c_str());
    if(ret != 0) {
        std::cerr << "[sandboxNeg_FN::SandboxRunner] Artifact execution failed: " << ret << std::endl;
    }
}

} // namespace sandbox2
