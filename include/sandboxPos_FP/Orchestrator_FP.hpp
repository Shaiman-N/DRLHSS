#pragma once
#include <string>
#include <map>
#include <functional>

namespace sandboxPos_FP {

class OrchestratorFP {
public:
    OrchestratorFP();
    // interactionSimulator allows simulating user/network input in sandbox
    // envPatches to inject fake files/configs into sandbox overlay
    void runSandbox(const std::string& artifactPath,
                    std::function<void()> interactionSimulator = nullptr,
                    const std::map<std::string, std::string>& envPatches = {});
};

} // namespace sandbox1