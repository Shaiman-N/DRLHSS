#pragma once
#include <string>
#include <map>
#include <functional>

namespace sandboxNeg_FN {
class OrchestratorFN {
public:
    OrchestratorFN();
    // interactionSimulator runs behavioral triggers while artifact runs
    // envPatches inject configs to overlay filesystem dynamically
    // multiStage enables chaining multi-phase execution (advanced simulation)
    void runSandbox(const std::string& artifactPath,
                    std::function<void()> interactionSimulator = nullptr,
                    const std::map<std::string, std::string>& envPatches = {},
                    bool multiStage = false);
};
} // namespace sandbox2