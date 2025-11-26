#pragma once
#include <string>
#include <functional>

namespace sandboxNeg_FN {

class SandboxRunner {
public:
    SandboxRunner();
    void launch(const std::string& overlayFS,
                const std::string& artifactPath,
                std::function<void()> interactionSimulator = nullptr);
};

} // namespace sandbox2
