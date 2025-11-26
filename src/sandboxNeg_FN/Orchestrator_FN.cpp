#include "sandboxNeg_FN/OrchestratorFN.hpp"
#include "sandboxNeg_FN/HostProfiler.hpp"
#include "sandboxNeg_FN/BaseImageSelector.hpp"
#include "sandboxNeg_FN/OverlayManager.hpp"
#include "sandboxNeg_FN/NamespaceManager.hpp"
#include "sandboxNeg_FN/SandboxRunner.hpp"
#include "sandboxNeg_FN/TelemetryCollector.hpp"
#include "sandboxNeg_FN/Cleaner.hpp"
#include <iostream>
#include <stdexcept>

namespace sandboxNeg_FN {

OrchestratorFN::OrchestratorFN() {}

void OrchestratorFN::runSandbox(const std::string& artifactPath,
                                std::function<void()> interactionSimulator,
                                const std::map<std::string, std::string>& envPatches,
                                bool multiStage) {
    std::cout << "[sandboxNeg_FN::OrchestratorFN] Starting sandbox run for artifact: " << artifactPath << std::endl;

    HostProfiler profiler;
    auto hostProfile = profiler.collectHostProfile();

    BaseImageSelector selector;
    auto baseImage = selector.selectBaseImage(hostProfile);

    OverlayManager overlay;
    auto overlayFS = overlay.createOverlay(baseImage, artifactPath, hostProfile);

    if (!envPatches.empty()) {
        overlay.patchOverlay(overlayFS, envPatches);
    }

    NamespaceManager nsManager;
    nsManager.setupNamespaces();

    TelemetryCollector telemetry;
    telemetry.startCollection();

    SandboxRunner runner;
    runner.launch(overlayFS, artifactPath, interactionSimulator);

    if (multiStage) {
        std::cout << "[sandboxNeg_FN::OrchestratorFN] Multi-stage execution enabled." << std::endl;
        // Advanced multi-stage execution logic example:
        // 1. Monitor initial execution telemetry
        // 2. Adapt overlay or envPatches dynamically for second stage
        // 3. Re-run or chain execution with updated environment
        // (actual multi-stage chaining logic to be implemented here)
    }

    telemetry.stopCollection();

    // Integrate detection logic based on telemetry here
    bool detectedMalicious = true; // Replace with real detection results

    if (detectedMalicious) {
        std::cout << "[sandboxNeg_FN::OrchestratorFN] Malicious behavior detected. Invoking cleaning module..." << std::endl;
        Cleaner cleaner;
        std::string cleanedFile = cleaner.cleanFile(artifactPath);
        if (!cleanedFile.empty()) {
            std::cout << "[sandboxNeg_FN::OrchestratorFN] Cleaning successful. Sanitized file: " << cleanedFile << std::endl;
        } else {
            std::cerr << "[sandboxNeg_FN::OrchestratorFN] Cleaning failed. Quarantine actions advised." << std::endl;
        }
    } else {
        std::cout << "[sandboxNeg_FN::OrchestratorFN] No malware behavior detected." << std::endl;
    }

    overlay.cleanupOverlay(overlayFS);
    std::cout << "[sandboxNeg_FN::OrchestratorFN] Sandbox run complete." << std::endl;
}
} // namespace sandbox2
