#include "sandboxPos_FP/Orchestrator_FP.hpp"
#include "sandboxPos_FP/HostProfiler.hpp"
#include "sandboxPos_FP/BaseImageSelector.hpp"
#include "sandboxPos_FP/OverlayManager.hpp"
#include "sandboxPos_FP/NamespaceManager.hpp"
#include "sandboxPos_FP/SandboxRunner.hpp"
#include "sandboxPos_FP/TelemetryCollector.hpp"
#include "sandboxPos_FP/Cleaner.hpp"
#include <iostream>
#include <stdexcept>

namespace sandboxPos_FP {

OrchestratorFP::OrchestratorFP() {}

void OrchestratorFP::runSandbox(const std::string& artifactPath,
                                std::function<void()> interactionSimulator,
                                const std::map<std::string, std::string>& envPatches) {
    std::cout << "[OrchestratorFP] Starting sandbox run for artifact: " << artifactPath << std::endl;

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

    telemetry.stopCollection();

    // Here, based on telemetry and sandbox analysis, suppose malicious detected
    bool detectedMalicious = true; // Replace with real detection integration

    if (detectedMalicious) {
        std::cout << "[OrchestratorFP] Malicious behavior detected. Cleaning file..." << std::endl;
        Cleaner cleaner;
        auto cleanedFile = cleaner.cleanFile(artifactPath);
        if (!cleanedFile.empty()) {
            std::cout << "[OrchestratorFP] Cleaning success. Sanitized file: " << cleanedFile << std::endl;
            // Handle sanitized file (re-scan, deploy, notify)
        } else {
            std::cerr << "[OrchestratorFP] Cleaning failed. Consider quarantining the file." << std::endl;
        }
    } else {
        std::cout << "[OrchestratorFP] No malicious behavior detected. No cleaning needed." << std::endl;
    }

    overlay.cleanupOverlay(overlayFS);
    std::cout << "[OrchestratorFP] Sandbox run completed" << std::endl;
}

} // namespace sandbox1