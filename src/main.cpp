#include "sandboxPos_FP/Orchestrator_FP.hpp"
#include "sandboxNeg_FN/OrchestratorFN.hpp"
#include <iostream>
#include <map>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// Placeholder for detection result interface (real detection would be integrated in orchestrator)
bool detectMalwareInSandbox(const std::string& artifactPath) {
    // This would actually come from telemetry + DRL feedback
    // For demo: treat filename containing "malware" as malicious
    return artifactPath.find("malware") != std::string::npos;
}

// Simulate behavioral interaction in sandboxes
void positiveSandboxInteraction() {
    std::cout << "[Main] Positive sandbox interaction simulation..." << std::endl;
    system("ping -c 1 8.8.8.8 > /dev/null 2>&1");
}

void negativeSandboxInteraction() {
    std::cout << "[Main] Negative sandbox interaction simulation..." << std::endl;
    system("ping -c 2 8.8.8.8 > /dev/null 2>&1");
    system("curl -s http://example.com > /dev/null 2>&1");
}

void printUsage(const std::string& exeName) {
    std::cout << "Usage: " << exeName << " <path_to_suspicious_file>" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string originalArtifact = argv[1];
    std::string artifactToProcess = originalArtifact;
    sandboxPos_FP::OrchestratorFP positiveSandbox;
    sandboxNeg_FN::OrchestratorFN negativeSandbox;

    std::map<std::string, std::string> positiveEnvPatches = {
        {"etc/fake_av.conf", "AV=PositiveScanner\nVersion=1.0\n"},
        {"var/fake_network", "state=connected\nlatency=10ms\n"}
    };

    std::map<std::string, std::string> negativeEnvPatches = {
        {"etc/fake_security.conf", "Scanner=NegativeScanner\nVersion=2.1\n"},
        {"var/fake_network", "state=active\nlatency=15ms\n"}
    };

    try {
        std::cout << "[Main] Running file in positive sandbox (sandbox1)..." << std::endl;
        positiveSandbox.runSandbox(artifactToProcess, positiveSandboxInteraction, positiveEnvPatches);

        if (detectMalwareInSandbox(artifactToProcess)) {
            std::cout << "[Main] Malware detected by positive sandbox. Initiating DRL learning and cleaning..." << std::endl;
            // Positive sandbox cleaning simulated inside runSandbox via Cleaner
            // After cleaning, assume new path for cleaned file:
            artifactToProcess += ".cleaned";

            std::cout << "[Main] Running cleaned file in negative sandbox (sandbox2) for false negative check..." << std::endl;
            negativeSandbox.runSandbox(artifactToProcess, negativeSandboxInteraction, negativeEnvPatches, true);

            if (detectMalwareInSandbox(artifactToProcess)) {
                std::cout << "[Main] Malware detected again by negative sandbox. Further cleaning and DRL learning..." << std::endl;
                // Negative sandbox cleaning simulated by its Cleaner module
                artifactToProcess += ".cleaned2";
                std::cout << "[Main] Fully cleaned data packet ready for host system: " << artifactToProcess << std::endl;
            } else {
                std::cout << "[Main] Negative sandbox found no malware - False negatives cleared." << std::endl;
            }
        } else {
            std::cout << "[Main] Positive sandbox found no malware, file is clean." << std::endl;
        }

        std::cout << "[Main] Malware processing pipeline completed successfully." << std::endl;
        std::cout << "[Main] Final artifact sent to host/system: " << artifactToProcess << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << "[Main] Exception occurred: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
