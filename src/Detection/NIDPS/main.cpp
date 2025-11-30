#include "nidps_engine.hpp"
#include <iostream>
#include <csignal>
#include <memory>

std::unique_ptr<nidps::NIDPSEngine> g_engine;

void signalHandler(int signum) {
    std::cout << "\nInterrupt signal (" << signum << ") received." << std::endl;
    if (g_engine) {
        g_engine->stop();
    }
    exit(signum);
}

int main(int argc, char* argv[]) {
    std::cout << R"(
    ╔═══════════════════════════════════════════════════════╗
    ║   Network Intrusion Detection & Prevention System    ║
    ║              with DRL-Enhanced Sandboxing             ║
    ╚═══════════════════════════════════════════════════════╝
    )" << std::endl;
    
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    nidps::NIDPSConfig config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-i" || arg == "--interface") {
            if (i + 1 < argc) {
                config.network_interface = argv[++i];
            }
        } else if (arg == "-f" || arg == "--filter") {
            if (i + 1 < argc) {
                config.capture_filter = argv[++i];
            }
        } else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                config.model_path = argv[++i];
            }
        } else if (arg == "-d" || arg == "--database") {
            if (i + 1 < argc) {
                config.database_path = argv[++i];
            }
        } else if (arg == "-t" || arg == "--threshold") {
            if (i + 1 < argc) {
                config.malware_threshold = std::stof(argv[++i]);
            }
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -i, --interface <name>    Network interface (default: eth0)\n"
                      << "  -f, --filter <filter>     BPF capture filter\n"
                      << "  -m, --model <path>        Path to ONNX model (default: mtl_model.onnx)\n"
                      << "  -d, --database <path>     Database path (default: nidps.db)\n"
                      << "  -t, --threshold <value>   Malware detection threshold (default: 0.7)\n"
                      << "  -h, --help                Show this help message\n";
            return 0;
        }
    }
    
    g_engine = std::make_unique<nidps::NIDPSEngine>(config);
    
    if (!g_engine->initialize()) {
        std::cerr << "Failed to initialize NIDPS Engine" << std::endl;
        return 1;
    }
    
    g_engine->start();
    
    // Keep running until interrupted
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    return 0;
}
