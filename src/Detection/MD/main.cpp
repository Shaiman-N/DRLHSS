#include "MalwareDetectionService.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Malware Detection System - Dual Sandbox Architecture    ║" << std::endl;
    std::cout << "║   with Deep Reinforcement Learning Framework              ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
    
    // Check if model path is provided
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_onnx_model> [files...]" << std::endl;
        std::cerr << "Example: " << argv[0] << " malware_dcnn.onnx test_file.exe" << std::endl;
        std::cerr << std::endl;
        std::cerr << "System Architecture:" << std::endl;
        std::cerr << "  1. Initial Detection (ONNX Model)" << std::endl;
        std::cerr << "  2. Positive Sandbox (if malware detected)" << std::endl;
        std::cerr << "  3. Negative Sandbox (false negative check)" << std::endl;
        std::cerr << "  4. DRL Learning & Database Storage" << std::endl;
        std::cerr << "  5. Cleaned data sent to host system" << std::endl;
        return 1;
    }
    
    std::string modelPath = argv[1];
    std::string databasePath = "malware_detection_db.txt";
    
    try {
        // Create and start the malware detection service
        std::cout << "Initializing system components..." << std::endl;
        MalwareDetectionService service(modelPath, databasePath);
        service.start();
        
        std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║              Service Running - Ready to Process            ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝\n" << std::endl;
        
        // Example: Queue some files for scanning
        if (argc > 2) {
            std::cout << "Queueing files for processing:\n" << std::endl;
            for (int i = 2; i < argc; i++) {
                std::cout << "  → " << argv[i] << std::endl;
                service.queueFileScan(argv[i]);
            }
            std::cout << std::endl;
        } else {
            std::cout << "No files specified. Running demo with test packet.\n" << std::endl;
        }
        
        // Example: Simulate scanning a data packet
        std::cout << "Queueing test data packet for processing...\n" << std::endl;
        std::vector<uint8_t> testPacket(1024);
        for (size_t i = 0; i < testPacket.size(); i++) {
            testPacket[i] = static_cast<uint8_t>(i % 256);
        }
        service.queueDataPacketScan(testPacket);
        
        // Let the service run for a while to process tasks
        std::cout << "Processing through dual-sandbox pipeline..." << std::endl;
        std::cout << "(This may take a few moments)\n" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(10));
        
        // Stop the service
        std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                    Shutting Down System                    ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝\n" << std::endl;
        service.stop();
        
        std::cout << "✓ Service terminated successfully." << std::endl;
        std::cout << "✓ DRL learning data saved to: " << databasePath << std::endl;
        std::cout << "\nThank you for using the Malware Detection System!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
