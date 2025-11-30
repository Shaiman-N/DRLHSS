#include "Sandbox/Linux/LinuxSandbox.hpp"
#include <iostream>
#include <cassert>
#include <vector>

void testSandboxInitialization() {
    std::cout << "[TEST] Testing Linux Sandbox Initialization...\n";
    
    sandbox::LinuxSandbox sandbox(sandbox::SandboxType::POSITIVE_FP);
    
    sandbox::SandboxConfig config;
    config.sandbox_id = "test_linux_001";
    config.memory_limit_mb = 256;
    config.cpu_limit_percent = 50;
    config.timeout_seconds = 10;
    
    bool result = sandbox.initialize(config);
    assert(result && "Sandbox initialization failed");
    assert(sandbox.isReady() && "Sandbox not ready after initialization");
    
    std::cout << "[TEST] ✓ Initialization test passed\n";
    
    sandbox.cleanup();
}

void testSandboxExecution() {
    std::cout << "[TEST] Testing Linux Sandbox Execution...\n";
    
    sandbox::LinuxSandbox sandbox(sandbox::SandboxType::POSITIVE_FP);
    
    sandbox::SandboxConfig config;
    config.sandbox_id = "test_linux_002";
    config.memory_limit_mb = 256;
    config.cpu_limit_percent = 50;
    config.timeout_seconds = 10;
    
    assert(sandbox.initialize(config) && "Initialization failed");
    
    // Test with /bin/echo
    std::vector<std::string> args = {"Hello", "Sandbox"};
    sandbox::SandboxResult result = sandbox.execute("/bin/echo", args);
    
    assert(result.success && "Execution failed");
    std::cout << "[TEST] Exit code: " << result.exit_code << "\n";
    std::cout << "[TEST] Execution time: " << result.execution_time.count() << "ms\n";
    std::cout << "[TEST] Threat score: " << result.threat_score << "\n";
    
    std::cout << "[TEST] ✓ Execution test passed\n";
    
    sandbox.cleanup();
}

void testSandboxPacketAnalysis() {
    std::cout << "[TEST] Testing Linux Sandbox Packet Analysis...\n";
    
    sandbox::LinuxSandbox sandbox(sandbox::SandboxType::POSITIVE_FP);
    
    sandbox::SandboxConfig config;
    config.sandbox_id = "test_linux_003";
    config.memory_limit_mb = 256;
    config.cpu_limit_percent = 50;
    config.timeout_seconds = 10;
    
    assert(sandbox.initialize(config) && "Initialization failed");
    
    // Create test packet data
    std::vector<uint8_t> packet_data = {
        0x45, 0x00, 0x00, 0x3c, 0x1c, 0x46, 0x40, 0x00,
        0x40, 0x06, 0xb1, 0xe6, 0xc0, 0xa8, 0x00, 0x68,
        0xc0, 0xa8, 0x00, 0x01
    };
    
    sandbox::SandboxResult result = sandbox.analyzePacket(packet_data);
    
    std::cout << "[TEST] Packet analysis complete\n";
    std::cout << "[TEST] Threat score: " << result.threat_score << "\n";
    std::cout << "[TEST] File system modified: " << result.file_system_modified << "\n";
    std::cout << "[TEST] Network activity: " << result.network_activity_detected << "\n";
    
    std::cout << "[TEST] ✓ Packet analysis test passed\n";
    
    sandbox.cleanup();
}

void testSandboxReset() {
    std::cout << "[TEST] Testing Linux Sandbox Reset...\n";
    
    sandbox::LinuxSandbox sandbox(sandbox::SandboxType::POSITIVE_FP);
    
    sandbox::SandboxConfig config;
    config.sandbox_id = "test_linux_004";
    config.memory_limit_mb = 256;
    config.cpu_limit_percent = 50;
    config.timeout_seconds = 10;
    
    assert(sandbox.initialize(config) && "Initialization failed");
    
    // Execute something
    sandbox.execute("/bin/ls", {});
    
    // Reset
    sandbox.reset();
    
    assert(sandbox.isReady() && "Sandbox not ready after reset");
    
    std::cout << "[TEST] ✓ Reset test passed\n";
    
    sandbox.cleanup();
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  Linux Sandbox Test Suite\n";
    std::cout << "========================================\n\n";
    
    try {
        testSandboxInitialization();
        testSandboxExecution();
        testSandboxPacketAnalysis();
        testSandboxReset();
        
        std::cout << "\n========================================\n";
        std::cout << "  All Tests Passed! ✓\n";
        std::cout << "========================================\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Test failed: " << e.what() << "\n";
        return 1;
    }
}

