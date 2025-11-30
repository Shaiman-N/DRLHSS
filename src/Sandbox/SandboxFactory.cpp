#include "Sandbox/SandboxFactory.hpp"
#include "Sandbox/Linux/LinuxSandbox.hpp"
#include "Sandbox/Windows/WindowsSandbox.hpp"
#include "Sandbox/MacOS/MacOSSandbox.hpp"
#include <stdexcept>

namespace sandbox {

Platform SandboxFactory::detectPlatform() {
#ifdef __linux__
    return Platform::LINUX;
#elif defined(_WIN32) || defined(_WIN64)
    return Platform::WINDOWS;
#elif defined(__APPLE__) && defined(__MACH__)
    return Platform::MACOS;
#else
    return Platform::UNKNOWN;
#endif
}

std::unique_ptr<ISandbox> SandboxFactory::createSandbox(SandboxType type,
                                                         const SandboxConfig& config) {
    Platform platform = detectPlatform();
    return createSandbox(platform, type, config);
}

std::unique_ptr<ISandbox> SandboxFactory::createSandbox(Platform platform,
                                                         SandboxType type,
                                                         const SandboxConfig& config) {
    std::unique_ptr<ISandbox> sandbox;
    
    switch (platform) {
        case Platform::LINUX:
#ifdef __linux__
            sandbox = std::make_unique<LinuxSandbox>(type);
#else
            throw std::runtime_error("Linux sandbox not available on this platform");
#endif
            break;
            
        case Platform::WINDOWS:
#ifdef _WIN32
            sandbox = std::make_unique<WindowsSandbox>(type);
#else
            throw std::runtime_error("Windows sandbox not available on this platform");
#endif
            break;
            
        case Platform::MACOS:
#ifdef __APPLE__
            sandbox = std::make_unique<MacOSSandbox>(type);
#else
            throw std::runtime_error("macOS sandbox not available on this platform");
#endif
            break;
            
        case Platform::UNKNOWN:
        default:
            throw std::runtime_error("Unsupported platform");
    }
    
    if (sandbox && !sandbox->initialize(config)) {
        throw std::runtime_error("Failed to initialize sandbox");
    }
    
    return sandbox;
}

std::string SandboxFactory::getPlatformName(Platform platform) {
    switch (platform) {
        case Platform::LINUX:   return "Linux";
        case Platform::WINDOWS: return "Windows";
        case Platform::MACOS:   return "macOS";
        case Platform::UNKNOWN: return "Unknown";
        default:                return "Unknown";
    }
}

bool SandboxFactory::isPlatformSupported(Platform platform) {
    switch (platform) {
        case Platform::LINUX:
#ifdef __linux__
            return true;
#else
            return false;
#endif
            
        case Platform::WINDOWS:
#ifdef _WIN32
            return true;
#else
            return false;
#endif
            
        case Platform::MACOS:
#ifdef __APPLE__
            return true;
#else
            return false;
#endif
            
        default:
            return false;
    }
}

} // namespace sandbox
