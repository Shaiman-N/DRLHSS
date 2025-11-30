#pragma once

#include "SandboxInterface.hpp"
#include <memory>
#include <string>

namespace sandbox {

/**
 * @brief Platform detection
 */
enum class Platform {
    LINUX,
    WINDOWS,
    MACOS,
    UNKNOWN
};

/**
 * @brief Factory for creating platform-specific sandbox instances
 */
class SandboxFactory {
public:
    /**
     * @brief Detect current platform
     * @return Detected platform
     */
    static Platform detectPlatform();
    
    /**
     * @brief Create sandbox for current platform
     * @param type Sandbox type (Positive/Negative)
     * @param config Sandbox configuration
     * @return Platform-specific sandbox instance
     */
    static std::unique_ptr<ISandbox> createSandbox(SandboxType type,
                                                    const SandboxConfig& config);
    
    /**
     * @brief Create sandbox for specific platform
     * @param platform Target platform
     * @param type Sandbox type
     * @param config Sandbox configuration
     * @return Platform-specific sandbox instance
     */
    static std::unique_ptr<ISandbox> createSandbox(Platform platform,
                                                    SandboxType type,
                                                    const SandboxConfig& config);
    
    /**
     * @brief Get platform name as string
     */
    static std::string getPlatformName(Platform platform);
    
    /**
     * @brief Check if platform is supported
     */
    static bool isPlatformSupported(Platform platform);
};

} // namespace sandbox
