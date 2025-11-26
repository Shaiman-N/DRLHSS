#pragma once
#include <string>

namespace sandboxPos_FP {

class Cleaner {
public:
    Cleaner();
    // Returns sanitized filename or empty string on failure
    std::string cleanFile(const std::string& inputFilePath);
};

} // namespace sandbox1