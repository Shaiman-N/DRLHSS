#pragma once
#include <string>

namespace sandboxNeg_FN {

class Cleaner {
public:
    Cleaner();
    std::string cleanFile(const std::string& inputFilePath);
};

} // namespace sandbox2
