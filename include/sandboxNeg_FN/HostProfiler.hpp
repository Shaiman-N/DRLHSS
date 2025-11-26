#pragma once
#include <string>
#include <map>

namespace sandboxNeg_FN {
class HostProfiler {
public:
    HostProfiler();
    std::map<std::string, std::string> collectHostProfile();
};
} // namespace sandbox2
