#pragma once
#include <string>
#include <map>

namespace sandboxPos_FP {

class HostProfiler {
public:
    HostProfiler();
    std::map<std::string, std::string> collectHostProfile();
};

} // namespace sandbox1