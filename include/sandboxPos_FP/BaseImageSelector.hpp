#pragma once
#include <map>
#include <string>

namespace sandboxPos_FP {

class BaseImageSelector {
public:
    BaseImageSelector();
    std::string selectBaseImage(const std::map<std::string, std::string>& hostProfile);
};

} // namespace sandbox1