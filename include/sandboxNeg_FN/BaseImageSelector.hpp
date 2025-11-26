#pragma once
#include <map>
#include <string>

namespace sandboxNeg_FN {

class BaseImageSelector {
public:
    BaseImageSelector();
    std::string selectBaseImage(const std::map<std::string, std::string>& hostProfile);
};

} // namespace sandbox2
