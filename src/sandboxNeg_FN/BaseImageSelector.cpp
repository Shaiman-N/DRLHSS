#include "sandboxNeg_FN/BaseImageSelector.hpp"
#include <iostream>

namespace sandboxNeg_FN {

BaseImageSelector::BaseImageSelector() {}

std::string BaseImageSelector::selectBaseImage(const std::map<std::string, std::string>& /*hostProfile*/) {
    const std::string baseImage = "/opt/sandbox-images/ubuntu-minimal-rootfs";
    std::cout << "[sandboxNeg_FN::BaseImageSelector] Selected base image: " << baseImage << std::endl;
    return baseImage;
}

} // namespace sandbox2
