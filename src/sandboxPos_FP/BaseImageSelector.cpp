#include "sandboxPos_FP/BaseImageSelector.hpp"
#include <iostream>

namespace sandboxPos_FP {

BaseImageSelector::BaseImageSelector() {}

std::string BaseImageSelector::selectBaseImage(const std::map<std::string, std::string>& hostProfile) {
    // For demo, static base image path; customize to profile dynamically
    const std::string selectedImage = "/opt/sandbox-images/ubuntu-minimal-rootfs";
    std::cout << "[BaseImageSelector] Selected base image: " << selectedImage << std::endl;
    return selectedImage;
}

} // namespace sandbox1