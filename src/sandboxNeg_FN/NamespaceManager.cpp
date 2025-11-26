#ifdef __linux__
#include "sandboxNeg_FN/NamespaceManager.hpp"
#include <iostream>
#include <sched.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>

namespace sandboxNeg_FN {

NamespaceManager::NamespaceManager() {}

void NamespaceManager::setupNamespaces() {
    if(unshare(CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWUTS | CLONE_NEWIPC) != 0) {
        std::cerr << "[sandboxNeg_FN::NamespaceManager] unshare failed: " << strerror(errno) << std::endl;
        throw std::runtime_error("Namespace setup failed");
    }
    std::cout << "[sandboxNeg_FN::NamespaceManager] Namespaces set." << std::endl;
}

} // namespace sandbox2
#endif