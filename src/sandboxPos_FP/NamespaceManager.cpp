#include "sandboxPos_FP/NamespaceManager.hpp"
#include <iostream>
#include <sched.h>
#include <unistd.h>
#include <cerrno>

namespace sandboxPos_FP {

NamespaceManager::NamespaceManager() {}

void NamespaceManager::setupNamespaces() {
    if (unshare(CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWUTS | CLONE_NEWIPC) != 0) {
        std::cerr << "[NamespaceManager] unshare failed: " << strerror(errno) << std::endl;
        throw std::runtime_error("Namespace setup failed");
    }
    std::cout << "[NamespaceManager] Namespaces setup successfully." << std::endl;
}

} // namespace sandbox1