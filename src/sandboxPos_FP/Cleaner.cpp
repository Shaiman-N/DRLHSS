#include "sandboxPos_FP/Cleaner.hpp"
#include <iostream>
#include <filesystem>
#include <cstdlib>

namespace sandboxPos_FP {

Cleaner::Cleaner() {}

static std::string generateCleanFilename(const std::string& original) {
    return original + ".cleaned";
}

std::string Cleaner::cleanFile(const std::string& inputFilePath) {
    std::cout << "[Cleaner] Starting cleaning for: " << inputFilePath << std::endl;

    // Detect file type rudimentarily
    std::string fileType;
    if (inputFilePath.size() > 4 && inputFilePath.compare(inputFilePath.size() - 4, 4, ".exe") == 0)
        fileType = "executable";
    else if (inputFilePath.size() > 4 && inputFilePath.compare(inputFilePath.size() - 4, 4, ".pdf") == 0)
        fileType = "pdf";
    else if (inputFilePath.find(".doc") != std::string::npos || inputFilePath.find(".xls") != std::string::npos)
        fileType = "office";
    else
        fileType = "unknown";

    std::string cleanedFile = generateCleanFilename(inputFilePath);

    if (fileType == "executable") {
        // Call external cleaning or stub: just copy for now
        std::filesystem::copy(inputFilePath, cleanedFile, std::filesystem::copy_options::overwrite_existing);
        std::cout << "[Cleaner] Executable cleaning (stub) done." << std::endl;
    } else if (fileType == "pdf") {
        // Use qpdf for example sanitization
        std::string cmd = "qpdf --linearize " + inputFilePath + " " + cleanedFile;
        if (system(cmd.c_str()) != 0) {
            std::cerr << "[Cleaner] PDF sanitization failed." << std::endl;
            return "";
        }
        std::cout << "[Cleaner] PDF sanitization done." << std::endl;
    } else if (fileType == "office") {
        // Stub: copy file, replace with real macro sanitizer
        std::filesystem::copy(inputFilePath, cleanedFile, std::filesystem::copy_options::overwrite_existing);
        std::cout << "[Cleaner] Office document cleaning (stub) done." << std::endl;
    } else {
        std::cerr << "[Cleaner] Unknown type, cleaning aborted." << std::endl;
        return "";
    }
    return cleanedFile;
}

} // namespace sandbox1
