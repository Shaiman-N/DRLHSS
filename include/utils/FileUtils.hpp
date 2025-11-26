#pragma once
#include <string>
#include <vector>

class FileUtils
{
    public:
        static bool readFileToString(const std::string& filepath, std::string& outContent);
        static bool writeStringToFile(const std::string& filepath, const std::string& content);
};