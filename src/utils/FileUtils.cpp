#include "utils/FileUtils.hpp"
#include <fstream>
#include <iostream>

bool FileUtils::readFileToString(const std::string& filepath, std::string& outContent)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "[FileUtils] Cannot open file for reading : " << filepath << std::endl;
        return false;
    }
    outContent.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return true;
}

bool FileUtils::writeStringToFile(const std::string& filepath, const std::string& content)
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "[FileUtils] Cannot open file for writing : " << filepath << std::endl;
        return false;
    }
    file << content;
    file.close();
    return true;
}