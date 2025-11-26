#pragma once
#include <string>

class PreProcessor
{
    public:
        PreProcessor();

        // Clean and normalize raw packet data (e.g., decoding, filtering)
        std::string process(const std::string& raw_packet);
};