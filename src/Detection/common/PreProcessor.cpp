#include "PreProcessor.hpp"
#include <algorithm>

PreProcessor::PreProcessor() {}

std::string PreProcessor::process(const std::string& raw_packet)
{
    // Stub : simple lowercase transformation and trimming example
    std::string processed = raw_packet;
    std::transform(processed.begin(), processed.end(), processed.begin(), ::tolower);
    // Additional cleaning/filtering logic here
    return processed;
}