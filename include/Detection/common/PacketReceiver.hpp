#pragma once
#include <string>
#include <vector>

class PacketReceiver
{
    public:
        // Initialize receiver (network interfaces, files, etc.)
        PacketReceiver();

        // Receive raw data packets; returns list of raw packet payloads as strings or bytes
        std::vector<std::string> receivePackets();
};