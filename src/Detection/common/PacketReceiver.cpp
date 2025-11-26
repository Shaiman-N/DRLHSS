#include "PacketReceiver.hpp"
#include <iostream>

PacketReceiver::PacketReceiver()
{
    // Initialize network capture or file handles here (stub for now)
}

std::vector<std::string> PacketReceiver::receivePackets()
{
    // Stub : Simulate receiving some dummy packets
    std::vector<std::string> packets;
    packets.push_back("packet_data_1");
    packets.push_back("packet_data_2");
    std::cout<<"Received 2 packets from network/input source." << std::endl;
    return packets;
}
