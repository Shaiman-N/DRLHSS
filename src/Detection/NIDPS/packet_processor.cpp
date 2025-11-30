#include "packet_processor.hpp"
#include <iostream>

namespace nidps {

PacketProcessor::PacketProcessor()
    : processed_count_(0), malicious_count_(0), clean_count_(0) {}

PacketProcessor::~PacketProcessor() {}

void PacketProcessor::setOutputCallback(OutputCallback callback) {
    output_callback_ = callback;
}

void PacketProcessor::sendToHost(const PacketPtr& packet) {
    if (!validatePacket(packet)) {
        std::cerr << "Invalid packet " << packet->packet_id 
                  << " - not forwarding to host" << std::endl;
        return;
    }
    
    processed_count_++;
    
    if (packet->status == PacketStatus::CLEAN || 
        packet->status == PacketStatus::CLEANED) {
        clean_count_++;
        
        std::cout << "Forwarding packet " << packet->packet_id 
                  << " to host system (Status: " 
                  << (packet->status == PacketStatus::CLEAN ? "CLEAN" : "CLEANED")
                  << ")" << std::endl;
        
        logPacketDelivery(packet);
        
        if (output_callback_) {
            output_callback_(packet);
        }
    } else if (packet->status == PacketStatus::MALICIOUS) {
        malicious_count_++;
        std::cout << "Blocked malicious packet " << packet->packet_id << std::endl;
    }
}

bool PacketProcessor::validatePacket(const PacketPtr& packet) {
    if (!packet) {
        return false;
    }
    
    if (packet->raw_data.empty()) {
        std::cerr << "Packet " << packet->packet_id << " has no data" << std::endl;
        return false;
    }
    
    if (packet->sandbox_pass_count < 2) {
        std::cerr << "Packet " << packet->packet_id 
                  << " has not completed both sandbox passes" << std::endl;
        return false;
    }
    
    return true;
}

void PacketProcessor::logPacketDelivery(const PacketPtr& packet) {
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - packet->timestamp);
    
    std::cout << "Packet " << packet->packet_id << " delivery: "
              << packet->source_ip << ":" << packet->source_port << " -> "
              << packet->dest_ip << ":" << packet->dest_port
              << " (Processing time: " << duration.count() << "ms)" << std::endl;
}

} // namespace nidps
