#ifndef PACKET_DATA_HPP
#define PACKET_DATA_HPP

#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <memory>

namespace nidps {

enum class PacketStatus {
    CLEAN,
    SUSPICIOUS,
    MALICIOUS,
    CLEANED,
    PROCESSING
};

struct PacketData {
    uint64_t packet_id;
    std::vector<uint8_t> raw_data;
    std::vector<float> features;
    PacketStatus status;
    std::chrono::system_clock::time_point timestamp;
    std::string source_ip;
    std::string dest_ip;
    uint16_t source_port;
    uint16_t dest_port;
    uint8_t protocol;
    bool false_negative_check;
    int sandbox_pass_count;
    
    PacketData() : packet_id(0), status(PacketStatus::CLEAN), 
                   source_port(0), dest_port(0), protocol(0),
                   false_negative_check(false), sandbox_pass_count(0) {
        timestamp = std::chrono::system_clock::now();
    }
};

using PacketPtr = std::shared_ptr<PacketData>;

} // namespace nidps

#endif // PACKET_DATA_HPP
