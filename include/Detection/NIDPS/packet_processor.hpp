#ifndef PACKET_PROCESSOR_HPP
#define PACKET_PROCESSOR_HPP

#include "packet_data.hpp"
#include <functional>
#include <atomic>

namespace nidps {

class PacketProcessor {
public:
    using OutputCallback = std::function<void(const PacketPtr&)>;
    
    PacketProcessor();
    ~PacketProcessor();
    
    void setOutputCallback(OutputCallback callback);
    void sendToHost(const PacketPtr& packet);
    
    uint64_t getProcessedCount() const { return processed_count_; }
    uint64_t getMaliciousCount() const { return malicious_count_; }
    uint64_t getCleanCount() const { return clean_count_; }
    
private:
    bool validatePacket(const PacketPtr& packet);
    void logPacketDelivery(const PacketPtr& packet);
    
    OutputCallback output_callback_;
    std::atomic<uint64_t> processed_count_;
    std::atomic<uint64_t> malicious_count_;
    std::atomic<uint64_t> clean_count_;
};

} // namespace nidps

#endif // PACKET_PROCESSOR_HPP
