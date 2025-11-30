#ifndef PACKET_CAPTURE_HPP
#define PACKET_CAPTURE_HPP

#include "packet_data.hpp"
#include <pcap.h>
#include <string>
#include <functional>
#include <atomic>
#include <thread>

namespace nidps {

class PacketCapture {
public:
    using PacketCallback = std::function<void(PacketPtr)>;
    
    PacketCapture(const std::string& interface, const std::string& filter = "");
    ~PacketCapture();
    
    bool initialize();
    void start(PacketCallback callback);
    void stop();
    bool isRunning() const { return running_; }
    
private:
    static void packetHandler(u_char* user_data, const struct pcap_pkthdr* header, 
                             const u_char* packet);
    void captureLoop();
    PacketPtr parsePacket(const struct pcap_pkthdr* header, const u_char* packet);
    std::vector<float> extractFeatures(const PacketPtr& packet);
    
    std::string interface_;
    std::string filter_;
    pcap_t* handle_;
    std::atomic<bool> running_;
    std::thread capture_thread_;
    PacketCallback callback_;
    uint64_t packet_counter_;
};

} // namespace nidps

#endif // PACKET_CAPTURE_HPP
