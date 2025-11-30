#include "packet_capture.hpp"
#include <iostream>
#include <cstring>
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>

namespace nidps {

PacketCapture::PacketCapture(const std::string& interface, const std::string& filter)
    : interface_(interface), filter_(filter), handle_(nullptr), 
      running_(false), packet_counter_(0) {}

PacketCapture::~PacketCapture() {
    stop();
    if (handle_) {
        pcap_close(handle_);
    }
}

bool PacketCapture::initialize() {
    char errbuf[PCAP_ERRBUF_SIZE];
    
    handle_ = pcap_open_live(interface_.c_str(), BUFSIZ, 1, 1000, errbuf);
    if (!handle_) {
        std::cerr << "Error opening device " << interface_ << ": " << errbuf << std::endl;
        return false;
    }
    
    if (!filter_.empty()) {
        struct bpf_program fp;
        bpf_u_int32 net, mask;
        
        if (pcap_lookupnet(interface_.c_str(), &net, &mask, errbuf) == -1) {
            std::cerr << "Can't get netmask for device " << interface_ << std::endl;
            net = 0;
            mask = 0;
        }
        
        if (pcap_compile(handle_, &fp, filter_.c_str(), 0, net) == -1) {
            std::cerr << "Couldn't parse filter " << filter_ << ": " 
                     << pcap_geterr(handle_) << std::endl;
            return false;
        }
        
        if (pcap_setfilter(handle_, &fp) == -1) {
            std::cerr << "Couldn't install filter " << filter_ << ": " 
                     << pcap_geterr(handle_) << std::endl;
            return false;
        }
        
        pcap_freecode(&fp);
    }
    
    std::cout << "Packet capture initialized on interface: " << interface_ << std::endl;
    return true;
}

void PacketCapture::start(PacketCallback callback) {
    if (running_) {
        return;
    }
    
    callback_ = callback;
    running_ = true;
    capture_thread_ = std::thread(&PacketCapture::captureLoop, this);
    std::cout << "Packet capture started" << std::endl;
}

void PacketCapture::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    std::cout << "Packet capture stopped. Total packets: " << packet_counter_ << std::endl;
}

void PacketCapture::captureLoop() {
    while (running_) {
        struct pcap_pkthdr header;
        const u_char* packet = pcap_next(handle_, &header);
        
        if (packet) {
            PacketPtr pkt = parsePacket(&header, packet);
            if (pkt && callback_) {
                callback_(pkt);
            }
        }
    }
}

PacketPtr PacketCapture::parsePacket(const struct pcap_pkthdr* header, const u_char* packet) {
    PacketPtr pkt = std::make_shared<PacketData>();
    pkt->packet_id = ++packet_counter_;
    pkt->timestamp = std::chrono::system_clock::now();
    
    // Copy raw packet data
    pkt->raw_data.assign(packet, packet + header->caplen);
    
    // Parse IP header (assuming Ethernet frame)
    const struct ip* ip_header = reinterpret_cast<const struct ip*>(packet + 14);
    
    if (header->caplen >= 34) {
        char src_ip[INET_ADDRSTRLEN];
        char dst_ip[INET_ADDRSTRLEN];
        
        inet_ntop(AF_INET, &(ip_header->ip_src), src_ip, INET_ADDRSTRLEN);
        inet_ntop(AF_INET, &(ip_header->ip_dst), dst_ip, INET_ADDRSTRLEN);
        
        pkt->source_ip = src_ip;
        pkt->dest_ip = dst_ip;
        pkt->protocol = ip_header->ip_p;
        
        // Parse transport layer
        int ip_header_len = ip_header->ip_hl * 4;
        const u_char* transport_header = packet + 14 + ip_header_len;
        
        if (pkt->protocol == IPPROTO_TCP && header->caplen >= 34 + ip_header_len) {
            const struct tcphdr* tcp = reinterpret_cast<const struct tcphdr*>(transport_header);
            pkt->source_port = ntohs(tcp->th_sport);
            pkt->dest_port = ntohs(tcp->th_dport);
        } else if (pkt->protocol == IPPROTO_UDP && header->caplen >= 34 + ip_header_len) {
            const struct udphdr* udp = reinterpret_cast<const struct udphdr*>(transport_header);
            pkt->source_port = ntohs(udp->uh_sport);
            pkt->dest_port = ntohs(udp->uh_dport);
        }
    }
    
    // Extract features for ML model
    pkt->features = extractFeatures(pkt);
    
    return pkt;
}

std::vector<float> PacketCapture::extractFeatures(const PacketPtr& packet) {
    std::vector<float> features;
    features.reserve(20);
    
    // Feature 1-2: Packet size features
    features.push_back(static_cast<float>(packet->raw_data.size()));
    features.push_back(static_cast<float>(packet->raw_data.size()) / 1500.0f);
    
    // Feature 3-4: Port features
    features.push_back(static_cast<float>(packet->source_port));
    features.push_back(static_cast<float>(packet->dest_port));
    
    // Feature 5: Protocol
    features.push_back(static_cast<float>(packet->protocol));
    
    // Feature 6-10: Payload statistics
    if (!packet->raw_data.empty()) {
        float sum = 0, mean = 0, variance = 0;
        for (uint8_t byte : packet->raw_data) {
            sum += byte;
        }
        mean = sum / packet->raw_data.size();
        
        for (uint8_t byte : packet->raw_data) {
            variance += (byte - mean) * (byte - mean);
        }
        variance /= packet->raw_data.size();
        
        features.push_back(mean);
        features.push_back(variance);
        features.push_back(static_cast<float>(*std::max_element(packet->raw_data.begin(), packet->raw_data.end())));
        features.push_back(static_cast<float>(*std::min_element(packet->raw_data.begin(), packet->raw_data.end())));
        features.push_back(sum);
    } else {
        features.insert(features.end(), 5, 0.0f);
    }
    
    // Feature 11-15: Entropy and pattern detection
    std::vector<int> byte_freq(256, 0);
    for (uint8_t byte : packet->raw_data) {
        byte_freq[byte]++;
    }
    
    float entropy = 0.0f;
    for (int freq : byte_freq) {
        if (freq > 0) {
            float p = static_cast<float>(freq) / packet->raw_data.size();
            entropy -= p * std::log2(p);
        }
    }
    features.push_back(entropy);
    
    // Additional statistical features
    int zero_count = std::count(packet->raw_data.begin(), packet->raw_data.end(), 0);
    features.push_back(static_cast<float>(zero_count) / packet->raw_data.size());
    
    // Repeating pattern detection
    int repeat_count = 0;
    for (size_t i = 1; i < packet->raw_data.size(); ++i) {
        if (packet->raw_data[i] == packet->raw_data[i-1]) {
            repeat_count++;
        }
    }
    features.push_back(static_cast<float>(repeat_count) / packet->raw_data.size());
    
    // Header to payload ratio
    float header_ratio = 54.0f / packet->raw_data.size();
    features.push_back(header_ratio);
    
    // Time-based feature (hour of day normalized)
    auto time_t = std::chrono::system_clock::to_time_t(packet->timestamp);
    struct tm* tm_info = localtime(&time_t);
    features.push_back(static_cast<float>(tm_info->tm_hour) / 24.0f);
    
    return features;
}

} // namespace nidps
