#include "DRL/ReplayBuffer.hpp"
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace drl {

ReplayBuffer::ReplayBuffer(size_t capacity)
    : capacity_(capacity), rng_(std::random_device{}()) {
}

void ReplayBuffer::add(const Experience& experience) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    if (!experience.isValid()) {
        std::cerr << "[ReplayBuffer] Warning: Invalid experience rejected" << std::endl;
        return;
    }
    
    if (buffer_.size() >= capacity_) {
        buffer_.pop_front();
    }
    
    buffer_.push_back(experience);
    total_added_++;
}

void ReplayBuffer::add(const std::vector<float>& state, int action, float reward,
                       const std::vector<float>& next_state, bool done) {
    add(Experience(state, action, reward, next_state, done));
}

std::vector<Experience> ReplayBuffer::sample(size_t batch_size) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    if (batch_size > buffer_.size()) {
        throw std::invalid_argument("Batch size (" + std::to_string(batch_size) + 
                                   ") exceeds buffer size (" + std::to_string(buffer_.size()) + ")");
    }
    
    std::vector<Experience> batch;
    batch.reserve(batch_size);
    
    // Create indices and shuffle
    std::vector<size_t> indices(buffer_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng_);
    
    // Sample batch
    for (size_t i = 0; i < batch_size; ++i) {
        batch.push_back(buffer_[indices[i]]);
    }
    
    total_sampled_ += batch_size;
    return batch;
}

size_t ReplayBuffer::size() const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    return buffer_.size();
}

bool ReplayBuffer::isReady(size_t min_size) const {
    return size() >= min_size;
}

void ReplayBuffer::clear() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    buffer_.clear();
}

double ReplayBuffer::getUtilization() const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    return static_cast<double>(buffer_.size()) / capacity_;
}

bool ReplayBuffer::saveToFile(const std::string& filepath) const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    try {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        
        // Write buffer size
        size_t size = buffer_.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        
        // Write each experience
        for (const auto& exp : buffer_) {
            // Write state size and data
            size_t state_size = exp.state.size();
            file.write(reinterpret_cast<const char*>(&state_size), sizeof(state_size));
            file.write(reinterpret_cast<const char*>(exp.state.data()), state_size * sizeof(float));
            
            // Write action
            file.write(reinterpret_cast<const char*>(&exp.action), sizeof(exp.action));
            
            // Write reward
            file.write(reinterpret_cast<const char*>(&exp.reward), sizeof(exp.reward));
            
            // Write next_state size and data
            size_t next_state_size = exp.next_state.size();
            file.write(reinterpret_cast<const char*>(&next_state_size), sizeof(next_state_size));
            file.write(reinterpret_cast<const char*>(exp.next_state.data()), next_state_size * sizeof(float));
            
            // Write done flag
            file.write(reinterpret_cast<const char*>(&exp.done), sizeof(exp.done));
        }
        
        file.close();
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ReplayBuffer] Error saving to file: " << e.what() << std::endl;
        return false;
    }
}

bool ReplayBuffer::loadFromFile(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    try {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        
        buffer_.clear();
        
        // Read buffer size
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        
        // Read each experience
        for (size_t i = 0; i < size; ++i) {
            Experience exp;
            
            // Read state
            size_t state_size;
            file.read(reinterpret_cast<char*>(&state_size), sizeof(state_size));
            exp.state.resize(state_size);
            file.read(reinterpret_cast<char*>(exp.state.data()), state_size * sizeof(float));
            
            // Read action
            file.read(reinterpret_cast<char*>(&exp.action), sizeof(exp.action));
            
            // Read reward
            file.read(reinterpret_cast<char*>(&exp.reward), sizeof(exp.reward));
            
            // Read next_state
            size_t next_state_size;
            file.read(reinterpret_cast<char*>(&next_state_size), sizeof(next_state_size));
            exp.next_state.resize(next_state_size);
            file.read(reinterpret_cast<char*>(exp.next_state.data()), next_state_size * sizeof(float));
            
            // Read done flag
            file.read(reinterpret_cast<char*>(&exp.done), sizeof(exp.done));
            
            buffer_.push_back(exp);
        }
        
        file.close();
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ReplayBuffer] Error loading from file: " << e.what() << std::endl;
        return false;
    }
}

} // namespace drl
