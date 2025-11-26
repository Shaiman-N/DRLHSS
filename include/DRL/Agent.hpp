#pragma once
#include <vector>
#include <string>
#include "ExperienceReplay.hpp"

class Agent
{
    public :
        Agent(size_t state_dim, size_t action_dim, size_t replay_capacity);
        void step(const std::vector<float>& state, int action, float reward, const std::vector<float>& next_state, bool done);
        std::vector<float> selectAction(const std::vector<float>& state, float epsilon);
        void train();
        void saveModel(const std::string& path);
        void loadModel(const std::string& path);

    private :
        size_t state_dim_;
        size_t action_dim_;
        ExperienceReplay replay_buffer_;
        // internal model representation omitted for brevity; integrate NN here
};