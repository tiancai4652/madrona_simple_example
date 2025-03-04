#pragma once

// #include <string>
// #include <vector>
#include "cuda_compatible_containers.hpp"

namespace AstraSim {

// 定义一个简单的数据结构用于模拟
struct alignas(8) SimulationData {
    float time;
    float value;
    int description[1024];

    CUDA_HOST_DEVICE SimulationData() 
        : time(0.0f), value(0.0f) {
        for (int i = 0; i < 1024; i++) {
            description[i] = 0;
        }
    }
};

// SystemLayer 类 - 移除virtual关键字
class SystemLayer {
public:
    CUDA_HOST_DEVICE SystemLayer();
    CUDA_HOST_DEVICE ~SystemLayer();

    // 基本方法
    CUDA_HOST_DEVICE void initialize();
    CUDA_HOST_DEVICE void update(float deltaTime);
    CUDA_HOST_DEVICE void shutdown();
    
    // 模拟相关方法
    CUDA_HOST_DEVICE void runSimulation(float duration);
    CUDA_HOST_DEVICE SimulationData getSimulationResult() const;
    
    // 状态查询方法
    CUDA_HOST_DEVICE bool isInitialized() const;
    CUDA_HOST_DEVICE const char* getStatus() const;
    
    // 配置方法
    CUDA_HOST_DEVICE void setParameter(const char* name, float value);
    CUDA_HOST_DEVICE float getParameter(const char* name) const;

private:
    bool initialized_;
    // std::string status_;
    // std::vector<std::pair<std::string, float>> parameters_;
    custom::FixedString<256> status_;
    custom::FixedVector<custom::Pair<custom::FixedString<64>, float>, 32> parameters_;

    SimulationData currentResult_;
};

} // namespace AstraSim 