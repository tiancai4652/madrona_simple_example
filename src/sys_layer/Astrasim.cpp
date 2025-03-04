#include "Astrasim.hpp"
#include <iostream>
#include "HelperTool.hpp"

namespace AstraSim {

CUDA_HOST_DEVICE
SystemLayer::SystemLayer() 
    : initialized_(false) {
    status_.assign("Not initialized");
    
    // 初始化一些默认参数
    parameters_.push_back(custom::Pair<custom::FixedString<64>, float>(
        custom::FixedString<64>("simulationSpeed"), 1.0f));
    parameters_.push_back(custom::Pair<custom::FixedString<64>, float>(
        custom::FixedString<64>("accuracy"), 0.95f));
    parameters_.push_back(custom::Pair<custom::FixedString<64>, float>(
        custom::FixedString<64>("complexity"), 3.0f));
        
    HelperTool::stringToIntArray("No simulation run yet", currentResult_.description);
    currentResult_.time = 0.0f;
    currentResult_.value = 0.0f;
}

CUDA_HOST_DEVICE
SystemLayer::~SystemLayer() {
    if (initialized_) {
        shutdown();
    }
}

CUDA_HOST_DEVICE
void SystemLayer::initialize() {
    #ifndef __CUDACC__
    std::cout << "SystemLayer: 初始化系统层..." << std::endl;
    #endif
    initialized_ = true;
    status_.assign("Initialized");
}

CUDA_HOST_DEVICE
void SystemLayer::update(float deltaTime) {
    if (!initialized_) {
        #ifndef __CUDACC__
        std::cout << "SystemLayer: 警告 - 尝试更新未初始化的系统" << std::endl;
        #endif
        return;
    }
    
    currentResult_.time += deltaTime;
    float speed = getParameter("simulationSpeed");
    float accuracy = getParameter("accuracy");
    currentResult_.value += deltaTime * speed * accuracy;
    
    #ifndef __CUDACC__
    std::cout << "SystemLayer: 更新完成，当前时间: " << currentResult_.time 
              << ", 当前值: " << currentResult_.value << std::endl;
    #endif
}

CUDA_HOST_DEVICE
void SystemLayer::shutdown() {
    if (!initialized_) {
        return;
    }
    
    #ifndef __CUDACC__
    std::cout << "SystemLayer: 关闭系统层..." << std::endl;
    #endif
    initialized_ = false;
    status_.assign("Shutdown");
}

CUDA_HOST_DEVICE
void SystemLayer::runSimulation(float duration) {
    if (!initialized_) {
        initialize();
    }
    
    #ifndef __CUDACC__
    std::cout << "SystemLayer: 开始运行模拟，持续时间: " << duration << std::endl;
    #endif
    
    float timeStep = 0.1f;
    for (float t = 0; t < duration; t += timeStep) {
        update(timeStep);
    }

    // 自定义数字转字符串实现
    char buffer[256];
    int pos = 0;
    
    // 添加固定字符串部分
    const char* prefix = "模拟完成，总时长: ";
    while (*prefix) {
        buffer[pos++] = *prefix++;
    }
    
    // 转换数字部分
    int int_part = (int)duration;
    int frac_part = (int)((duration - int_part) * 100);
    
    // 转换整数部分
    if (int_part == 0) {
        buffer[pos++] = '0';
    } else {
        char temp[20];
        int temp_pos = 0;
        while (int_part > 0) {
            temp[temp_pos++] = '0' + (int_part % 10);
            int_part /= 10;
        }
        while (temp_pos > 0) {
            buffer[pos++] = temp[--temp_pos];
        }
    }
    
    // 添加小数点
    buffer[pos++] = '.';
    
    // 添加小数部分（保证两位）
    buffer[pos++] = '0' + (frac_part / 10);
    buffer[pos++] = '0' + (frac_part % 10);
    
    // 添加结束符
    buffer[pos] = '\0';
    
    custom::FixedString<256> message;
    message.assign(buffer);
    
    HelperTool::stringToIntArray(message.c_str(), currentResult_.description);
}

CUDA_HOST_DEVICE
SimulationData SystemLayer::getSimulationResult() const {
    return currentResult_;
}

CUDA_HOST_DEVICE
bool SystemLayer::isInitialized() const {
    return initialized_;
}

CUDA_HOST_DEVICE
const char* SystemLayer::getStatus() const {
    return status_.c_str();
}

CUDA_HOST_DEVICE
void SystemLayer::setParameter(const char* name, float value) {
    custom::FixedString<64> paramName(name);
    
    for (size_t i = 0; i < parameters_.size(); ++i) {
        if (parameters_[i].first == paramName) {
            parameters_[i].second = value;
            #ifndef __CUDACC__
            std::cout << "SystemLayer: 参数 '" << name << "' 设置为 " << value << std::endl;
            #endif
            return;
        }
    }
    
    parameters_.push_back(custom::Pair<custom::FixedString<64>, float>(paramName, value));
    #ifndef __CUDACC__
    std::cout << "SystemLayer: 添加新参数 '" << name << "' 值为 " << value << std::endl;
    #endif
}

CUDA_HOST_DEVICE
float SystemLayer::getParameter(const char* name) const {
    custom::FixedString<64> paramName(name);
    
    for (size_t i = 0; i < parameters_.size(); ++i) {
        if (parameters_[i].first == paramName) {
            return parameters_[i].second;
        }
    }
    
    #ifndef __CUDACC__
    std::cout << "SystemLayer: 警告 - 参数 '" << name << "' 不存在，返回默认值0" << std::endl;
    #endif
    return 0.0f;
}

} // namespace AstraSim