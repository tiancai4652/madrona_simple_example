#include "Astrasim.hpp"
#include <iostream>
#include "HelperTool.hpp"

namespace AstraSim {


SystemLayer::SystemLayer() 
    : initialized_(false),
      chunkIdGenerator_(),
      currentResult_() {
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


SystemLayer::~SystemLayer() {
    if (initialized_) {
        shutdown();
    }
}


void SystemLayer::initialize() {
    printf("SystemLayer: 初始化系统层...\n");
    initialized_ = true;
    status_.assign("Initialized");
    
    // 使用成员变量，不要创建新的局部变量
    // chunkIdGenerator_.increment_send_id();
    // chunkIdGenerator_.increment_recv_id();
}


void SystemLayer::update(float deltaTime) {
    if (!initialized_) {
        printf("SystemLayer: 警告 - 尝试更新未初始化的系统\n");
        return;
    }
    
    currentResult_.time += deltaTime;
    float speed = getParameter("simulationSpeed");
    float accuracy = getParameter("accuracy");
    currentResult_.value += deltaTime * speed * accuracy;
    
    printf("SystemLayer: 更新完成，当前时间: %f, 当前值: %f\n", 
           currentResult_.time, currentResult_.value);
}


void SystemLayer::shutdown() {
    if (!initialized_) {
        return;
    }
    
    printf("SystemLayer: 关闭系统层...\n");
    initialized_ = false;
    status_.assign("Shutdown");
}


void SystemLayer::runSimulation(float duration) {
    if (!initialized_) {
        initialize();
    }
    
    printf("SystemLayer: 开始运行模拟，持续时间: %f\n", duration);
    
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


SimulationData SystemLayer::getSimulationResult() const {
    return currentResult_;
}


bool SystemLayer::isInitialized() const {
    return initialized_;
}


const char* SystemLayer::getStatus() const {
    return status_.c_str();
}


void SystemLayer::setParameter(const char* name, float value) {
    custom::FixedString<64> paramName(name);
    
    for (size_t i = 0; i < parameters_.size(); ++i) {
        if (parameters_[i].first == paramName) {
            parameters_[i].second = value;
            printf("SystemLayer: 参数 '%s' 设置为 %f\n", name, value);
            return;
        }
    }
    
    parameters_.push_back(custom::Pair<custom::FixedString<64>, float>(paramName, value));
    printf("SystemLayer: 添加新参数 '%s' 值为 %f\n", name, value);
}


float SystemLayer::getParameter(const char* name) const {
    custom::FixedString<64> paramName(name);
    
    for (size_t i = 0; i < parameters_.size(); ++i) {
        if (parameters_[i].first == paramName) {
            return parameters_[i].second;
        }
    }
    
    printf("SystemLayer: 警告 - 参数 '%s' 不存在，返回默认值0\n", name);
    return 0.0f;
}

} // namespace AstraSim