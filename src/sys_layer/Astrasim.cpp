#include "Astrasim.hpp"
#include <iostream>
#include "HelperTool.hpp"
namespace AstraSim {

SystemLayer::SystemLayer() 
    : initialized_(false), status_("Not initialized") {
    // 初始化一些默认参数
    parameters_.push_back({"simulationSpeed", 1.0f});
    parameters_.push_back({"accuracy", 0.95f});
    parameters_.push_back({"complexity", 3.0f});
    HelperTool::stringToIntArray("No simulation run yet", currentResult_.description);
    // currentResult_ = {0.0f, 0.0f, "No simulation run yet"};
}

SystemLayer::~SystemLayer() {
    if (initialized_) {
        shutdown();
    }
}

void SystemLayer::initialize() {
    std::cout << "SystemLayer: 初始化系统层..." << std::endl;
    initialized_ = true;
    status_ = "Initialized";
}

void SystemLayer::update(float deltaTime) {
    if (!initialized_) {
        std::cout << "SystemLayer: 警告 - 尝试更新未初始化的系统" << std::endl;
        return;
    }
    
    // 简单的更新逻辑
    currentResult_.time += deltaTime;
    
    // 使用一些参数进行模拟计算
    float speed = getParameter("simulationSpeed");
    float accuracy = getParameter("accuracy");
    
    currentResult_.value += deltaTime * speed * accuracy;
    
    std::cout << "SystemLayer: 更新完成，当前时间: " << currentResult_.time 
              << ", 当前值: " << currentResult_.value << std::endl;
}

void SystemLayer::shutdown() {
    if (!initialized_) {
        return;
    }
    
    std::cout << "SystemLayer: 关闭系统层..." << std::endl;
    initialized_ = false;
    status_ = "Shutdown";
}

void SystemLayer::runSimulation(float duration) {
    if (!initialized_) {
        initialize();
    }
    
    std::cout << "SystemLayer: 开始运行模拟，持续时间: " << duration << std::endl;
    
    // 模拟运行，以0.1为步长
    float timeStep = 0.1f;
    for (float t = 0; t < duration; t += timeStep) {
        update(timeStep);
    }
    std::string message = "模拟完成，总时长: " + std::to_string(duration);
    HelperTool::stringToIntArray(message.c_str(), currentResult_.description);
    // currentResult_.description = "模拟完成，总时长: " + std::to_string(duration);
    // std::cout << "SystemLayer: 模拟完成" << std::endl;
}

SimulationData SystemLayer::getSimulationResult() const {
    return currentResult_;
}

bool SystemLayer::isInitialized() const {
    return initialized_;
}

std::string SystemLayer::getStatus() const {
    return status_;
}

void SystemLayer::setParameter(const std::string& name, float value) {
    for (auto& param : parameters_) {
        if (param.first == name) {
            param.second = value;
            std::cout << "SystemLayer: 参数 '" << name << "' 设置为 " << value << std::endl;
            return;
        }
    }
    
    // 如果参数不存在，添加它
    parameters_.push_back({name, value});
    std::cout << "SystemLayer: 添加新参数 '" << name << "' 值为 " << value << std::endl;
}

float SystemLayer::getParameter(const std::string& name) const {
    for (const auto& param : parameters_) {
        if (param.first == name) {
            return param.second;
        }
    }
    
    std::cout << "SystemLayer: 警告 - 参数 '" << name << "' 不存在，返回默认值0" << std::endl;
    return 0.0f;
}

} // namespace AstraSim 