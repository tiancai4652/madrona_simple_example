#pragma once

#include <madrona/components.hpp>
#include "Astrasim.hpp"
#include "../astra-sim/network_frontend/analytical/congestion_unaware/CongestionUnawareNetworkApi.hh"

namespace madsimple {

// 定义系统配置结构
struct alignas(8) SystemConfig {
    float simulationSpeed = 1.0f;
    float accuracy = 0.95f;
    int maxIterations = 1000;
    int simulationMode[32];
    
    // 网络相关配置
    AstraSim::NetworkConfig networkConfig;
};

struct alignas(8) SystemLayerComponent {
    AstraSim::SystemLayer systemLayer;  // 改为更清晰的命名
    bool isInitialized = false;         // 添加初始化标志
    
    // 添加网络API实例
    AstraSim::CongestionUnawareNetworkApi networkApi;
    
    CUDA_HOST_DEVICE
    SystemLayerComponent() : networkApi(0) {}  // 默认rank为0
};

// 定义 SystemLayerAgent 原型
struct SystemLayerAgent : public madrona::Archetype<
    SystemLayerComponent,
    SystemConfig
> {};

} // namespace madsimple 