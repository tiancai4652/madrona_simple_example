#pragma once

#include <madrona/components.hpp>
#include "Astrasim.hpp"

namespace madsimple {

// 定义系统配置结构
struct alignas(8) SystemConfig {
    float simulationSpeed = 1.0f;
    float accuracy = 0.95f;
    int maxIterations = 1000;
    int simulationMode[32];
};

struct alignas(8) SystemLayerComponent {
    AstraSim::SystemLayer systemLayer;  // 改为更清晰的命名
    bool isInitialized = false;         // 添加初始化标志
};

// 定义 SystemLayerAgent 原型
struct SystemLayerAgent : public madrona::Archetype<
    SystemLayerComponent,
    SystemConfig
> {};

} // namespace madsimple 