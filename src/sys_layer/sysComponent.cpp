#pragma once

#include <madrona/components.hpp>
#include "Astrasim.hpp"

namespace madsimple {

// 定义系统配置结构
struct SystemConfig {
    float simulationSpeed = 1.0f;
    float accuracy = 0.95f;
    int maxIterations = 1000;
    std::string simulationMode = "standard";
};

struct SystemLayerComponent {
    // 指向原始AstraSim系统层对象的指针
    AstraSim::SystemLayer* systemLayerPtr = nullptr;
};

// 定义 SystemLayerAgent 原型
struct SystemLayerAgent : public madrona::Archetype<
    SystemLayerComponent,
    SystemConfig
> {};
} // namespace madsimple