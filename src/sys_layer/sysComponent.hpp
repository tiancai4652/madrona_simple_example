#pragma once

#include <madrona/components.hpp>
#include "Astrasim.hpp"

namespace madsimple {

// 定义系统配置结构
struct SystemConfig {
    float simulationSpeed = 1.0f;
    float accuracy = 0.95f;
    int maxIterations = 1000;
    int simulationMode[32];
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

// 将字符串转换为整数数组
void stringToIntArray(const char *str, int *intArray) {
   size_t i = 0;
       while (str[i] != '\0') {
         intArray[i] = static_cast<int>(str[i]);
       ++i;
   }
}

// 将整数数组转换为字符串
void intArrayToString(const int *intArray, char *str, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        str[i] = static_cast<char>(intArray[i]);
    }
    str[length] = '\0'; // 确保字符串以空字符结尾
}

} // namespace madsimple 