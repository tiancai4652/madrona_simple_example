#pragma once

#include <madrona/components.hpp>
#include "Astrasim.hpp"

namespace HelperTool {

// 将字符串转换为整数数组
void stringToIntArray(const char *str, int *intArray);

// 将整数数组转换为字符串
void intArrayToString(const int *intArray, char *str, size_t length);

} // namespace HelperTool