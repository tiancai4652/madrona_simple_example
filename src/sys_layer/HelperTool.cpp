#include "HelperTool.hpp"

namespace HelperTool {

void stringToIntArray(const char *str, int *intArray) {
    size_t i = 0;
    while (str[i] != '\0') {
        intArray[i] = static_cast<int>(str[i]);
        ++i;
    }
}

void intArrayToString(const int *intArray, char *str, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        str[i] = static_cast<char>(intArray[i]);
    }
    str[length] = '\0'; // 确保字符串以空字符结尾
}

} // namespace HelperTool