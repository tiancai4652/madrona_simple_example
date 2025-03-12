#pragma once
#include <cstddef>

namespace custom {
template<typename T>
class CustomOptional {
private:
    T* value_;      // 指针指向值，使用动态内存
    bool hasValue_; // 是否有值的标志

    // 清理现有值
    void destroy() {
        if (hasValue_) {
            value_->~T();  // 调用析构函数
            delete value_;
            hasValue_ = false;
        }
    }

public:
    // 默认构造函数（空值）
    CustomOptional() : value_(nullptr), hasValue_(false) {}

    // 值初始化构造函数
    CustomOptional(const T& value) : value_(new T(value)), hasValue_(true) {}

    // 拷贝构造函数
    CustomOptional(const CustomOptional& other) : value_(nullptr), hasValue_(false) {
        if (other.hasValue_) {
            value_ = new T(*other.value_);
            hasValue_ = true;
        }
    }

    // 析构函数
    ~CustomOptional() {
        destroy();
    }

    // 赋值运算符（拷贝）
    CustomOptional& operator=(const CustomOptional& other) {
        if (this != &other) {
            destroy();
            if (other.hasValue_) {
                value_ = new T(*other.value_);
                hasValue_ = true;
            }
        }
        return *this;
    }

    // 赋值运算符（值）
    CustomOptional& operator=(const T& value) {
        destroy();
        value_ = new T(value);
        hasValue_ = true;
        return *this;
    }

    // 检查是否有值
    bool has_value() const { return hasValue_; }
    explicit operator bool() const { return hasValue_; }

    // 获取值（无异常替代方案）
    T& value() {
        if (!hasValue_) {
            // CUDA不支持异常，这里返回默认构造的值
            // 或者可以添加断言/调试标记，实际使用时需根据需求调整
            static T defaultValue = T();
            return defaultValue;
        }
        return *value_;
    }

    const T& value() const {
        if (!hasValue_) {
            static T defaultValue = T();
            return defaultValue;
        }
        return *value_;
    }

    // 获取值或默认值
    T value_or(const T& defaultValue) const {
        return hasValue_ ? *value_ : defaultValue;
    }

    // 解引用运算符
    T& operator*() { return *value_; }
    const T& operator*() const { return *value_; }

    T* operator->() { return value_; }
    const T* operator->() const { return value_; }

    // 清空值
    void reset() {
        destroy();
    }

    // 就地构造
    template<typename... Args>
    T& emplace(Args&&... args) {
        destroy();
        value_ = new T(args...);
        hasValue_ = true;
        return *value_;
    }

    // 比较运算符（可选，用于完整性）
    bool operator==(const CustomOptional& other) const {
        if (hasValue_ != other.hasValue_) return false;
        if (!hasValue_) return true;
        return *value_ == *other.value_;
    }

    bool operator!=(const CustomOptional& other) const {
        return !(*this == other);
    }
};
}
// // 测试代码
// int main() {
//     CustomOptional<int> opt1;          // 空值
//     CustomOptional<int> opt2(42);      // 有值
//     CustomOptional<int> opt3(opt2);    // 拷贝

//     printf("opt1 has value: %d\n", opt1.has_value());  // 0
//     printf("opt2 value: %d\n", opt2.value());          // 42
//     printf("opt3 value: %d\n", opt3.value());          // 42

//     opt1 = 10;  // 赋值
//     printf("opt1 value after assign: %d\n", *opt1);    // 10

//     printf("opt1 value_or 5: %d\n", opt1.value_or(5)); // 10
//     opt1.reset();
//     printf("opt1 value_or 5 after reset: %d\n", opt1.value_or(5)); // 5

//     opt1.emplace(20);  // 就地构造
//     printf("opt1 value after emplace: %d\n", opt1.value()); // 20

//     return 0;
// }