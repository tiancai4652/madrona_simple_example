#pragma once

#include <cassert>

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace custom {

// 固定大小的可选值类
template<typename T>
class FixedOptional {
private:
    T value_;
    bool has_value_;

public:
    // 构造函数
    CUDA_HOST_DEVICE
    FixedOptional() : has_value_(false) {}

    CUDA_HOST_DEVICE
    FixedOptional(const T& value) : value_(value), has_value_(true) {}

    // 赋值操作
    CUDA_HOST_DEVICE
    FixedOptional& operator=(const T& value) {
        value_ = value;
        has_value_ = true;
        return *this;
    }

    // 检查是否有值
    CUDA_HOST_DEVICE
    bool has_value() const {
        return has_value_;
    }

    // 获取值
    CUDA_HOST_DEVICE
    const T& value() const {
        assert(has_value_);
        return value_;
    }

    CUDA_HOST_DEVICE
    T& value() {
        assert(has_value_);
        return value_;
    }

    // 重置
    CUDA_HOST_DEVICE
    void reset() {
        has_value_ = false;
    }

    // 比较操作
    CUDA_HOST_DEVICE
    bool operator==(const FixedOptional& other) const {
        if (has_value_ != other.has_value_) {
            return false;
        }
        if (!has_value_) {
            return true;
        }
        return value_ == other.value_;
    }
};

// 创建空可选值
template<typename T>
CUDA_HOST_DEVICE
FixedOptional<T> make_nullopt() {
    return FixedOptional<T>();
}

// 创建有值的可选值
template<typename T>
CUDA_HOST_DEVICE
FixedOptional<T> make_optional(const T& value) {
    return FixedOptional<T>(value);
}

} // namespace custom 