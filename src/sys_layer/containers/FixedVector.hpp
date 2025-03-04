#pragma once

#include <cassert>

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace custom {

// 固定大小的向量类
template<typename T, size_t MaxSize>
class FixedVector {
private:
    T data_[MaxSize];
    size_t size_;

public:
    // 构造函数
    CUDA_HOST_DEVICE
    FixedVector() : size_(0) {}

    // 添加元素
    CUDA_HOST_DEVICE
    bool push_back(const T& value) {
        if (size_ >= MaxSize) {
            return false;
        }
        data_[size_++] = value;
        return true;
    }

    // 访问操作
    CUDA_HOST_DEVICE
    T& operator[](size_t index) {
        assert(index < size_);
        return data_[index];
    }

    CUDA_HOST_DEVICE
    const T& operator[](size_t index) const {
        assert(index < size_);
        return data_[index];
    }

    CUDA_HOST_DEVICE
    size_t size() const { return size_; }

    CUDA_HOST_DEVICE
    bool empty() const { return size_ == 0; }

    CUDA_HOST_DEVICE
    bool full() const { return size_ >= MaxSize; }

    // 迭代器支持
    CUDA_HOST_DEVICE
    T* begin() { return data_; }

    CUDA_HOST_DEVICE
    const T* begin() const { return data_; }

    CUDA_HOST_DEVICE
    T* end() { return data_ + size_; }

    CUDA_HOST_DEVICE
    const T* end() const { return data_ + size_; }

    // 清空
    CUDA_HOST_DEVICE
    void clear() { size_ = 0; }
};

} // namespace custom 