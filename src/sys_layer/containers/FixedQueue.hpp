#pragma once

#include <cassert>

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace custom {

// 固定大小的循环队列类
template<typename T, size_t MaxSize>
class FixedQueue {
private:
    T data_[MaxSize];
    size_t front_;    // 队列头
    size_t back_;     // 队列尾
    size_t size_;     // 当前大小

public:
    // 构造函数
    CUDA_HOST_DEVICE
    FixedQueue() : front_(0), back_(0), size_(0) {}

    // 入队操作
    CUDA_HOST_DEVICE
    bool push(const T& value) {
        if (size_ >= MaxSize) {
            return false;
        }
        data_[back_] = value;
        back_ = (back_ + 1) % MaxSize;
        size_++;
        return true;
    }

    // 出队操作
    CUDA_HOST_DEVICE
    bool pop() {
        if (empty()) {
            return false;
        }
        front_ = (front_ + 1) % MaxSize;
        size_--;
        return true;
    }

    // 获取队首元素
    CUDA_HOST_DEVICE
    T& front() {
        assert(!empty());
        return data_[front_];
    }

    CUDA_HOST_DEVICE
    const T& front() const {
        assert(!empty());
        return data_[front_];
    }

    // 获取队尾元素
    CUDA_HOST_DEVICE
    T& back() {
        assert(!empty());
        return data_[(back_ - 1 + MaxSize) % MaxSize];
    }

    CUDA_HOST_DEVICE
    const T& back() const {
        assert(!empty());
        return data_[(back_ - 1 + MaxSize) % MaxSize];
    }

    // 状态查询
    CUDA_HOST_DEVICE
    size_t size() const { return size_; }

    CUDA_HOST_DEVICE
    bool empty() const { return size_ == 0; }

    CUDA_HOST_DEVICE
    bool full() const { return size_ >= MaxSize; }

    // 清空队列
    CUDA_HOST_DEVICE
    void clear() {
        front_ = back_ = size_ = 0;
    }
};

} // namespace custom 