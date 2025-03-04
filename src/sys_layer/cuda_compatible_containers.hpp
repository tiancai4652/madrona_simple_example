#pragma once

#include <cstring>
#include <cassert>

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace custom {

// 简单的字符串类
template<size_t MaxSize>
class FixedString {
private:
    char data_[MaxSize];
    size_t length_;

public:
    // 构造函数
    CUDA_HOST_DEVICE
    FixedString() : length_(0) {
        data_[0] = '\0';
    }

    CUDA_HOST_DEVICE
    FixedString(const char* str) {
        assign(str);
    }

    // 赋值操作
    CUDA_HOST_DEVICE
    void assign(const char* str) {
        length_ = 0;
        while (str[length_] != '\0' && length_ < MaxSize - 1) {
            data_[length_] = str[length_];
            ++length_;
        }
        data_[length_] = '\0';
    }

    // 访问操作
    CUDA_HOST_DEVICE
    const char* c_str() const { return data_; }

    CUDA_HOST_DEVICE
    size_t length() const { return length_; }

    CUDA_HOST_DEVICE
    bool empty() const { return length_ == 0; }

    // 比较操作
    CUDA_HOST_DEVICE
    bool operator==(const FixedString& other) const {
        if (length_ != other.length_) {
            return false;
        }
        for (size_t i = 0; i < length_; i++) {
            if (data_[i] != other.data_[i]) {
                return false;
            }
        }
        return true;
    }

    CUDA_HOST_DEVICE
    bool operator==(const char* str) const {
        size_t i = 0;
        while (i < length_ && str[i] != '\0') {
            if (data_[i] != str[i]) {
                return false;
            }
            i++;
        }
        return i == length_ && str[i] == '\0';
    }
};

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

// 键值对结构体
template<typename K, typename V>
struct Pair {
    K first;
    V second;

    CUDA_HOST_DEVICE
    Pair() {}

    CUDA_HOST_DEVICE
    Pair(const K& k, const V& v) : first(k), second(v) {}
};

} // namespace custom