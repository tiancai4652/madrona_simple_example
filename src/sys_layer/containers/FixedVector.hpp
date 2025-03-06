#pragma once

#include <cassert>

namespace custom {

// 固定大小的向量类
template<typename T, size_t MaxSize>
class FixedVector {
private:
    T data_[MaxSize];
    size_t size_;

public:
    // 构造函数
    
    FixedVector() : size_(0) {}

    // 添加元素
    
    bool push_back(const T& value) {
        if (size_ >= MaxSize) {
            return false;
        }
        data_[size_++] = value;
        return true;
    }

    // 访问操作
    
    T& operator[](size_t index) {
        assert(index < size_);
        return data_[index];
    }

    
    const T& operator[](size_t index) const {
        assert(index < size_);
        return data_[index];
    }

    
    size_t size() const { return size_; }

    
    bool empty() const { return size_ == 0; }

    
    bool full() const { return size_ >= MaxSize; }

    // 迭代器支持
    
    T* begin() { return data_; }

    
    const T* begin() const { return data_; }

    
    T* end() { return data_ + size_; }

    
    const T* end() const { return data_ + size_; }

    // 清空
    
    void clear() { size_ = 0; }
};

} // namespace custom 