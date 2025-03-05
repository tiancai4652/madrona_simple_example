#pragma once

#include <cassert>

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace custom {

// 固定大小的映射类
template<typename Key, typename Value, size_t MaxSize>
class FixedMap {
private:
    struct Entry {
        Key key;
        Value value;
        bool used;
        
        CUDA_HOST_DEVICE
        Entry() : used(false) {}
    };

    Entry data_[MaxSize];
    size_t size_;

public:
    // 构造函数
    CUDA_HOST_DEVICE
    FixedMap() : size_(0) {}

    // 插入操作
    CUDA_HOST_DEVICE
    bool insert(const Key& key, const Value& value) {
        if (size_ >= MaxSize) {
            return false;
        }

        // 查找已存在的key或空位
        for (size_t i = 0; i < MaxSize; i++) {
            if (data_[i].used && data_[i].key == key) {
                data_[i].value = value;
                return true;
            }
            if (!data_[i].used) {
                data_[i].key = key;
                data_[i].value = value;
                data_[i].used = true;
                size_++;
                return true;
            }
        }
        return false;
    }

    // 查找操作
    CUDA_HOST_DEVICE
    Value* find(const Key& key) {
        for (size_t i = 0; i < MaxSize; i++) {
            if (data_[i].used && data_[i].key == key) {
                return &data_[i].value;
            }
        }
        return nullptr;
    }

    CUDA_HOST_DEVICE
    const Value* find(const Key& key) const {
        for (size_t i = 0; i < MaxSize; i++) {
            if (data_[i].used && data_[i].key == key) {
                return &data_[i].value;
            }
        }
        return nullptr;
    }

    // 删除操作
    CUDA_HOST_DEVICE
    bool erase(const Key& key) {
        for (size_t i = 0; i < MaxSize; i++) {
            if (data_[i].used && data_[i].key == key) {
                data_[i].used = false;
                size_--;
                return true;
            }
        }
        return false;
    }

    // 访问操作
    CUDA_HOST_DEVICE
    Value& operator[](const Key& key) {
        Value* found = find(key);
        if (found) {
            return *found;
        }
        
        // 如果不存在则插入默认值
        for (size_t i = 0; i < MaxSize; i++) {
            if (!data_[i].used) {
                data_[i].key = key;
                data_[i].used = true;
                size_++;
                return data_[i].value;
            }
        }
        assert(false); // Map is full
        return data_[0].value;
    }

    // 状态查询
    CUDA_HOST_DEVICE
    size_t size() const { return size_; }

    CUDA_HOST_DEVICE
    bool empty() const { return size_ == 0; }

    CUDA_HOST_DEVICE
    bool full() const { return size_ >= MaxSize; }

    // 清空映射
    CUDA_HOST_DEVICE
    void clear() {
        for (size_t i = 0; i < MaxSize; i++) {
            data_[i].used = false;
        }
        size_ = 0;
    }
};

} // namespace custom 