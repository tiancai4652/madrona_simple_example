#pragma once

#include <cassert>
#include <cstddef>
#include "FixedTuple.hpp"
#include "Pair.hpp"

namespace custom {

// 固定大小的键值对映射类
template<typename K, typename V, size_t MaxSize>
class FixedMap {
private:
    struct Entry : public Pair<K, V> {
        bool occupied;
        
        Entry() : Pair<K, V>(), occupied(false) {}
        Entry(const K& k, const V& v) : Pair<K, V>(k, v), occupied(true) {}
    };

public:
    class Iterator {
    private:
        Entry* data_;
        size_t pos_;
        size_t max_size_;

        void find_next_valid() {
            while (pos_ < max_size_ && !data_[pos_].occupied) {
                pos_++;
            }
        }

    public:
        Iterator(Entry* data, size_t pos, size_t max_size)
            : data_(data), pos_(pos), max_size_(max_size) {
            find_next_valid();
        }

        Iterator& operator++() {
            if (pos_ < max_size_) {
                pos_++;
                find_next_valid();
            }
            return *this;
        }

        bool operator!=(const Iterator& other) const {
            return pos_ != other.pos_;
        }

        bool operator==(const Iterator& other) const {
            return pos_ == other.pos_;
        }

        Entry& operator*() {
            return data_[pos_];
        }

        const Entry& operator*() const {
            return data_[pos_];
        }

        Entry* operator->() {
            return &data_[pos_];
        }

        const Entry* operator->() const {
            return &data_[pos_];
        }
    };

private:
    Entry data_[MaxSize];
    size_t size_;

public:
    // 构造函数
    FixedMap() : size_(0) {}

    Pair<Iterator, bool> emplace(const K& key, const V& value) {
        for (size_t i = 0; i < MaxSize; i++) {
            if (data_[i].occupied && data_[i].first == key) {
                data_[i].second = value;
                return {Iterator(data_, i, MaxSize), false};
            }
        }

        for (size_t i = 0; i < MaxSize; i++) {
            if (!data_[i].occupied) {
                data_[i].first = key;
                data_[i].second = value;
                data_[i].occupied = true;
                size_++;
                return {Iterator(data_, i, MaxSize), true};
            }
        }

        return {end(), false};
    }

    // 插入或更新键值对
    bool insert(const K& key, const V& value) {
        if (size_ >= MaxSize) {
            // 先查找是否已存在该key
            for (size_t i = 0; i < MaxSize; i++) {
                if (data_[i].occupied && data_[i].first == key) {
                    data_[i].second = value;
                    return true;
                }
            }
            return false;  // 空间已满且无法更新
        }

        // 查找可用位置或已存在的key
        for (size_t i = 0; i < MaxSize; i++) {
            if (!data_[i].occupied) {
                data_[i].first = key;
                data_[i].second = value;
                data_[i].occupied = true;
                size_++;
                return true;
            }
            if (data_[i].occupied && data_[i].first == key) {
                data_[i].second = value;
                return true;
            }
        }

        return false;
    }

    // 查找值
    bool find(const K& key, V& out_value) const {
        for (size_t i = 0; i < MaxSize; i++) {
            if (data_[i].occupied && data_[i].first == key) {
                out_value = data_[i].second;
                return true;
            }
        }
        return false;
    }

    // 查找并返回迭代器
    Iterator find(const K& key) {
        for (size_t i = 0; i < MaxSize; i++) {
            if (data_[i].occupied && data_[i].first == key) {
                return Iterator(data_, i, MaxSize);
            }
        }
        return end();
    }

    // 删除键值对
    bool erase(const K& key) {
        for (size_t i = 0; i < MaxSize; i++) {
            if (data_[i].occupied && data_[i].first == key) {
                data_[i].occupied = false;
                size_--;
                return true;
            }
        }
        return false;
    }

    // 访问操作符 - 如果key不存在会插入默认值
    V& operator[](const K& key) {
        for (size_t i = 0; i < MaxSize; i++) {
            if (data_[i].occupied && data_[i].first == key) {
                return data_[i].second;
            }
        }
        
        // 找到第一个空位
        for (size_t i = 0; i < MaxSize; i++) {
            if (!data_[i].occupied) {
                data_[i].first = key;
                data_[i].occupied = true;
                size_++;
                return data_[i].second;
            }
        }
        
        assert(false && "Map is full");
        return data_[0].second; // 不会执行到这里，只是为了避免编译警告
    }

    // 基本操作
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    bool full() const { return size_ >= MaxSize; }

    // 清空
    void clear() {
        for (size_t i = 0; i < MaxSize; i++) {
            data_[i].occupied = false;
        }
        size_ = 0;
    }

    Iterator begin() { return Iterator(data_, 0, MaxSize); }
    Iterator end() { return Iterator(data_, MaxSize, MaxSize); }
    const Iterator begin() const { return Iterator(data_, 0, MaxSize); }
    const Iterator end() const { return Iterator(data_, MaxSize, MaxSize); }
};

} // namespace custom

// custom::FixedMap<int, float, 64> map;

// // 插入
// map.insert(1, 1.0f);
// map.insert(2, 2.0f);

// // 访问
// float value;
// if (map.find(1, value)) {
//     // 使用value
// }

// // 使用operator[]
// map[3] = 3.0f;

// // 迭代
// for (auto& entry : map) {
//     // 使用entry.key和entry.value
// }