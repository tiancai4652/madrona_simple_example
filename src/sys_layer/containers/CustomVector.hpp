#pragma once
#include <cstddef>

namespace mycuda {

template <typename T>
class Vector {
private:
    T* data_;           // 数据指针
    size_t size_;       // 当前元素数量
    size_t capacity_;   // 当前分配的容量

    // 扩展容量
    void resize(size_t new_capacity) {
        if (new_capacity <= capacity_) return;

        T* new_data = new T[new_capacity];
        for (size_t i = 0; i < size_; ++i) {
            new_data[i] = data_[i]; // 复制现有数据
        }

        delete[] data_; // 释放旧内存
        data_ = new_data;
        capacity_ = new_capacity;
    }

public:
    // 默认构造函数
    Vector(size_t initial_capacity = 4)
        : data_(new T[initial_capacity]), size_(0), capacity_(initial_capacity) {}

    // 拷贝构造函数
    Vector(const Vector& other) : size_(other.size_), capacity_(other.capacity_) {
        data_ = new T[capacity_];
        for (size_t i = 0; i < size_; ++i) {
            data_[i] = other.data_[i];
        }
    }

    // 析构函数
    ~Vector() {
        delete[] data_;
    }

    // 赋值运算符
    Vector& operator=(const Vector& other) {
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            data_ = new T[capacity_];
            for (size_t i = 0; i < size_; ++i) {
                data_[i] = other.data_[i];
            }
        }
        return *this;
    }

    // 基本访问接口
    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }

    T& at(size_t index) {
        if (index >= size_) return data_[0]; // 无异常处理
        return data_[index];
    }

    const T& at(size_t index) const {
        if (index >= size_) return data_[0];
        return data_[index];
    }

    T& front() { return data_[0]; }
    const T& front() const { return data_[0]; }

    T& back() { return data_[size_ - 1]; }
    const T& back() const { return data_[size_ - 1]; }

    T* data() { return data_; }
    const T* data() const { return data_; }

    // 大小和容量相关
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }

    // 修改器
    void push_back(const T& value) {
        if (size_ >= capacity_) {
            resize(capacity_ * 2); // 容量翻倍
        }
        data_[size_++] = value;
    }

    void pop_back() {
        if (size_ > 0) --size_;
    }

    void clear() { size_ = 0; }

    void reserve(size_t new_capacity) {
        if (new_capacity > capacity_) resize(new_capacity);
    }

    // 迭代器
    T* begin() { return data_; }
    const T* begin() const { return data_; }
    T* end() { return data_ + size_; }
    const T* end() const { return data_ + size_; }

    // 比较运算符
    bool operator==(const Vector& other) const {
        if (size_ != other.size_) return false;
        for (size_t i = 0; i < size_; ++i) {
            if (data_[i] != other.data_[i]) return false;
        }
        return true;
    }

    bool operator!=(const Vector& other) const { return !(*this == other); }
};

// // 示例用法
// int main() {
//     Vector<int> vec;
//     vec.push_back(1);
//     vec.push_back(2);
//     vec.push_back(3);
//     vec.push_back(4); // 触发扩容

//     int a = vec[0];     // 1
//     int b = vec.back(); // 4
//     size_t s = vec.size(); // 4

//     return 0;
// }

} // namespace mycuda