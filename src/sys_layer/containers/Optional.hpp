#pragma once
#define CUDA_HPP

#include <new>

namespace custom {

template<typename T>
class Optional {
private:
    alignas(T) unsigned char storage_[sizeof(T)];
    bool has_value_;

public:
    // 默认构造函数
    Optional() : has_value_(false) {}

    // 从值构造
    Optional(const T& value) : has_value_(true) {
        new (&storage_) T(value);
    }

    // 移动构造
    Optional(T&& value) : has_value_(true) {
        new (&storage_) T(static_cast<T&&>(value));
    }

    // 拷贝构造函数
    Optional(const Optional& other) : has_value_(other.has_value_) {
        if (has_value_) {
            new (&storage_) T(other.value());
        }
    }

    // 移动构造函数
    Optional(Optional&& other) : has_value_(other.has_value_) {
        if (has_value_) {
            new (&storage_) T(static_cast<T&&>(other.value()));
            other.reset();
        }
    }

    // 析构函数
    ~Optional() {
        reset();
    }

    // 赋值操作符
    Optional& operator=(const Optional& other) {
        if (this != &other) {
            reset();
            has_value_ = other.has_value_;
            if (has_value_) {
                new (&storage_) T(other.value());
            }
        }
        return *this;
    }

    // 移动赋值操作符
    Optional& operator=(Optional&& other) {
        if (this != &other) {
            reset();
            has_value_ = other.has_value_;
            if (has_value_) {
                new (&storage_) T(static_cast<T&&>(other.value()));
                other.reset();
            }
        }
        return *this;
    }

    // 直接赋值操作符
    Optional& operator=(const T& value) {
        reset();
        has_value_ = true;
        new (&storage_) T(value);
        return *this;
    }

    // 移动赋值操作符
    Optional& operator=(T&& value) {
        reset();
        has_value_ = true;
        new (&storage_) T(static_cast<T&&>(value));
        return *this;
    }

    // 检查是否有值
    bool has_value() const {
        return has_value_;
    }

    // 获取值的引用
    T& value() {
        return *reinterpret_cast<T*>(&storage_);
    }

    // 获取值的const引用
    const T& value() const {
        return *reinterpret_cast<const T*>(&storage_);
    }

    // 重置Optional
    void reset() {
        if (has_value_) {
            value().~T();
            has_value_ = false;
        }
    }
};

} // namespace custom 

// custom::Optional<int> opt;           // 创建空Optional
// assert(!opt.has_value());           // 检查是否为空

// opt = 42;                           // 赋值
// assert(opt.has_value());           // 现在有值
// assert(opt.value() == 42);         // 访问值

// opt.reset();                       // 重置为空
// assert(!opt.has_value());          // 再次为空