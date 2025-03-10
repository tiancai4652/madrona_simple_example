#pragma once
#define CUDA_HPP

namespace custom {

template<typename T>
class Optional {
public:
    // 默认构造函数 - 创建一个空的Optional
    __host__ __device__
    Optional() noexcept : has_value_(false) {}

    // 从值构造
    __host__ __device__
    Optional(const T& value) noexcept : has_value_(true) {
        new (&storage_) T(value);
    }

    // 移动构造
    __host__ __device__
    Optional(T&& value) noexcept : has_value_(true) {
        new (&storage_) T(static_cast<T&&>(value));
    }

    // 拷贝构造函数
    __host__ __device__
    Optional(const Optional& other) noexcept : has_value_(other.has_value_) {
        if (has_value_) {
            new (&storage_) T(other.value());
        }
    }

    // 移动构造函数
    __host__ __device__
    Optional(Optional&& other) noexcept : has_value_(other.has_value_) {
        if (has_value_) {
            new (&storage_) T(static_cast<T&&>(other.value()));
            other.reset();
        }
    }

    // 析构函数
    __host__ __device__
    ~Optional() {
        reset();
    }

    // 赋值操作符
    __host__ __device__
    Optional& operator=(const Optional& other) noexcept {
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
    __host__ __device__
    Optional& operator=(Optional&& other) noexcept {
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
    __host__ __device__
    Optional& operator=(const T& value) noexcept {
        reset();
        has_value_ = true;
        new (&storage_) T(value);
        return *this;
    }

    // 移动赋值操作符
    __host__ __device__
    Optional& operator=(T&& value) noexcept {
        reset();
        has_value_ = true;
        new (&storage_) T(static_cast<T&&>(value));
        return *this;
    }

    // 检查是否有值
    __host__ __device__
    bool has_value() const noexcept {
        return has_value_;
    }

    // 获取值的引用
    __host__ __device__
    T& value() & noexcept {
        return *reinterpret_cast<T*>(&storage_);
    }

    // 获取值的const引用
    __host__ __device__
    const T& value() const & noexcept {
        return *reinterpret_cast<const T*>(&storage_);
    }

    // 获取值的右值引用
    __host__ __device__
    T&& value() && noexcept {
        return static_cast<T&&>(*reinterpret_cast<T*>(&storage_));
    }

    // 重置Optional
    __host__ __device__
    void reset() noexcept {
        if (has_value_) {
            value().~T();
            has_value_ = false;
        }
    }

private:
    // 使用aligned_storage来存储T类型的值
    alignas(T) unsigned char storage_[sizeof(T)];
    bool has_value_;
};

} // namespace custom 

// custom::Optional<int> opt;           // 创建空Optional
// assert(!opt.has_value());           // 检查是否为空

// opt = 42;                           // 赋值
// assert(opt.has_value());           // 现在有值
// assert(opt.value() == 42);         // 访问值

// opt.reset();                       // 重置为空
// assert(!opt.has_value());          // 再次为空