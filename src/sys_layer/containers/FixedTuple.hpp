#pragma once

#include <cassert>

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace custom {

// 固定大小的元组类
template<typename... Types>
class FixedTuple;

// 特化: 空元组
template<>
class FixedTuple<> {
public:
    CUDA_HOST_DEVICE
    bool operator==(const FixedTuple&) const { return true; }
    
    CUDA_HOST_DEVICE
    bool operator<(const FixedTuple&) const { return false; }
};

// 特化: 非空元组
template<typename Head, typename... Tail>
class FixedTuple<Head, Tail...> {
private:
    Head head_;
    FixedTuple<Tail...> tail_;

public:
    // 构造函数
    CUDA_HOST_DEVICE
    FixedTuple() = default;

    CUDA_HOST_DEVICE
    FixedTuple(const Head& head, const Tail&... tail)
        : head_(head), tail_(tail...) {}

    // 获取元素
    CUDA_HOST_DEVICE
    const Head& head() const { return head_; }
    
    CUDA_HOST_DEVICE
    Head& head() { return head_; }
    
    CUDA_HOST_DEVICE
    const FixedTuple<Tail...>& tail() const { return tail_; }
    
    CUDA_HOST_DEVICE
    FixedTuple<Tail...>& tail() { return tail_; }

    // 比较操作
    CUDA_HOST_DEVICE
    bool operator==(const FixedTuple& other) const {
        return head_ == other.head_ && tail_ == other.tail_;
    }

    CUDA_HOST_DEVICE
    bool operator<(const FixedTuple& other) const {
        if (head_ < other.head_) return true;
        if (other.head_ < head_) return false;
        return tail_ < other.tail_;
    }
};

// 辅助函数: 创建元组
template<typename... Types>
CUDA_HOST_DEVICE
FixedTuple<Types...> make_fixed_tuple(const Types&... args) {
    return FixedTuple<Types...>(args...);
}

// 辅助函数: 获取元组元素
template<size_t I, typename Head, typename... Tail>
struct GetHelper {
    CUDA_HOST_DEVICE
    static auto& get(FixedTuple<Head, Tail...>& t) {
        return GetHelper<I-1, Tail...>::get(t.tail());
    }
    
    CUDA_HOST_DEVICE
    static const auto& get(const FixedTuple<Head, Tail...>& t) {
        return GetHelper<I-1, Tail...>::get(t.tail());
    }
};

template<typename Head, typename... Tail>
struct GetHelper<0, Head, Tail...> {
    CUDA_HOST_DEVICE
    static Head& get(FixedTuple<Head, Tail...>& t) {
        return t.head();
    }
    
    CUDA_HOST_DEVICE
    static const Head& get(const FixedTuple<Head, Tail...>& t) {
        return t.head();
    }
};

template<size_t I, typename... Types>
CUDA_HOST_DEVICE
auto& get(FixedTuple<Types...>& t) {
    return GetHelper<I, Types...>::get(t);
}

template<size_t I, typename... Types>
CUDA_HOST_DEVICE
const auto& get(const FixedTuple<Types...>& t) {
    return GetHelper<I, Types...>::get(t);
}

} // namespace custom 