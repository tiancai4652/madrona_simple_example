#pragma once

#include <cstddef>

namespace custom {

// 存储单个值的辅助类型
template<size_t Index, typename T>
struct Storage {
    T value;
    Storage() = default;
    Storage(const T& v) : value(v) {}
};

// FixedTuple的主要实现
template<typename... Types>
class FixedTuple;

// 特化：空tuple
template<>
class FixedTuple<> {
public:
    FixedTuple() = default;
    bool operator==(const FixedTuple&) const { return true; }
    bool operator!=(const FixedTuple& other) const { return !(*this == other); }
};

// 特化：单个类型
template<typename T>
class FixedTuple<T> {
    Storage<0, T> storage_;
public:
    FixedTuple() = default;
    FixedTuple(const T& t) : storage_(t) {}

    T& get() { return storage_.value; }
    const T& get() const { return storage_.value; }

    bool operator==(const FixedTuple& other) const {
        return storage_.value == other.storage_.value;
    }
    bool operator!=(const FixedTuple& other) const {
        return !(*this == other);
    }
};

// 递归计算索引的辅助类型
template<size_t I, typename T, typename... Rest>
struct TypeAtIndex {
    using type = typename TypeAtIndex<I-1, Rest...>::type;
};

template<typename T, typename... Rest>
struct TypeAtIndex<0, T, Rest...> {
    using type = T;
};

// 特化：多个类型
template<typename First, typename... Rest>
class FixedTuple<First, Rest...> {
    Storage<0, First> first_;
    FixedTuple<Rest...> rest_;

public:
    FixedTuple() = default;
    
    FixedTuple(const First& first, const Rest&... rest)
        : first_(first), rest_(rest...) {}

    template<size_t I>
    auto& get() {
        if constexpr (I == 0) {
            return first_.value;
        } else {
            return rest_.template get<I-1>();
        }
    }

    template<size_t I>
    const auto& get() const {
        if constexpr (I == 0) {
            return first_.value;
        } else {
            return rest_.template get<I-1>();
        }
    }

    // 修改比较运算符实现
    bool operator==(const FixedTuple& other) const {
        return first_.value == other.first_.value && 
               rest_ == other.rest_;
    }

    bool operator!=(const FixedTuple& other) const {
        return !(*this == other);
    }
};

// 辅助函数：创建FixedTuple
template<typename... Types>
FixedTuple<Types...> makeTuple(const Types&... args) {
    return FixedTuple<Types...>(args...);
}

// 辅助函数：获取tuple元素
template<size_t I, typename... Types>
auto& get(FixedTuple<Types...>& tuple) {
    return tuple.template get<I>();
}

template<size_t I, typename... Types>
const auto& get(const FixedTuple<Types...>& tuple) {
    return tuple.template get<I>();
}

} // namespace custom

// 使用示例：
/*
custom::FixedTuple<int, float, double> tuple(1, 2.0f, 3.0);
int first = custom::get<0>(tuple);
float second = custom::get<1>(tuple);
double third = custom::get<2>(tuple);

// 或者使用makeTuple
auto tuple2 = custom::makeTuple(1, 2.0f, 3.0);
*/


// // 创建tuple
// custom::FixedTuple<int, float, double> tuple(1, 2.0f, 3.0);

// // 使用get访问元素
// int a = custom::get<0>(tuple);
// float b = custom::get<1>(tuple);
// double c = custom::get<2>(tuple);

// // 使用makeTuple创建
// auto tuple2 = custom::makeTuple(1, 2.0f, 3.0);

// // 修改元素
// custom::get<0>(tuple) = 10;
