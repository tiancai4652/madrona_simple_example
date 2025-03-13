// 一个简化的 tuple 实现，兼容 CUDA 设备端
namespace custom {

// 辅助模板：获取类型的索引
template <size_t I, typename T, typename... Ts>
struct IndexOf;

template <size_t I, typename T, typename Head, typename... Tail>
struct IndexOf<I, T, Head, Tail...> {
    static constexpr size_t value = (I == 0 && __is_same(T, Head)) ? 0 : (IndexOf<I - 1, T, Tail...>::value + 1);
};

template <size_t I, typename T, typename Last>
struct IndexOf<I, T, Last> {
    static constexpr size_t value = (I == 0 && __is_same(T, Last)) ? 0 : 1;
};

// Tuple 的前置声明
template <typename... Ts>
struct Tuple;

// Tuple 的具体实现，使用递归继承
template <typename T, typename... Rest>
struct Tuple<T, Rest...> : Tuple<Rest...> {
    T value;

    // 构造函数
    constexpr Tuple() : value(), Tuple<Rest...>() {}
    constexpr Tuple(const T& val, const Rest&... rest) : value(val), Tuple<Rest...>(rest...) {}

    // 获取元素（get 函数的辅助）
    template <size_t I>
    auto& get() {
        if constexpr (I == 0) {
            return value;
        } else {
            return Tuple<Rest...>::template get<I - 1>();
        }
    }

    template <size_t I>
    const auto& get() const {
        if constexpr (I == 0) {
            return value;
        } else {
            return Tuple<Rest...>::template get<I - 1>();
        }
    }
};

// 空 Tuple 的特化
template <>
struct Tuple<> {
    constexpr Tuple() {}
    template <size_t I>
    auto& get() {
        // 空 tuple 不应访问任何元素，这里仅为占位符
        static int dummy = 0;
        return dummy;
    }

    template <size_t I>
    const auto& get() const {
        static int dummy = 0;
        return dummy;
    }
};

// tuple_size 实现
template <typename T>
struct tuple_size;

template <typename... Ts>
struct tuple_size<Tuple<Ts...>> {
    static constexpr size_t value = sizeof...(Ts);
};

// tuple_element 实现
template <size_t I, typename T>
struct tuple_element;

template <size_t I, typename Head, typename... Tail>
struct tuple_element<I, Tuple<Head, Tail...>> {
    using type = typename tuple_element<I - 1, Tuple<Tail...>>::type;
};

template <typename Head, typename... Tail>
struct tuple_element<0, Tuple<Head, Tail...>> {
    using type = Head;
};

// get 函数的独立实现
template <size_t I, typename... Ts>
auto& get(Tuple<Ts...>& t) {
    return t.template get<I>();
}

template <size_t I, typename... Ts>
const auto& get(const Tuple<Ts...>& t) {
    return t.template get<I>();
}

// make_tuple 实现
template <typename... Ts>
constexpr auto make_tuple(Ts&&... args) {
    return Tuple<Ts...>(args...);
}

// 比较运算符（相等）
template <typename... Ts, typename... Us>
constexpr bool operator==(const Tuple<Ts...>& lhs, const Tuple<Us...>& rhs) {
    if constexpr (sizeof...(Ts) == 0 && sizeof...(Us) == 0) {
        return true;
    } else if constexpr (sizeof...(Ts) != sizeof...(Us)) {
        return false;
    } else {
        return (lhs.template get<0>() == rhs.template get<0>()) &&
               (operator==(static_cast<const Tuple<Ts...>&>(lhs), static_cast<const Tuple<Us...>&>(rhs)));
    }
}

template <>
constexpr bool operator==(const Tuple<>&, const Tuple<>&) {
    return true;
}

} // namespace mycuda

// // 示例用法
// int main() {
//     using namespace mycuda;
    
//     // 创建一个 tuple
//     auto t = make_tuple(1, 2.0f, 'c');
    
//     // 获取元素
//     int a = get<0>(t);      // 1
//     float b = get<1>(t);    // 2.0f
//     char c = get<2>(t);     // 'c'
    
//     // 检查大小
//     static_assert(tuple_size<decltype(t)>::value == 3, "Tuple size mismatch");
    
//     // 检查类型
//     static_assert(__is_same(tuple_element<0, decltype(t)>::type, int), "Type mismatch");
//     static_assert(__is_same(tuple_element<1, decltype(t)>::type, float), "Type mismatch");
//     static_assert(__is_same(tuple_element<2, decltype(t)>::type, char), "Type mismatch");
    
//     // 比较
//     auto t2 = make_tuple(1, 2.0f, 'c');
//     bool eq = (t == t2);    // true
    
//     return 0;
// }