#pragma once

namespace custom {

// 键值对结构体
template<typename K, typename V>
struct Pair {
    K first;
    V second;

    
    Pair() {}

    
    Pair(const K& k, const V& v) : first(k), second(v) {}
};

} // namespace custom 