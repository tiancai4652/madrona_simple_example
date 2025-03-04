#pragma once

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace custom {

// 键值对结构体
template<typename K, typename V>
struct Pair {
    K first;
    V second;

    CUDA_HOST_DEVICE
    Pair() {}

    CUDA_HOST_DEVICE
    Pair(const K& k, const V& v) : first(k), second(v) {}
};

} // namespace custom 