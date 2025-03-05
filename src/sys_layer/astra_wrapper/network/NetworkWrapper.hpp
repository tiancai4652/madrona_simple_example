#pragma once

#include "../../../astra-sim/network_frontend/analytical/congestion_unaware/CongestionUnawareNetworkApi.hh"
#include "../../containers/FixedVector.hpp"

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace AstraWrapper {

// 网络配置结构
struct alignas(8) NetworkConfig {
    int rank;
    float bandwidth;
    int dims_count;
    float latency;
};

// 网络状态结构
struct alignas(8) NetworkState {
    float current_time;
    float total_data_transferred;
    int active_connections;
    bool is_congested;
};

// 网络包装类
class NetworkWrapper {
public:
    CUDA_HOST_DEVICE NetworkWrapper();
    CUDA_HOST_DEVICE ~NetworkWrapper();

    // 初始化函数
    CUDA_HOST_DEVICE void initialize(const NetworkConfig& config);
    
    // 更新函数
    CUDA_HOST_DEVICE void update(float deltaTime);
    
    // 获取状态
    CUDA_HOST_DEVICE NetworkState getState() const;
    
    // 发送数据
    CUDA_HOST_DEVICE bool sendData(int dst, uint64_t size);
    
    // 接收数据
    CUDA_HOST_DEVICE bool receiveData(int src, uint64_t size);

private:
    NetworkState state_;
    NetworkConfig config_;
    bool initialized_;
    
    // 用于追踪连接的固定大小向量
    custom::FixedVector<int, 32> active_connections_;
};

} // namespace AstraWrapper 