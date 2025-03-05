#pragma once

#include "../../../../sys_layer/containers/FixedVector.hpp"
#include "../../../../sys_layer/containers/FixedString.hpp"

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace AstraSim {

// 基本配置结构
struct alignas(8) NetworkConfig {
    int rank;
    float bandwidth;
    int dims_count;
    float latency;
};

// 请求结构
struct alignas(8) SimRequest {
    int tag;
    int src;
    int dst;
    uint64_t size;
    uint64_t chunk_id;
};

class CongestionUnawareNetworkApi {
public:
    CUDA_HOST_DEVICE CongestionUnawareNetworkApi(const int rank) noexcept;
    CUDA_HOST_DEVICE ~CongestionUnawareNetworkApi();

    // 初始化
    CUDA_HOST_DEVICE void initialize(const NetworkConfig& config);

    // 发送数据
    CUDA_HOST_DEVICE int sim_send(const int dst, const uint64_t count);

    // 接收数据
    CUDA_HOST_DEVICE int sim_recv(const int src, const uint64_t count);

    // 更新
    CUDA_HOST_DEVICE void update(float deltaTime);

    // 获取状态
    CUDA_HOST_DEVICE int get_rank() const { return rank_; }
    CUDA_HOST_DEVICE float get_bandwidth() const { return config_.bandwidth; }
    CUDA_HOST_DEVICE bool is_congested() const { return active_requests_.size() > config_.dims_count; }

private:
    NetworkConfig config_;
    int rank_;
    bool initialized_;
    uint64_t next_chunk_id_;
    
    // 使用固定大小容器
    custom::FixedVector<SimRequest, 32> active_requests_;
    custom::FixedVector<SimRequest, 32> completed_requests_;
};

} // namespace AstraSim 