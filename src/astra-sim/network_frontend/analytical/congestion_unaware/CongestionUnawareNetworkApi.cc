/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "CongestionUnawareNetworkApi.hh"
#include <cassert>

namespace AstraSim {

CUDA_HOST_DEVICE
CongestionUnawareNetworkApi::CongestionUnawareNetworkApi(const int rank) noexcept
    : rank_(rank),
      initialized_(false),
      next_chunk_id_(0)
{
}

CUDA_HOST_DEVICE
CongestionUnawareNetworkApi::~CongestionUnawareNetworkApi() {
    active_requests_.clear();
    completed_requests_.clear();
}

CUDA_HOST_DEVICE
void CongestionUnawareNetworkApi::initialize(const NetworkConfig& config) {
    if (initialized_) {
        return;
    }
    
    config_ = config;
    initialized_ = true;
    
    // 清空请求队列
    active_requests_.clear();
    completed_requests_.clear();
}

CUDA_HOST_DEVICE
int CongestionUnawareNetworkApi::sim_send(const int dst, const uint64_t count) {
    if (!initialized_ || dst < 0) {
        return -1;
    }
    
    // 创建新的请求
    SimRequest req;
    req.tag = 0;  // 简化版本使用固定tag
    req.src = rank_;
    req.dst = dst;
    req.size = count;
    req.chunk_id = next_chunk_id_++;
    
    // 添加到活动请求列表
    if (!active_requests_.push_back(req)) {
        return -1;
    }
    
    return 0;
}

CUDA_HOST_DEVICE
int CongestionUnawareNetworkApi::sim_recv(const int src, const uint64_t count) {
    if (!initialized_ || src < 0) {
        return -1;
    }
    
    // 检查是否有匹配的已完成请求
    for (size_t i = 0; i < completed_requests_.size(); i++) {
        const auto& req = completed_requests_[i];
        if (req.src == src && req.dst == rank_ && req.size == count) {
            // 找到匹配的请求，移除它
            if (i < completed_requests_.size() - 1) {
                completed_requests_[i] = completed_requests_[completed_requests_.size() - 1];
            }
            completed_requests_.clear();  // 简化版本，直接清空
            return 0;
        }
    }
    
    return -1;
}

CUDA_HOST_DEVICE
void CongestionUnawareNetworkApi::update(float deltaTime) {
    if (!initialized_) {
        return;
    }
    
    // 简单的网络模拟：
    // 1. 计算每个请求的传输进度
    // 2. 将完成的请求移到completed_requests_
    
    float bandwidth_per_request = config_.bandwidth;
    if (active_requests_.size() > config_.dims_count) {
        // 简单的拥塞模型：如果连接数超过维度数，带宽平分
        bandwidth_per_request = config_.bandwidth / active_requests_.size();
    }
    
    // 处理所有活动请求
    for (size_t i = 0; i < active_requests_.size(); ) {
        auto& req = active_requests_[i];
        float transfer_time = static_cast<float>(req.size) / bandwidth_per_request;
        
        // 假设这个时间步内可以完成传输
        if (transfer_time <= deltaTime) {
            // 将请求移到已完成列表
            if (completed_requests_.push_back(req)) {
                // 成功移动到已完成列表，从活动列表移除
                if (i < active_requests_.size() - 1) {
                    active_requests_[i] = active_requests_[active_requests_.size() - 1];
                }
                active_requests_.clear();  // 简化版本，直接清空
            }
        } else {
            // 请求还在进行中
            i++;
        }
    }
}

} // namespace AstraSim
