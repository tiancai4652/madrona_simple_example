#include "NetworkWrapper.hpp"

namespace AstraWrapper {

CUDA_HOST_DEVICE
NetworkWrapper::NetworkWrapper()
    : initialized_(false)
{
    state_.current_time = 0.0f;
    state_.total_data_transferred = 0.0f;
    state_.active_connections = 0;
    state_.is_congested = false;
}

CUDA_HOST_DEVICE
NetworkWrapper::~NetworkWrapper() {
    // 清理资源
    active_connections_.clear();
}

CUDA_HOST_DEVICE
void NetworkWrapper::initialize(const NetworkConfig& config) {
    if (initialized_) {
        return;
    }
    
    config_ = config;
    initialized_ = true;
    
    // 重置状态
    state_.current_time = 0.0f;
    state_.total_data_transferred = 0.0f;
    state_.active_connections = 0;
    state_.is_congested = false;
}

CUDA_HOST_DEVICE
void NetworkWrapper::update(float deltaTime) {
    if (!initialized_) {
        return;
    }
    
    // 更新时间
    state_.current_time += deltaTime;
    
    // 更新连接状态
    state_.active_connections = active_connections_.size();
    
    // 简单的拥塞检测
    state_.is_congested = (state_.active_connections > config_.dims_count);
}

CUDA_HOST_DEVICE
NetworkState NetworkWrapper::getState() const {
    return state_;
}

CUDA_HOST_DEVICE
bool NetworkWrapper::sendData(int dst, uint64_t size) {
    if (!initialized_ || dst < 0) {
        return false;
    }
    
    // 检查是否已经有这个连接
    for (size_t i = 0; i < active_connections_.size(); i++) {
        if (active_connections_[i] == dst) {
            return false;
        }
    }
    
    // 添加新连接
    if (!active_connections_.push_back(dst)) {
        return false;
    }
    
    // 更新传输数据量
    float transfer_time = static_cast<float>(size) / config_.bandwidth;
    if (!state_.is_congested) {
        state_.total_data_transferred += size;
    }
    
    return true;
}

CUDA_HOST_DEVICE
bool NetworkWrapper::receiveData(int src, uint64_t size) {
    if (!initialized_ || src < 0) {
        return false;
    }
    
    // 在活动连接中查找并移除源连接
    for (size_t i = 0; i < active_connections_.size(); i++) {
        if (active_connections_[i] == src) {
            // 移除连接（通过将最后一个元素移到当前位置）
            if (i < active_connections_.size() - 1) {
                active_connections_[i] = active_connections_[active_connections_.size() - 1];
            }
            active_connections_.clear(); // 重置size
            return true;
        }
    }
    
    return false;
}

} // namespace AstraWrapper 