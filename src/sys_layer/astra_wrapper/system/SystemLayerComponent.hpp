#pragma once

#include <madrona/components.hpp>
#include "../../../../astra-sim/system/Sys.hh"
#include "../../../../astra-sim/system/AstraNetworkAPI.hh"
#include "../../../../astra-sim/system/AstraRemoteMemoryAPI.hh"
#include "../network/NetworkWrapper.hpp"

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace madsimple {

// 系统配置结构
struct alignas(8) SystemConfig {
    float simulationSpeed = 1.0f;
    float accuracy = 0.95f;
    int maxIterations = 1000;
    float peakPerformance = 1000.0f;  // GFLOPS
    float localMemoryBandwidth = 900.0f;  // GB/s
    
    // 网络相关配置
    AstraSim::NetworkConfig networkConfig;
};

// 系统状态结构
struct alignas(8) SystemState {
    bool initialized = false;
    float currentTime = 0.0f;
    int totalEvents = 0;
    float totalComputeTime = 0.0f;
    float totalCommTime = 0.0f;
};

// 系统层组件
class SystemLayerComponent {
public:
    CUDA_HOST_DEVICE
    SystemLayerComponent() : networkApi(0) {}
    
    CUDA_HOST_DEVICE
    ~SystemLayerComponent() {
        if (sys != nullptr) {
            delete sys;
        }
    }
    
    CUDA_HOST_DEVICE
    void initialize(const SystemConfig& config) {
        if (state.initialized) {
            return;
        }
        
        // 初始化网络API
        networkApi.initialize(config.networkConfig);
        
        // 创建系统实例
        std::vector<int> physical_dims = {1, 1, 1};  // 简化版本使用1D拓扑
        std::vector<int> queues_per_dim = {1};       // 每个维度一个队列
        
        sys = new AstraSim::Sys(
            0,  // system id
            "default_workload",  // workload configuration
            "default_comm_group",  // comm group configuration
            "default_system",  // system configuration
            nullptr,  // remote memory API
            &networkApi,  // network API
            physical_dims,
            queues_per_dim,
            1.0,  // injection scale
            1.0,  // comm scale
            true  // rendezvous enabled
        );
        
        state.initialized = true;
        state.currentTime = 0.0f;
        state.totalEvents = 0;
        state.totalComputeTime = 0.0f;
        state.totalCommTime = 0.0f;
    }
    
    CUDA_HOST_DEVICE
    void update(float deltaTime) {
        if (!state.initialized) {
            return;
        }
        
        // 更新网络
        networkApi.update(deltaTime);
        
        // 更新系统状态
        state.currentTime += deltaTime;
        
        // 处理系统事件
        if (sys != nullptr) {
            sys->call_events();
        }
    }
    
    CUDA_HOST_DEVICE
    const SystemState& getState() const {
        return state;
    }
    
    CUDA_HOST_DEVICE
    bool isInitialized() const {
        return state.initialized;
    }

private:
    AstraSim::Sys* sys = nullptr;
    AstraSim::CongestionUnawareNetworkApi networkApi;
    SystemState state;
};

// 定义 SystemLayerAgent 原型
struct SystemLayerAgent : public madrona::Archetype<
    SystemLayerComponent,
    SystemConfig
> {};

} // namespace madsimple 