/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "extern/graph_frontend/chakra/et_feeder/et_feeder.h"

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace AstraSim {

class Sys;
class DataSet;

// 工作负载配置结构
struct alignas(8) WorkloadConfig {
    float simulation_speed;
    float accuracy;
    int max_iterations;
    bool trace_enabled;
    bool replay_only;
};

// 工作负载管理类
class Workload : public Callable {
  public:
    CUDA_HOST_DEVICE
    Workload(Sys* sys, const char* et_filename, const char* comm_group_filename);
    CUDA_HOST_DEVICE
    ~Workload();

    // communicator groups
    CUDA_HOST_DEVICE
    void initialize_comm_group(const char* comm_group_filename);

    // event-based simulation
    CUDA_HOST_DEVICE
    void issue_dep_free_nodes();
    CUDA_HOST_DEVICE
    void issue(std::shared_ptr<Chakra::ETFeederNode> node);
    CUDA_HOST_DEVICE
    void issue_replay(std::shared_ptr<Chakra::ETFeederNode> node);
    CUDA_HOST_DEVICE
    void issue_remote_mem(std::shared_ptr<Chakra::ETFeederNode> node);
    CUDA_HOST_DEVICE
    void issue_comp(std::shared_ptr<Chakra::ETFeederNode> node);
    CUDA_HOST_DEVICE
    void issue_comm(std::shared_ptr<Chakra::ETFeederNode> node);
    CUDA_HOST_DEVICE
    void skip_invalid(std::shared_ptr<Chakra::ETFeederNode> node);
    CUDA_HOST_DEVICE
    void call(EventType event, CallData* data);
    CUDA_HOST_DEVICE
    void fire();

    // stats
    CUDA_HOST_DEVICE
    void report();

    Chakra::ETFeeder* et_feeder;
    CommunicatorGroup* comm_group;
    HardwareResource* hw_resource;
    Sys* sys;
    
    // 配置
    WorkloadConfig config;
    
    // 集合通信映射
    std::unordered_map<int, uint64_t> collective_comm_node_id_map;
    std::unordered_map<int, DataSet*> collective_comm_wrapper_map;
    bool is_finished;
};

}  // namespace AstraSim
