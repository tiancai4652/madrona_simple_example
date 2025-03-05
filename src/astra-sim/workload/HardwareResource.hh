/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

// TODO: HardwareResource.hh should be moved to the system layer.

#pragma once

#include <cstdint>
#include <memory>
#include "extern/graph_frontend/chakra/et_feeder/et_feeder.h"

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace AstraSim {

// 硬件资源管理类
class HardwareResource {
public:
    CUDA_HOST_DEVICE
    HardwareResource(uint32_t num_npus);
    
    CUDA_HOST_DEVICE
    void occupy(const std::shared_ptr<Chakra::ETFeederNode> node);
    
    CUDA_HOST_DEVICE
    void release(const std::shared_ptr<Chakra::ETFeederNode> node);
    
    CUDA_HOST_DEVICE
    bool is_available(const std::shared_ptr<Chakra::ETFeederNode> node) const;

    // 资源状态
    const uint32_t num_npus;
    uint32_t num_in_flight_cpu_ops;
    uint32_t num_in_flight_gpu_comp_ops;
    uint32_t num_in_flight_gpu_comm_ops;
    
    // 性能配置
    float peak_performance;  // GFLOPS
    float memory_bandwidth; // GB/s
};

}  // namespace AstraSim
