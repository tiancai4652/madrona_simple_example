/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/Common.hh"

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

namespace AstraSim {

class BasicLogicalTopology;

// 逻辑拓扑基类
class LogicalTopology {
public:
    CUDA_HOST_DEVICE virtual ~LogicalTopology() = default;

    CUDA_HOST_DEVICE virtual LogicalTopology* get_topology() {
        return this;
    }
    
    CUDA_HOST_DEVICE virtual int get_num_of_dimensions() = 0;
    CUDA_HOST_DEVICE virtual int get_num_of_nodes_in_dimension(int dimension) = 0;
    CUDA_HOST_DEVICE virtual BasicLogicalTopology* get_basic_topology_at_dimension(int dimension, ComType type) = 0;
};

}  // namespace AstraSim
