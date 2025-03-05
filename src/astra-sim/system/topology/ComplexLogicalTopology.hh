/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/topology/LogicalTopology.hh"

namespace AstraSim {

// 复杂逻辑拓扑类
class ComplexLogicalTopology : public LogicalTopology {
public:
    CUDA_HOST_DEVICE ComplexLogicalTopology() {}
    CUDA_HOST_DEVICE virtual ~ComplexLogicalTopology() = default;

    CUDA_HOST_DEVICE virtual int get_num_of_dimensions() override = 0;
    CUDA_HOST_DEVICE virtual int get_num_of_nodes_in_dimension(int dimension) override = 0;
    CUDA_HOST_DEVICE virtual BasicLogicalTopology* get_basic_topology_at_dimension(int dimension, ComType type) override = 0;
};

}  // namespace AstraSim
