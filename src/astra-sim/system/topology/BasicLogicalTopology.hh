/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "astra-sim/system/topology/LogicalTopology.hh"

namespace AstraSim {

// 基础逻辑拓扑类
class BasicLogicalTopology : public LogicalTopology {
public:
    enum class BasicTopology { 
        Ring = 0,
        BinaryTree 
    };

    CUDA_HOST_DEVICE
    BasicLogicalTopology(BasicTopology basic_topology) {
        this->basic_topology = basic_topology;
    }

    CUDA_HOST_DEVICE virtual ~BasicLogicalTopology() = default;

    CUDA_HOST_DEVICE int get_num_of_dimensions() override {
        return 1;
    }

    CUDA_HOST_DEVICE virtual int get_num_of_nodes_in_dimension(int dimension) override = 0;

    CUDA_HOST_DEVICE BasicLogicalTopology* get_basic_topology_at_dimension(int dimension, ComType type) override {
        return this;
    }

    BasicTopology basic_topology;
};

} // namespace AstraSim
